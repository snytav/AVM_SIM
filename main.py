import os
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
import dolfinx
import basix.ufl
from dolfinx.nls.petsc import NewtonSolver

print("Initializing FEniCSx Transient Solver on Loaded Mesh Geometry...")

# 1. Read mesh layout via modern XDMF container wrappers
mesh_path = "simulation_mesh.xdmf"
if not os.path.exists(mesh_path):
    raise FileNotFoundError(f"Missing required mesh container file: {mesh_path}")

with dolfinx.io.XDMFFile(MPI.COMM_WORLD, mesh_path, "r") as xdmf_mesh:
    mesh = xdmf_mesh.read_mesh(name="Grid")

# 2. Define Modern Taylor-Hood P2-P1 Elements via Basix factory
v_el = basix.ufl.element("Lagrange", mesh.ufl_cell().cellname(), 2, shape=(mesh.geometry.dim,))
p_el = basix.ufl.element("Lagrange", mesh.ufl_cell().cellname(), 1)

mixed_el = basix.ufl.mixed_element([v_el, p_el])
V = dolfinx.fem.functionspace(mesh, mixed_el)

# Define State Functions
u = dolfinx.fem.Function(V)
u_old = dolfinx.fem.Function(V)

v, p = ufl.split(u)
v_old, p_old = ufl.split(u_old)
v_test, p_test = ufl.TestFunctions(V)

# Create a linear vector space (P1) matching the mesh degree for file output
V_velocity_out = dolfinx.fem.functionspace(
    mesh, basix.ufl.element("Lagrange", mesh.ufl_cell().cellname(), 1, shape=(mesh.geometry.dim,))
)
velocity_interpolated = dolfinx.fem.Function(V_velocity_out)
velocity_interpolated.name = "Velocity_Displacement"

# 3. Configure Time Control Parameters
t = 0.0
T = 1.0
dt = 0.05
num_steps = int(T / dt)

# 4. Build Time-Dependent Boundary Condition Functions
V_velocity, _ = V.sub(0).collapse()
V_pressure, _ = V.sub(1).collapse()

applied_pressure_func = dolfinx.fem.Function(V_pressure)
applied_pressure_func.x.array[:] = 0.0

# NumPy/Python boundary facet locator block (immune to std::bad_cast)
tdim = mesh.topology.dim
mesh.topology.create_connectivity(tdim - 1, tdim)
all_boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
facet_to_vertex = mesh.topology.connectivity(tdim - 1, 0)
local_coords = mesh.geometry.x

x_min = mesh.comm.allreduce(np.min(local_coords[:, 0]) if len(local_coords) > 0 else np.inf, op=MPI.MIN)
y_min = mesh.comm.allreduce(np.min(local_coords[:, 1]) if len(local_coords) > 0 else np.inf, op=MPI.MIN)
y_max = mesh.comm.allreduce(np.max(local_coords[:, 1]) if len(local_coords) > 0 else -np.inf, op=MPI.MAX)

inlet_list = []
wall_list = []

for facet in all_boundary_facets:
    vertices = facet_to_vertex.links(facet)
    facet_coords = local_coords[vertices]
    if np.all(np.isclose(facet_coords[:, 0], x_min)):
        inlet_list.append(facet)
    elif np.all(np.isclose(facet_coords[:, 1], y_min)) or np.all(np.isclose(facet_coords[:, 1], y_max)):
        wall_list.append(facet)

inlet_facets = np.array(inlet_list, dtype=np.int32)
wall_facets = np.array(wall_list, dtype=np.int32)

# Track topological Degrees of Freedom (DoFs)
inlet_p_dofs = dolfinx.fem.locate_dofs_topological(V.sub(1), mesh.topology.dim - 1, inlet_facets)
wall_v_dofs = dolfinx.fem.locate_dofs_topological(V.sub(0), mesh.topology.dim - 1, wall_facets)

bc_inlet_pressure = dolfinx.fem.dirichletbc(applied_pressure_func, inlet_p_dofs)
bc_wall_no_slip = dolfinx.fem.dirichletbc(np.zeros(mesh.geometry.dim, dtype=PETSc.ScalarType), wall_v_dofs, V_velocity)
bcs = [bc_inlet_pressure, bc_wall_no_slip]

# 5. Formulate Non-Linear Variational Forms
rho = dolfinx.fem.Constant(mesh, PETSc.ScalarType(1.0))
mu = dolfinx.fem.Constant(mesh, PETSc.ScalarType(0.01))

dx = ufl.dx
grad, div, inner, sym = ufl.grad, ufl.div, ufl.inner, ufl.sym

F_fluid = (
        rho * inner((v - v_old) / dt, v_test) * dx
        + rho * inner(grad(v) * v, v_test) * dx
        + 2.0 * mu * inner(sym(grad(v)), sym(grad(v_test))) * dx
        - p * div(v_test) * dx
        + p_test * div(v) * dx
)

# 6. Analytical Jacobian derivative and explicit form compilation
J_fluid = ufl.derivative(F_fluid, u)

residual_form = dolfinx.fem.form(F_fluid)
jacobian_form = dolfinx.fem.form(J_fluid)


# Interface-compliant Problem container class matching NewtonSolver requirements
class FSIProblem:
    def __init__(self, F_form, J_form, boundary_conditions):
        self.L = F_form
        self.a = J_form
        self.bcs = boundary_conditions
        self.form = lambda solver_obj: None

    def F(self, x, b_vec):
        # FIX: Typo removed (unmatched parenthesis fixed)
        indices = np.arange(len(u.x.array), dtype=np.int32)
        u.x.petsc_vec.setValues(indices, x.getArray())
        u.x.petsc_vec.assemble()
        u.x.scatter_forward()

        b_vec.set(0.0)
        dolfinx.fem.petsc.assemble_vector(b_vec, self.L)
        dolfinx.fem.petsc.apply_lifting(b_vec, [self.a], [self.bcs], x0=[u.x.petsc_vec])
        b_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        dolfinx.fem.petsc.set_bc(b_vec, self.bcs, x0=u.x.petsc_vec)

    def J(self, x, A_mat):
        A_mat.zeroEntries()
        dolfinx.fem.petsc.assemble_matrix(A_mat, self.a, bcs=self.bcs)
        A_mat.assemble()


# Instantiate problem and register inside NewtonSolver
custom_problem = FSIProblem(residual_form, jacobian_form, bcs)
solver = NewtonSolver(mesh.comm, custom_problem)

solver.atol = 1e-7
solver.rtol = 1e-7
solver.max_it = 50

# Configure high-performance direct linear MUMPS matrix options
ksp = solver.krylov_solver
ksp.setType(PETSc.KSP.Type.PREONLY)
pc = ksp.getPC()
pc.setType(PETSc.PC.Type.LU)
pc.setFactorSolverType("mumps")

# 7. Run Solver Time-Loop
xdmf_path = "fsi_results.xdmf"
if os.path.exists(xdmf_path): os.remove(xdmf_path)
if os.path.exists("fsi_results.h5"): os.remove("fsi_results.h5")

print("Executing fully-coupled Transient Fluid SNES Solver...")

with dolfinx.io.XDMFFile(mesh.comm, xdmf_path, "w") as xdmf:
    xdmf.write_mesh(mesh)

    for step in range(num_steps):
        t += dt

        current_load = 25.0 * np.sin(np.pi * t / T)
        applied_pressure_func.x.array[:] = current_load

        print(f"\n>>> Time Step {step + 1}/{num_steps} | Time: {t:.2f}s | Applied Load: {current_load:.2f} Pa")

        n_iters, converged = solver.solve(u)
        u.x.scatter_forward()

        if converged:
            print(f"    Success! Convergence verified cleanly in {n_iters} iterations.")
        else:
            print("    Warning: Solver reached iteration maximum limit.")

        u_old.x.array[:] = u.x.array[:]

        # Collapse the sub-functions out of the coupled state space
        velocity_p2 = u.sub(0).collapse()
        pressure_out = u.sub(1).collapse()
        pressure_out.name = "Pressure"

        # Interpolate high-order P2 velocity into the mesh-compatible linear space P1
        velocity_interpolated.interpolate(velocity_p2)

        # Append snapshots directly to disk
        xdmf.write_function(pressure_out, t)
        xdmf.write_function(velocity_interpolated, t)

print(f"\nSimulation complete. Results successfully exported to '{xdmf_path}'.")