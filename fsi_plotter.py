import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.tri import Triangulation

h5_filename = "fsi_results.h5"
print(f"Opening binary data target: {h5_filename}")

# 1. Parse mesh geometry from the newly generated file
with h5py.File(h5_filename, "r") as f:
    points = f["Mesh/Grid/geometry"][:]
    cells = f["Mesh/Grid/topology"][:]

    # Extract and sort all time step keys under the Pressure function node
    step_keys = sorted(list(f["Function/Pressure"].keys()), key=int)
    num_frames = len(step_keys)
    print(f"Detected {num_frames} time steps for animation.")

    # Load all steps into memory for smooth playback
    pressure_history = [f[f"Function/Pressure/{step}"][:].flatten() for step in step_keys]

# 2. Extract spatial coordinates
x = points[:, 0]
y = points[:, 1]
z = points[:, 2] if points.shape[1] > 2 else np.zeros_like(x)

# 3. Convert 4-node elements (quads) into triangles for Matplotlib
if cells.shape[1] == 4:
    tri1 = cells[:, [0, 1, 2]]
    tri2 = cells[:, [0, 2, 3]]
    triangles = np.vstack((tri1, tri2))
else:
    triangles = cells

triang = Triangulation(x, y, triangles)

# 4. Set up the Matplotlib 3D figure layout
fig = plt.figure(figsize=(14, 9))
ax = fig.add_subplot(111, projection='3d')

# Establish global color bar limits based on min/max across the entire history
global_min = np.min(pressure_history)
global_max = np.max(pressure_history)

# Render initial frame (Time step 0)
surf = ax.plot_trisurf(
    x, y, z,
    triangles=triang.triangles,
    cmap='turbo',
    linewidth=0.1,
    antialiased=True,
    edgecolors='#2c3e50'
)
surf.set_array(pressure_history[0])
surf.set_clim(global_min, global_max)

cbar = fig.colorbar(surf, ax=ax, pad=0.08, shrink=0.55)
cbar.set_label("Pressure (Pa)", fontsize=11, fontweight='bold')

ax.set_xlabel("X Space")
ax.set_ylabel("Y Space")
ax.set_zlabel("Z Elevation")
ax.view_init(elev=30, azim=-60)

title_text = ax.set_title(f"FSI Pressure Wave Animation - Step 1/{num_frames} (t = 0.05s)", fontsize=13,
                          fontweight='bold', pad=15)


# 5. Define the update loop function for the animator framework
def update_plot(frame_idx):
    ax.collections.clear()  # Clear the old frame surface

    # Re-draw the surface mesh for the current time step
    new_surf = ax.plot_trisurf(
        x, y, z,
        triangles=triang.triangles,
        cmap='turbo',
        linewidth=0.1,
        antialiased=True,
        edgecolors='#2c3e50'
    )
    new_surf.set_array(pressure_history[frame_idx])
    new_surf.set_clim(global_min, global_max)

    # Update title timestamp (dt = 0.05s per step)
    current_time = (frame_idx + 1) * 0.05
    title_text.set_text(f"FSI Pressure Wave Animation - Step {frame_idx + 1}/{num_frames} (t = {current_time:.2f}s)")
    return new_surf, title_text


# 6. Initialize and launch the interactive loop animation
print("Starting interactive time-series animation player...")
ani = animation.FuncAnimation(
    fig, update_plot, frames=num_frames, interval=150, blit=False, repeat=True
)

plt.show()