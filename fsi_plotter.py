import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.tri import Triangulation

h5_filename = "fsi_results.h5"
print(f"Opening binary data target: {h5_filename}")

# 1. Parse mesh geometry from the generated file
with h5py.File(h5_filename, "r") as f:
    points = f["Mesh/Grid/geometry"][:]
    cells = f["Mesh/Grid/topology"][:]

    # Extract and sort all time-step keys chronologically
    step_keys = sorted(list(f["Function/Pressure"].keys()), key=int)
    num_frames = len(step_keys)
    print(f"Detected {num_frames} time steps for animation processing.")

    # Load all steps into memory
    pressure_history = [f[f"Function/Pressure/{step}"][:].flatten() for step in step_keys]

# 2. Extract spatial coordinates
x = points[:, 0]
y = points[:, 1]
z = points[:, 2] if points.shape[1] > 2 else np.zeros_like(x)

# 3. FIX: Mathematically correct 4-node quad element subdivision into triangles
if cells.shape[1] == 4:
    print("Converting 4-node elements into 3-node triangles safely...")
    # Split each quadrilateral (0, 1, 2, 3) into two triangles: (0, 1, 2) and (0, 2, 3)
    tri1 = cells[:, [0, 1, 2]]
    tri2 = cells[:, [0, 2, 3]]
    triangles = np.vstack((tri1, tri2))
else:
    triangles = cells

triang = Triangulation(x, y, triangles)

# 4. Set up the Matplotlib 3D figure layout
fig = plt.figure(figsize=(14, 9))
ax = fig.add_subplot(111, projection='3d')

# FIX: Force your strict requested visualization scale ranges
local_min = -1e-2
local_max = 1e-2

# Render a placeholder structure setup to initialize colorbar tracking mapping
surf_dummy = ax.plot_trisurf(x, y, z, triangles=triang.triangles, cmap='turbo')
surf_dummy.set_clim(local_min, local_max)
cbar = fig.colorbar(surf_dummy, ax=ax, pad=0.08, shrink=0.55)
cbar.set_label("Pressure Range (Pa)", fontsize=11, fontweight='bold')


# 5. Define the update loop function for the animator framework
def update_plot(frame_idx):
    ax.clear()  # Clear out old geometry arrays smoothly

    # Re-apply structural axes annotations and viewpoint constraints
    ax.set_xlabel("X Space")
    ax.set_ylabel("Y Space")
    ax.set_zlabel("Z Elevation")
    ax.view_init(elev=30, azim=-60)

    # Draw the active step surface using the corrected triangulation matrix
    new_surf = ax.plot_trisurf(
        x, y, z,
        triangles=triang.triangles,
        cmap='turbo',
        linewidth=0.1,
        antialiased=True,
        edgecolors='#2c3e50'
    )
    new_surf.set_array(pressure_history[frame_idx])

    # Force the color limits onto the face properties of this frame
    new_surf.set_clim(local_min, local_max)

    current_time = (frame_idx + 1) * 0.05
    new_title = ax.set_title(
        f"FSI Pressure Wave Animation - Step {frame_idx + 1}/{num_frames} (t = {current_time:.2f}s)\n"
        f"Fixed Scale: [{local_min:.2e} to {local_max:.2e}] Pa",
        fontsize=13, fontweight='bold', pad=15
    )
    return new_surf, new_title


# Initialize the animation object loop tracker
ani = animation.FuncAnimation(
    fig, update_plot, frames=num_frames, interval=200, blit=False, repeat=True
)

# 6. Write animation out to a high-definition MP4 file using FFmpeg
output_video = "fsi_pressure_wave.mp4"
print(f"Compiling and saving video encoding to '{output_video}'... Please wait.")

writer = animation.FFMpegWriter(
    fps=5,
    metadata=dict(artist='FEniCSx Solver'),
    bitrate=2000
)

ani.save(output_video, writer=writer)
print(f"Video file saved successfully as '{os.path.abspath(output_video)}'!")

print("Opening interactive player layout window...")
plt.show()