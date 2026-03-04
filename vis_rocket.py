import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

def plot_frame(ax, origin, R_matrix, label_suffix, colors=['r', 'g', 'b'], length=1.0):
    """
    Helper function to plot a 3D coordinate frame with labels.
    """
    # The columns of the rotation matrix are the basis vectors of the frame
    # expressed in world coordinates.
    x_axis = R_matrix[:, 0]
    y_axis = R_matrix[:, 1]
    z_axis = R_matrix[:, 2]

    # Plot the axes using quiver (arrows)
    ax.quiver(*origin, *x_axis, color=colors[0], length=length, normalize=True)
    ax.quiver(*origin, *y_axis, color=colors[1], length=length, normalize=True)
    ax.quiver(*origin, *z_axis, color=colors[2], length=length, normalize=True)

    # Determine label format based on suffix
    sep = "_" if label_suffix else ""
    
    # Add labels near the tip of each axis
    label_offset = 1.1 * length
    ax.text(*(origin + x_axis * label_offset), f'xg{sep}{label_suffix}', color=colors[0], fontsize=10)
    ax.text(*(origin + y_axis * label_offset), f'yg{sep}{label_suffix}', color=colors[1], fontsize=10)
    ax.text(*(origin + z_axis * label_offset), f'zg{sep}{label_suffix}', color=colors[2], fontsize=10)

def plot_rocket_body(ax, position, R_matrix, length=3.0):
    """
    Draws a simple representation of a rocket body along its local z-axis.
    """
    # Define the rocket's body vector in its own frame (along local z)
    body_vector_local = np.array([0, 0, length])
    
    # Rotate the body vector to the world frame
    body_vector_world = R_matrix.dot(body_vector_local)

    # Calculate start (tail) and end (nose) points centered on the position
    start = position - body_vector_world / 2
    end = position + body_vector_world / 2

    # Plot the rocket body as a thick line
    ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 'k', linewidth=5, alpha=0.6, label='Rocket Body')
    # Add a marker for the nose
    ax.scatter(*end, color='k', s=100, marker='^')

# --- Main Script ---

# 1. Setup the 3D plot figure
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# 2. Define and plot the World Frame (fixed at origin)
world_origin = np.array([0.0, 0.0, 0.0])
R_world = np.eye(3)  # Identity matrix means no rotation from world to world
plot_frame(ax, world_origin, R_world, 'world', length=2.0)

# 3. Define the Rocket's State
# Arbitrary position in space
rocket_position = np.array([3.0, 3.0, 3.0])

# Define orientation using a quaternion (i, j, k, real) -> (x, y, z, w)
# This quaternion represents a tilt.
q_val = np.array([0.3, 0.2, 0.1, 0.9])
# Important: Quaternions for rotation must be normalized
q_norm = q_val / np.linalg.norm(q_val)

# Create a Rotation object and get the Body-to-World rotation matrix
# scipy's convention is [x, y, z, w]
r = R.from_quat(q_norm)
R_body_to_world = r.as_matrix()

# 4. Plot the Rocket and its Body Frame
plot_rocket_body(ax, rocket_position, R_body_to_world)
plot_frame(ax, rocket_position, R_body_to_world, '', length=1.5)

# 5. Display the Rotation Matrix and Quaternion information
# Format the matrix into a readable string
matrix_str = np.array2string(R_body_to_world, formatter={'float': lambda x: f'{x: .3f}'}, separator=', ')
matrix_lines = [line.strip(" []") for line in matrix_str.split('\n')]
Formatted_matrix = "\n".join(matrix_lines)

# Create the text block
info_text = "Transformation Info:\n\n"
info_text += "Rotation Matrix (Body -> World):\n"
info_text += f"[[{Formatted_matrix}]]\n\n"
info_text += "Quaternion (i, j, k, real):\n"
info_text += f"({q_norm[0]:.3f}, {q_norm[1]:.3f}, {q_norm[2]:.3f}, {q_norm[3]:.3f})"

# Add the text box to the upper-left corner of the plot
props = dict(boxstyle='round', facecolor='white', alpha=0.8)
ax.text2D(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=11,
          verticalalignment='top', bbox=props, family='monospace')


# 6. Final Plot Settings
ax.set_xlim([-1, 6])
ax.set_ylim([-1, 6])
ax.set_zlim([-1, 6])
ax.set_xlabel('X World')
ax.set_ylabel('Y World')
ax.set_zlabel('Z World')
ax.set_title('3D Visualization of Rocket and Coordinate Frames')
ax.legend(loc='lower right')
ax.view_init(elev=30, azim=-60) # Set a good initial view angle

plt.tight_layout()
plt.show()