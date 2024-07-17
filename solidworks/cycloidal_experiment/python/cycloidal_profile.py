import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from matplotlib import path
import pandas as pd
from scipy.spatial import distance

from shapely.geometry import LineString
from shapely.affinity import translate

plt.close('all')
#%%

# Define parameters
amplification_ratio = 48
eccentricity = 0.5  # mm
pin_diameter = 3  # mm
pin_position_diameter = 62  # mm
radius = pin_position_diameter / 2
secondary_line_length = 0.5  # Fixed length for secondary lines
#%%

# Step 1: Initial circle radius
initial_radius = radius

# Step 2: Horizontal line segments
long_segment = 30.367
short_segment = 0.622

# Step 3: Big circle radius
big_circle_radius = long_segment

# Step 4: Duplicate long segments
angles = np.linspace(0, 7.5, 32)  # 0.5 degrees steps
# angles = np.linspace(0, 7.5*48, 32*48)  # 0.5 degrees steps

# Plot initial setup
plt.figure(figsize=(8, 8))
circle = plt.Circle((0, 0), initial_radius, fill=False, linestyle='dashed')
plt.gca().add_patch(circle)

# Draw primary lines
for angle in angles:
    x = long_segment * np.cos(np.deg2rad(angle))
    y = long_segment * np.sin(np.deg2rad(angle))
    plt.plot([0, x], [0, y], 'b-')

plt.xlim(-35, 35)
plt.ylim(-35, 35)
plt.gca().set_aspect('equal', adjustable='box')
plt.title('Cycloidal Drive Design - Initial Setup')
plt.show()

#%%
# Step 5: Secondary lines
secondary_lines = []

for angle in angles:
    primary_angle = angle
    secondary_angle = primary_angle * (amplification_ratio + 1)
    
    x_primary = long_segment * np.cos(np.deg2rad(primary_angle))
    y_primary = long_segment * np.sin(np.deg2rad(primary_angle))
    
    x_secondary = x_primary + secondary_line_length * np.cos(np.deg2rad(secondary_angle))
    y_secondary = y_primary + secondary_line_length * np.sin(np.deg2rad(secondary_angle))
    
    secondary_lines.append(((x_primary, y_primary), (x_secondary, y_secondary)))

# Plot primary and secondary lines
plt.figure(figsize=(8, 8))
circle = plt.Circle((0, 0), initial_radius, fill=False, linestyle='dashed')
plt.gca().add_patch(circle)

for primary, secondary in secondary_lines:
    plt.plot([0, primary[0]], [0, primary[1]], 'b-')
    plt.plot([primary[0], secondary[0]], [primary[1], secondary[1]], 'r--')

plt.xlim(-35, 35)
plt.ylim(-35, 35)
plt.gca().set_aspect('equal', adjustable='box')
plt.title('Cycloidal Drive Design - Primary and Secondary Lines with Fixed Length')
plt.show()
#%%
# Extract secondary points
sec_points = np.array([secondary[1] for secondary in secondary_lines])
tck, u = splprep([sec_points[:, 0], sec_points[:, 1]], s=0)
unew = np.linspace(0, 1.0, 1000)
out = splev(unew, tck)

# Plot spline
plt.figure(figsize=(8, 8))
plt.plot(out[0], out[1], 'g-')
plt.title('Cycloidal Drive Design - Spline Fit')
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
#%%

# Convert the spline points to a Shapely LineString
spline_line = LineString(np.column_stack((out[0], out[1])))

# Offset the spline line
offset_distance = pin_diameter / 2  # Half the pin diameter
offset_spline_line = spline_line.parallel_offset(offset_distance, 'left', join_style=2)



# Plot the original and offset splines
plt.figure(figsize=(8, 8))
plt.plot(out[0], out[1], 'g-', label='Original Spline')
offset_coords = np.array(offset_spline_line.coords)
plt.plot(offset_coords[:, 0], offset_coords[:, 1], 'r-', label='Offset Spline')
plt.title('Cycloidal Drive Design - Offset Spline')
plt.legend()
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

#%%
# Offset function

def offset_curve(vertices, offset):
    offset_vertices = []
    num_vertices = len(vertices)

    for i in range(num_vertices):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % num_vertices]

        # Calculate the direction vector between two points
        direction = np.array([p1[1] - p2[1], p2[0] - p1[0]])  # Perpendicular direction
        direction = direction / np.linalg.norm(direction)  # Normalize the vector

        # Apply the offset to both points
        offset_vertices.append(p1 + offset * direction)
        offset_vertices.append(p2 + offset * direction)

    return np.array(offset_vertices)

# Original spline points
original_points = np.vstack((out[0], out[1])).T

# Offset points
offset_points = offset_curve(original_points, pin_diameter / 2)

start_idx = 0
end_idx=-2
# Plot offset spline
plt.figure(figsize=(8, 8))
plt.plot(out[0], out[1], 'g-', label='Original Spline')
plt.plot(offset_points[start_idx:end_idx, 0], offset_points[start_idx:end_idx, 1], 'r-', label='Offset Spline')
plt.title('Cycloidal Drive Design - Offset Spline')
plt.legend()
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

#%%
# Get the coordinates of the original and offset splines
spline_coords = np.array(spline_line.coords)
offset_coords = np.array(offset_spline_line.coords)

# Function to rotate points around the origin
def rotate_points(points, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                [np.sin(angle_rad), np.cos(angle_rad)]])
    return np.dot(points, rotation_matrix.T)

# Duplicate the pattern 'amplification_ratio' times
duplicated_coords = []

z_increment = 0.001  # Increment for z dimension
z_value = 0.0
for i in range(amplification_ratio):
    angle = i * (360 / amplification_ratio)
    # rotated_spline = rotate_points(spline_coords, angle)
    rotated_offset = rotate_points(offset_coords, angle)
    
    # for point in rotated_spline:
    #     duplicated_coords.append([point[0], point[1], 0.0])  # Add z=0 for 2D points
    
    lil_clippin = 2
    for point in rotated_offset[lil_clippin:-lil_clippin]:  # Skip the last point
        duplicated_coords.append([point[0], point[1], z_value])  # Add z=0 for 2D points
        # z_value += z_increment

dup_coords = np.array(duplicated_coords)

# Plot offset spline
plt.figure(figsize=(8, 8))
# plt.plot(out[0], out[1], 'g-', label='Original Spline')
plt.plot(dup_coords[:, 0], dup_coords[:, 1], 'r-', label='Offset Spline')
plt.title('Cycloidal Drive Design - Offset Spline')
plt.legend()
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
  
# Convert to a list of strings for the .txt file
lines = ["{:.6f}, {:.6f}, {:.6f}".format(x, y, z) for x, y, z in duplicated_coords]

# Save to a .txt file
with open('cycloidal_drive_points.txt', 'w') as file:
    file.write("\n".join(lines))


#%%
# Convert to DataFrame
df = pd.DataFrame(duplicated_coords, columns=['x', 'y', 'z'])

# Save to CSV
df.to_csv('cycloidal_drive_points.csv', index=False)

# Plot the duplicated pattern
plt.figure(figsize=(8, 8))
for i in range(amplification_ratio):
    angle = i * (360 / amplification_ratio)
    rotated_spline = rotate_points(spline_coords, angle)
    rotated_offset = rotate_points(offset_coords, angle)
    
    plt.plot(rotated_spline[:, 0], rotated_spline[:, 1], 'g-', label='Original Spline' if i == 0 else "")
    plt.plot(rotated_offset[:, 0], rotated_offset[:, 1], 'r-', label='Offset Spline' if i == 0 else "")

plt.title('Cycloidal Drive Design - Circular Duplication')
plt.legend()
plt.gca().set_aspect('equal', adjustable='box')
plt.show()

#%%

#%%
# Convert lines to a list of tuples
lines_tuples = [(float(line.split(', ')[0]), float(line.split(', ')[1]), float(line.split(', ')[2])) for line in lines]

# Check for duplicates using a set
unique_points = set()
duplicates = set()
for point in lines_tuples:
    if point in unique_points:
        duplicates.add(point)
    else:
        unique_points.add(point)

# Print the number of unique points and duplicates
print(f"Total points: {len(lines_tuples)}")
print(f"Unique points: {len(unique_points)}")
print(f"Duplicate points: {len(duplicates)}")

# Print the duplicate points if any
if duplicates:
    print("Duplicate points:")
    for point in duplicates:
        print(f"{point}")

# If needed, remove duplicates from the original list
unique_lines = list(unique_points)

# Convert back to the original string format if needed
unique_lines_str = ["{:.6f}, {:.6f}, {:.6f}".format(x, y, z) for x, y, z in unique_lines]

# Optionally save the unique points back to a .txt file
with open('cycloidal_drive_unique_points.txt', 'w') as file:
    file.write("\n".join(unique_lines_str))
    
    

