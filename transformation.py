# import numpy as np
# import cv2
# from scipy.spatial.transform import Rotation

# # File paths
# poses_ = "/home/venk/Downloads/OneDrive_2025-04-16/Isaac Sim Test Set/isaac_sim_grasp_data/banana/poses.txt"
# img_path = "/home/venk/Downloads/OneDrive_2025-04-16/Isaac Sim Test Set/isaac_sim_grasp_data/banana/0_rgb.png"
# depth_path = "/home/venk/Downloads/OneDrive_2025-04-16/Isaac Sim Test Set/isaac_sim_grasp_data/banana/0_depth.tiff"
# grasps_file = "/home/venk/Downloads/OneDrive_2025-04-16/Isaac Sim Test Set/isaac_sim_grasp_data/banana/0_rgb.png_grasps.txt"

# # Read file lines
# def read_file_lines(filepath):
#     with open(filepath, 'r') as file:
#         lines = file.readlines()
#     return lines

# # Load image to get resolution
# img = cv2.imread(img_path)
# if img is None:
#     raise ValueError(f"Could not load image at {img_path}")
# height, width = img.shape[:2]
# print(f"Image resolution: {width}x{height}")

# # Intrinsic matrix
# horizontal_aperture = 20.955  # cm
# focal_length = 50.0  # cm (50 tenths of a scene unit)
# fov = 2 * np.arctan(horizontal_aperture / (2 * focal_length))
# f_x = (width / 2) / np.tan(fov / 2)
# f_y = f_x  # Square pixels
# c_x, c_y = width / 2, height / 2
# K = np.array([[f_x, 0, c_x],
#               [0, f_y, c_y],
#               [0, 0, 1]])
# print("Intrinsic Matrix:\n", K)

# # Camera extrinsics (camera at (0, 0, 6), looking straight down)
# R_c = np.array([[1, 0, 0],
#                 [0, 0, -1],
#                 [0, 1, 0]])
# T_c = np.array([0, 0, 6])  # Camera 6 meters above the table
# print("Camera Extrinsics:\nR_c =\n", R_c, "\nT_c =", T_c)

# # Read pose and grasp data
# for k, ori in enumerate(read_file_lines(poses_)):
#     if k < 1:
#         val = ori.split()
#         world_x, world_y, world_z = float(val[1]), float(val[2]), float(val[3])
#         world_r, world_p, world_yaw = float(val[4]), float(val[5]), float(val[6])
#         print(f"world_x = {world_x}, world_y = {world_y}, world_z = {world_z}")
#         print(f"world_r = {world_r}, world_p = {world_p}, world_yaw = {world_yaw}")
        
#         # Banana's world-frame position
#         P_w = np.array([world_x, world_y, world_z])
        
#         # Transform to camera frame
#         P_c = R_c @ P_w + T_c
#         print(f"Point in camera frame: {P_c}")
        
#         # Banana's orientation in world frame
#         R_obj = Rotation.from_euler('zyx', [world_r, world_p, world_yaw], degrees=True).as_matrix()
        
#         # Transform orientation to camera frame
#         R_obj_cam = R_c @ R_obj
        
#         # Project center to image plane
#         P_img = K @ P_c
#         P_img /= P_img[2]  # Normalize
#         u, v = int(P_img[0]), int(P_img[1])
#         print(f"Projected point: ({u}, {v})")
        
#         # Define axes in banana's local frame (50 cm long for visibility)
#         axis_length = 0.5  # meters
#         axes = {
#             'x': np.array([axis_length, 0, 0]),
#             'y': np.array([0, axis_length, 0]),
#             'z': np.array([0, 0, axis_length])
#         }
        
#         # Transform axes to camera frame and project
#         for axis_name, axis_vec in axes.items():
#             # Transform axis to camera frame
#             axis_cam = P_c + R_obj_cam @ axis_vec
#             # Project to image
#             axis_img = K @ axis_cam
#             axis_img /= axis_img[2]
#             axis_u, axis_v = int(axis_img[0]), int(axis_img[1])
            
#             # Draw axis
#             color = (255, 0, 0) if axis_name == 'x' else (0, 255, 0) if axis_name == 'y' else (0, 0, 255)
#             cv2.line(img, (u, v), (axis_u, axis_v), color, 2)
        
#         # Draw the center point for reference
#         cv2.circle(img, (u, v), 5, (0, 255, 255), -1)  # Yellow dot at center
        
#         # Draw grasp center
#         for line in read_file_lines(grasps_file):
#             values = line.split()
#             center_x, center_y = float(values[-5]), float(values[-4])
#             angle = float(values[-3])
#             width = float(values[-2])
#             height = float(values[-1])
#             print(f"center_x = {center_x}, center_y = {center_y}, angle = {angle}, width = {width}, height= {height}")
#             cv2.circle(img, (int(center_x), int(center_y)), 5, (255, 0, 255), -1)  # Magenta dot at grasp center
        
#     else:
#         break

# # Save or display the image
# cv2.imwrite("output_with_axes_corrected_rotation.png", img)


"""#############################BREAK################################################################"""


# import numpy as np
# import cv2
# from scipy.spatial.transform import Rotation
# import os
# import glob
# import matplotlib.pyplot as plt

# def analyze_depth_image(depth_path):
#     depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    
#     if depth_img is None:
#         print(f"Could not read depth image: {depth_path}")
#         return None
    
#     depth_float = depth_img.astype(np.float32)
#     min_depth = np.min(depth_float)
#     max_depth = np.max(depth_float)
    
#     depth_normalized = (depth_float - min_depth) / (max_depth - min_depth)
#     object_min_depth = min_depth + (max_depth - min_depth) * 0.1  
#     object_max_depth = min_depth + (max_depth - min_depth) * 0.5  
#     object_mask = np.logical_and(
#         depth_float > object_min_depth,
#         depth_float < object_max_depth
#     ).astype(np.uint8) * 255

#     num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(object_mask)

#     if num_labels > 1:
#         largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
#         object_mask = (labels == largest_label).astype(np.uint8) * 255

#         object_center = centroids[largest_label]
#     else:
#         print("No distinct object found in the depth image")
#         object_center = None
    
#     if object_center is not None:
#         plt.plot(object_center[0], object_center[1], 'ro', markersize=10)
    
#     plt.tight_layout()
#     plt.savefig("depth_analysis.png")
#     print("Saved visualization to depth_analysis.png")
    
#     return {
#         "min_depth": min_depth,
#         "max_depth": max_depth,
#         "object_center": object_center,
#         "object_mask": object_mask,
#         "depth_normalized": depth_normalized
#     }

# poses_ = "/home/venk/Downloads/OneDrive_2025-04-16/Isaac Sim Test Set/isaac_sim_grasp_data/banana/poses.txt"
# img_path = "/home/venk/Downloads/OneDrive_2025-04-16/Isaac Sim Test Set/isaac_sim_grasp_data/banana/"
# grasps_file = "/home/venk/Downloads/OneDrive_2025-04-16/Isaac Sim Test Set/isaac_sim_grasp_data/banana/"
# poses_ = "/home/venk/Downloads/OneDrive_2025-04-16/Isaac Sim Test Set/isaac_sim_grasp_data/banana/poses.txt"

# def read_file_lines(filepath):
#     with open(filepath, 'r') as file:
#         lines = file.readlines()
#     return lines

# rgb_images = glob.glob(os.path.join(img_path, "*_rgb.png"))
# rgb_images.sort()
# depth_images = glob.glob(os.path.join(img_path, "*_depth.tiff"))
# depth_images.sort()
# grasps_files = glob.glob(os.path.join(img_path, "*_rgb.png_grasps.txt"))
# grasps_files.sort()

# for rgb_img_path, depth_img_path, grasp_file, ori in zip(rgb_images, depth_images, grasps_files, read_file_lines(poses_)):
#     print(f"Found {grasp_file} grasp file.")
#     results = analyze_depth_image(depth_img_path)
#     object_center = results["object_center"]
#     ori = ori.split()
#     world_x, world_y, world_z = float(ori[1]), float(ori[2]), float(ori[3])
#     world_r, world_p, world_yaw = float(ori[4]), float(ori[5]), float(ori[6])
#     print(f"world_x = {world_x}, world_y = {world_y}, world_z = {world_z}")
#     print(f"world_r = {world_r}, world_p = {world_p}, world_yaw = {world_yaw}")
#     rgb_img = cv2.imread(rgb_img_path)
#     if rgb_img is None:
#         print(f"Could not load RGB image at {rgb_img_path}")
#         continue
#     img_center = rgb_img.shape[1] // 2, rgb_img.shape[0] // 2
#     print(f"Image center: {img_center}")
#     for rgb_img_path, depth_img_path, grasp_file, ori in zip(rgb_images, depth_images, grasps_files, read_file_lines(poses_)):
#         print(f"Found {grasp_file} grasp file.")
#         results = analyze_depth_image(depth_img_path)
#         object_center = results["object_center"]
#         ori = ori.split()
#         world_x, world_y, world_z = float(ori[1]), float(ori[2]), float(ori[3])
#         world_r, world_p, world_yaw = float(ori[4]), float(ori[5]), float(ori[6])
#         print(f"world_x = {world_x}, world_y = {world_y}, world_z = {world_z}")
#         print(f"world_r = {world_r}, world_p = {world_p}, world_yaw = {world_yaw}")

#         rgb_img = cv2.imread(rgb_img_path)
#         if rgb_img is None:
#             print(f"Could not load RGB image at {rgb_img_path}")
#             continue

#         img_center = rgb_img.shape[1] // 2, rgb_img.shape[0] // 2
#         print(f"Image center: {img_center}")

#         for grasp in read_file_lines(grasp_file):
#             values = grasp.split()
#             center_x, center_y = float(values[-5]), float(values[-4])
#             angle = float(values[-3])
#             width = float(values[-2])
#             height = float(values[-1])
#             print(f"center_x = {center_x}, center_y = {center_y}, angle = {angle}, width = {width}, height= {height}")

#         if object_center is not None:
#             object_center_x, object_center_y = int(object_center[0]), int(object_center[1])

#             rot = Rotation.from_euler('xyz', [world_r, world_p, world_yaw], degrees=True)
#             rot_matrix = rot.as_matrix()
#             rot_matrix = np.linalg.inv(rot_matrix)  
#             axis_length = 50 

#             z_axis = rot_matrix[:, 0]  # Red (X)
#             y_axis = rot_matrix[:, 1]  # Green (Y)
#             x_axis = rot_matrix[:, 2]  # Blue (Z)

#             def project(v):
#                 return int(v[0] * axis_length), int(v[1] * axis_length)

#             x_end = (object_center_x + project(x_axis)[0], object_center_y - project(x_axis)[1])
#             y_end = (object_center_x + project(y_axis)[0], object_center_y - project(y_axis)[1])
#             z_end = (object_center_x + project(z_axis)[0], object_center_y - project(z_axis)[1])

#             cv2.arrowedLine(rgb_img, (object_center_x, object_center_y), x_end, (0, 0, 255), 2, tipLength=0.2) 
#             cv2.arrowedLine(rgb_img, (object_center_x, object_center_y), y_end, (0, 255, 0), 2, tipLength=0.2)  
#             cv2.arrowedLine(rgb_img, (object_center_x, object_center_y), z_end, (255, 0, 0), 2, tipLength=0.2) 

#         out_path = os.path.basename(rgb_img_path).replace('.png', '_with_axes.png')
#         cv2.imwrite(out_path, rgb_img)
#         print(f"Saved visualization with axes to {out_path}")


"""#############################BREAK################################################################"""
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
import os
import glob
import matplotlib.pyplot as plt

def analyze_depth_image(depth_path):
    depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    
    if depth_img is None:
        print(f"Could not read depth image: {depth_path}")
        return None
    
    depth_float = depth_img.astype(np.float32)
    min_depth = np.min(depth_float)
    max_depth = np.max(depth_float)
    
    depth_normalized = (depth_float - min_depth) / (max_depth - min_depth)
    object_min_depth = min_depth + (max_depth - min_depth) * 0.1  
    object_max_depth = min_depth + (max_depth - min_depth) * 0.5  
    object_mask = np.logical_and(
        depth_float > object_min_depth,
        depth_float < object_max_depth
    ).astype(np.uint8) * 255

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(object_mask)

    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        object_mask = (labels == largest_label).astype(np.uint8) * 255

        object_center = centroids[largest_label]
    else:
        print("No distinct object found in the depth image")
        object_center = None
    
    # if object_center is not None:
    #     plt.plot(object_center[0], object_center[1], 'ro', markersize=10)
    
    # plt.tight_layout()
    # plt.savefig("depth_analysis.png")
    # print("Saved visualization to depth_analysis.png")
    
    return {
        "min_depth": min_depth,
        "max_depth": max_depth,
        "object_center": object_center,
        "object_mask": object_mask,
        "depth_normalized": depth_normalized
    }

poses_ = "/home/venk/Downloads/OneDrive_2025-04-16/Isaac Sim Test Set/isaac_sim_grasp_data/banana/poses.txt"
img_path = "/home/venk/Downloads/OneDrive_2025-04-16/Isaac Sim Test Set/isaac_sim_grasp_data/banana/"
grasps_file = "/home/venk/Downloads/OneDrive_2025-04-16/Isaac Sim Test Set/isaac_sim_grasp_data/banana/"

def read_file_lines(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()
    return lines

rgb_images = glob.glob(os.path.join(img_path, "*_rgb.png"))
rgb_images.sort()
depth_images = glob.glob(os.path.join(img_path, "*_depth.tiff"))
depth_images.sort()
grasps_files = glob.glob(os.path.join(img_path, "*_rgb.png_grasps.txt"))
grasps_files.sort()

# Camera extrinsics
R_c = np.array([[1, 0, 0],
                [0, 0, -1],
                [0, 1, 0]])
T_c = np.array([0, 0, 6])

for idx, (rgb_img_path, depth_img_path, grasp_file, ori) in enumerate(zip(rgb_images, depth_images, grasps_files, read_file_lines(poses_))):
    print(f"Found {grasp_file} grasp file.")
    results = analyze_depth_image(depth_img_path)
    object_center = results["object_center"]
    if object_center is None:
        print("Skipping image due to no object center detected.")
        continue
    ori = ori.split()
    world_x, world_y, world_z = float(ori[1]), float(ori[2]), float(ori[3])
    world_r, world_p, world_yaw = float(ori[4]), float(ori[5]), float(ori[6])
    print(f"world_x = {world_x}, world_y = {world_y}, world_z = {world_z}")
    print(f"world_r = {world_r}, world_p = {world_p}, world_yaw = {world_yaw}")
    rgb_img = cv2.imread(rgb_img_path)
    if rgb_img is None:
        print(f"Could not load RGB image at {rgb_img_path}")
        continue
    img_center = rgb_img.shape[1] // 2, rgb_img.shape[0] // 2
    print(f"Image center: {img_center}")

    # Compute intrinsic matrix
    height, width = rgb_img.shape[:2]
    horizontal_aperture = 20.955  # cm
    focal_length = 5.0  # cm (50 tenths of a scene unit)
    fov = 2 * np.arctan(horizontal_aperture / (2 * focal_length))
    f_x = (width / 2) / np.tan(fov / 2)
    f_y = f_x  # Square pixels
    c_x, c_y = width / 2, height / 2
    K = np.array([[f_x, 0, c_x],
                  [0, f_y, c_y],
                  [0, 0, 1]])
    
    # Banana's world-frame position
    P_w = np.array([world_x, world_y, world_z])
    
    # Transform to camera frame
    P_c = R_c @ P_w + T_c
    
    # Banana's orientation in world frame
    R_obj = Rotation.from_euler('zyx', [world_r, world_p, world_yaw], degrees=True).as_matrix()
    
    # Transform orientation to camera frame
    R_obj_cam = R_c @ R_obj
    
    # Convert back to Euler angles in camera frame (for reference)
    euler_angles_cam = Rotation.from_matrix(R_obj_cam).as_euler('zyx', degrees=True)
    roll_cam, pitch_cam, yaw_cam = euler_angles_cam
    print(f"Euler angles in camera frame: roll={roll_cam:.2f}, pitch={pitch_cam:.2f}, yaw={yaw_cam:.2f} degrees")
    
    # Project the center to verify alignment (optional, for debugging)
    P_img = K @ P_c
    P_img /= P_img[2]
    projected_u, projected_v = int(P_img[0]), int(P_img[1])
    
    # Use object_center from depth analysis as the starting point
    u, v = int(object_center[0]), int(object_center[1])
    
    # Compute scaling factor to map physical axis length to pixels
    depth = P_c[2]  # z-coordinate in camera frame
    axis_length_pixels = 50  # Length in pixels in the image plane
    axis_length_meters = (axis_length_pixels / f_x) * depth  # Convert pixel length to meters at this depth
    
    # Define axes in banana's local frame (in meters)
    axes = {
        'x': np.array([axis_length_meters, 0, 0]),  # X-axis (roll)
        'y': np.array([0, axis_length_meters, 0]),  # Y-axis (pitch)
        'z': np.array([0, 0, axis_length_meters])   # Z-axis (yaw)
    }
    
    # Transform axes to camera frame and project
    for axis_name, axis_vec in axes.items():
        # Transform axis to camera frame
        axis_cam = P_c + R_obj_cam @ axis_vec
        # Project to image
        axis_img = K @ axis_cam
        axis_img /= axis_img[2]
        axis_u, axis_v = int(axis_img[0]), int(axis_img[1])
        output_path
    cv2.circle(rgb_img, (projected_u, projected_v), 5, (255, 0, 0), -1)  # Red dot at projected center
    
    # Draw grasp center
    for grasp in read_file_lines(grasp_file):
        values = grasp.split()
        center_x, center_y = float(values[-5]), float(values[-4])
        angle = float(values[-3])
        width = float(values[-2])
        height = float(values[-1])
        print(f"center_x = {center_x}, center_y = {center_y}, angle = {angle}, width = {width}, height= {height}")
        cv2.circle(rgb_img, (int(center_x), int(center_y)), 5, (255, 0, 255), -1)  # Magenta dot at grasp center
    
    # Save the image with axes
    output_path = os.path.join("/home/venk/Downloads/grasp_annotations/transformation_imgs", f"image_{idx}_with_axes.png")
    cv2.imwrite(f"image_{idx}_with_axes.png", rgb_img)
    print(f"Saved image with axes to {output_path}")