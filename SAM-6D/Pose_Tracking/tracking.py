import os
import numpy as np
import cv2
import time  # For FPS calculation
from helpers import *

def visualize_real_time_overlay_with_fps(data_folder, cad_model_path, init_pose_path, cam_K_path):
    """
    Real-time visualization of CAD model projected onto RGB frames with FPS measurement.
    Args:
        data_folder: Path to the folder containing RGB, depth, and mask frames.
        cad_model_path: Path to the CAD model point cloud.
        init_pose_path: Path to the initial transformation matrix file.
        cam_K_path: Path to the camera intrinsic parameters file.
    """
    # Paths to RGB, depth, and mask folders
    rgb_folder = os.path.join(data_folder, "rgb")
    depth_folder = os.path.join(data_folder, "depth")
    mask_folder = os.path.join(data_folder, "masks")

    # Load camera intrinsics
    intrinsics = load_camera_intrinsics(cam_K_path)

    # Load frames
    rgb_frames = load_frames_from_folder(rgb_folder, extension="png")
    depth_frames = load_frames_from_folder(depth_folder, extension="png")
    masks = load_frames_from_folder(mask_folder, extension="png")

    # Load and preprocess the CAD model
    cad_model = load_cad_model(cad_model_path).voxel_down_sample(0.01)
    cad_points = np.asarray(cad_model.points)

    # Load the initial transformation matrix
    init_transform = np.load(init_pose_path)
    print("Initial Transformation Matrix:\n", init_transform)

    # Initialize pose and motion prediction data structures (Step A)
    current_transform = init_transform
    poses = []                 # List to store poses over time
    poses.append(current_transform)  # Store the initial pose

    # Transform CAD model using the initial pose
    transformed_cad_points = (init_transform[:3, :3] @ cad_points.T).T + init_transform[:3, 3]

    # FPS calculation variables
    start_time = time.time()
    frame_count = 0
    cumulative_projection_time = 0
    cumulative_overlay_time = 0

    for i, (rgb_frame, depth_frame, mask) in enumerate(zip(rgb_frames, depth_frames, masks)):
        frame_start_time = time.time()

        print(f"Processing frame {i + 1}/{len(rgb_frames)}...")

        # Step 1: Crop the depth frame using the mask
        cropped_depth = crop_depth_with_mask(depth_frame, mask)

        # Step 2: Convert cropped depth to point cloud
        curr_cloud = depth_to_point_cloud_optimized(cropped_depth, intrinsics).voxel_down_sample(0.01)

        if i == 0:
            print("Skipping first")
        else:
            # Step 3: Perform ICP alignment with the cropped point cloud
            current_transform = track_pose(cad_model, curr_cloud, current_transform)
            print("Current Transformation", current_transform)
        
        # Step 4: Transform CAD model to the current pose
        transformed_cad_points = (current_transform[:3, :3] @ cad_points.T).T + current_transform[:3, 3]

        # Step 5: Project the transformed CAD model points to 2D
        projection_start_time = time.time()
        projected_points = project_point_cloud_to_2d(transformed_cad_points, intrinsics)
        projection_time = time.time() - projection_start_time
        cumulative_projection_time += projection_time

        # Step 6: Overlay the projected points on the RGB frame
        overlay_start_time = time.time()
        overlay = overlay_points_on_frame(rgb_frame.copy(), projected_points, color=(0, 0, 255))
        overlay_time = time.time() - overlay_start_time
        cumulative_overlay_time += overlay_time

        # Step 7: Calculate FPS excluding projection and overlay
        processing_time = time.time() - frame_start_time - (projection_time + overlay_time)
        fps = 1 / processing_time if processing_time > 0 else 0
        cv2.putText(overlay, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show the frame
        cv2.imshow("Real-Time CAD Model Overlay", overlay)
        frame_count += 1

        # Exit if 'q' is pressed
        key = cv2.waitKey(33)  # Wait 30ms
        if key == ord('q'):
            break

    # Calculate overall FPS
    # Calculate overall FPS excluding projection and overlay times
    effective_total_time = time.time() - start_time - (cumulative_projection_time + cumulative_overlay_time)
    overall_fps = frame_count / effective_total_time
    print(f"Overall FPS (excluding projection and overlay): {overall_fps:.2f}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Define paths
    data_folder = "/home/martyn/Thesis/pose-tracking/data/frames/frames_part/scene_01/"
    cad_model_path = "/home/martyn/Thesis/pose-estimation/data/point-clouds/A6544132042_003_point_cloud_scaled.ply"
    init_pose_path = "/home/martyn/Thesis/pose-estimation/SAM-6D/SAM-6D/Data/output_sam6d/init_pose_results/init_T.npy"
    cam_K_path = "/home/martyn/Thesis/pose-tracking/data/frames/cam_K.txt"

    # Visualize real-time CAD model overlay on RGB frames
    visualize_real_time_overlay_with_fps(data_folder, cad_model_path, init_pose_path, cam_K_path)

