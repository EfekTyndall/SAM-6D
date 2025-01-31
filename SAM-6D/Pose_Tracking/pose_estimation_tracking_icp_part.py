import os
import sys
import time
import cv2
import numpy as np
import open3d as o3d
from ultralytics import YOLO

###########################################
# 1. Ensure we can import sam6d_inference_api
###########################################
pose_estimation_model_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "Pose_Estimation_Model")
)
if pose_estimation_model_path not in sys.path:
    sys.path.append(pose_estimation_model_path)

# Import your inline SAM-6D function
from run_pem import sam6d_inference_single_frame

###########################################
# 2. Import existing helpers from helpers.py
###########################################
from helpers import (
    load_camera_intrinsics,           # For .txt intrinsics (ICP)
    crop_depth_with_mask,
    depth_to_point_cloud_optimized,
    load_cad_model,
    track_pose,
    project_point_cloud_to_2d,
    overlay_points_on_frame
)

###########################################
# 3. Simple helper: R,t -> 4x4
###########################################
def pose_to_matrix(R, t):
    """
    R: (3,3), t: (3,) => 4x4
    """
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

###########################################
# 4. Integrated pipeline with video saving
###########################################
def run_pose_estimation_tracking(
    data_folder,
    mesh_for_sam6d,         # <--- CAD mesh path for SAM-6D
    pc_for_icp,             # <--- CAD point cloud path for ICP
    cam_sam6d_json,         # <--- JSON intrinsics used by SAM-6D
    cam_icp_txt,            # <--- .txt intrinsics used by ICP
    seg_json_first_frame,   # <--- segmentation/detection JSON for first frame
    tem_path,               # <--- path to templates
    output_dir,
    scale_to_meters=True
):
    """
    1) Use SAM-6D (with a mesh) for the first frame's pose estimation.
    2) Use ICP with a point cloud for subsequent frames.
    3) Overlays + real-time display, then save as mp4
    4) Save first-frame overlay and runtime stats.
    Args:
        data_folder: folder with subfolders rgb/, depth/, masks/, each containing .png
        mesh_for_sam6d: path to the mesh CAD model for SAM-6D
        pc_for_icp: path to the point cloud (ply, pcd) for ICP
        cam_sam6d_json: path to JSON intrinsics used by SAM-6D
        cam_icp_txt: path to .txt intrinsics for ICP
        seg_json_first_frame: path to the segmentation/detection JSON for the first frame
        tem_path: path to template folder (for SAM-6D)
        output_dir: where SAM-6D might store intermediate results
        scale_to_meters: bool - convert mm → m for ICP
    """

    os.makedirs(output_dir, exist_ok=True)  # Ensure output dir exists

    # Additionally, create a subfolder for per-frame overlays
    overlay_frames_dir = os.path.join(output_dir, "overlay_frames")
    os.makedirs(overlay_frames_dir, exist_ok=True)

    # === 1) Gather frames ===
    rgb_folder = os.path.join(data_folder, "rgb")
    depth_folder = os.path.join(data_folder, "depth")
    masks_folder = os.path.join(data_folder, "masks")

    # Sort by filename
    rgb_files = sorted([os.path.join(rgb_folder, f) for f in os.listdir(rgb_folder) if f.endswith(".png")])
    depth_files = sorted([os.path.join(depth_folder, f) for f in os.listdir(depth_folder) if f.endswith(".png")])
    mask_files = sorted([os.path.join(masks_folder, f) for f in os.listdir(masks_folder) if f.endswith(".png")])

    # Check same length
    assert len(rgb_files) == len(depth_files) == len(mask_files), "Mismatch in frames."

    # === 2) ICP intrinsics (from .txt) ===
    intrinsics_icp = load_camera_intrinsics(cam_icp_txt)
    fx, fy, cx, cy = intrinsics_icp["fx"], intrinsics_icp["fy"], intrinsics_icp["cx"], intrinsics_icp["cy"]

    # === 3) Load point cloud for Overlay and ICP===
    cad_o3d_dense = load_cad_model(pc_for_icp)
    cad_points_dense = np.asarray(cad_o3d_dense.points)  # Nx3
    cad_o3d_icp = cad_o3d_dense.voxel_down_sample(0.01)

    # === 5) Set up video writer (30 FPS) ===
    test_img   = cv2.imread(rgb_files[0])  # shape (H, W, 3)
    H, W       = test_img.shape[:2]
    fourcc     = cv2.VideoWriter_fourcc(*'mp4v')
    video_path = os.path.join(output_dir, "pose_estimation_tracking_result.mp4")
    video_out  = cv2.VideoWriter(video_path, fourcc, 30.0, (W, H))

    # === 4) SAM-6D on the first frame (using a mesh) ===
    # We take the first frame
    first_rgb_path = rgb_files[0]
    first_depth_path = depth_files[0]

    print("[SAM-6D] Running on first frame:", first_rgb_path)
    R_3x3, t_3x1_mm, estimation_time = sam6d_inference_single_frame(
        output_dir=output_dir,
        cad_path=mesh_for_sam6d,        # pass the mesh path here
        rgb_path=first_rgb_path,
        depth_path=first_depth_path,
        cam_path=cam_sam6d_json,       # JSON for SAM-6D
        seg_path=seg_json_first_frame,
        tem_path=tem_path
    )
    print("[SAM-6D] R:\n", R_3x3)
    print("[SAM-6D] t (mm):", t_3x1_mm)

    # Build initial transform
    T_init = pose_to_matrix(R_3x3, t_3x1_mm)
    current_transform = T_init.copy()

    # For performance stats
    frame_times = []  # store each ICP time
    #cumulative_projection_time = 0.0
    #cumulative_overlay_time    = 0.0

    # === 6) Main loop ===
    for i, (rgb_path, depth_path, mask_path) in enumerate(zip(rgb_files, depth_files, mask_files)):
        frame_start_time = time.time()

        print(f"Processing frame {i + 1}/{len(rgb_files)}...")

        # Load frame
        rgb = cv2.imread(rgb_path)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        mask = (mask > 0)

        ###########################################
        # Generate binary mask
        ###########################################
        # Input : rgb
        # Output: mask
        """
        1.  If possible, model needs to be first loaded outside the function,
            to avoid loading the model every time in the loop
        2.  Run inference on input: rgb
        3.  Create binary mask from inference result
        4.  Output: mask
        """

        # Crop + depth -> point cloud
        cropped_depth = crop_depth_with_mask(depth, mask)
        cloud_o3d = depth_to_point_cloud_optimized(cropped_depth, intrinsics_icp).voxel_down_sample(0.01)

        # ICP for frames > 0
        if i == 0:
            print("[Frame 0] Using SAM-6D pose, skip ICP.")
        else:
            current_transform = track_pose(cad_o3d_icp, cloud_o3d, current_transform)
            print(f"[Frame {i}] ICP transform:\n{current_transform}")

        # Transform the point cloud CAD model
        R_mat = current_transform[:3, :3]
        t_vec = current_transform[:3, 3]

        # Processing time of each frame
        processing_time = time.time() - frame_start_time
        frame_times.append(processing_time)

        # === 7) Visualization ===
        transformed_points = (R_mat @ cad_points_dense.T).T + t_vec
        # Project + overlay
        projected_2d = project_point_cloud_to_2d(transformed_points, intrinsics_icp)
        overlay_frame = overlay_points_on_frame(rgb.copy(), projected_2d, color=(0,0,255))
        
        #fps = 1.0 / processing_time if processing_time > 0 else 0
        #cv2.putText(overlay_frame, f"FPS: {fps:.2f}", (10,30),
        #            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        
        # Write to video
        video_out.write(overlay_frame)

        # For convenience, save the first frame overlay to a PNG
        if i == 0:
            first_frame_img_path = os.path.join(output_dir, "first_frame_estimation.png")
            cv2.imwrite(first_frame_img_path, overlay_frame)
            print(f"Saved first frame overlay to: {first_frame_img_path}")
        
        # === Save each frame as PNG ===
        frame_png = os.path.join(overlay_frames_dir, f"frame_{i:06d}.png")
        cv2.imwrite(frame_png, overlay_frame)

        # Show live
        cv2.imshow("Pose Estimation + Tracking", overlay_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_out.release()
    cv2.destroyAllWindows()

    # === 8) Calculate final stats ===
    if len(frame_times) > 0:
        avg_frame_time = sum(frame_times) / len(frame_times)
    avg_fps = 1 / avg_frame_time

    print(f"\nEstimation time on first frame: {estimation_time:.4f} seconds")
    print(f"Average tracking time (subsequent frames): {avg_frame_time:.4f} seconds")
    print(f"Average FPS: {avg_fps:.2f}")

    # === 8) Save runtime stats to .txt
    stats_path = os.path.join(output_dir, "runtime.txt")
    with open(stats_path, "w") as f:
        f.write(f"Estimation time on first frame: {estimation_time:.4f} seconds\n")
        f.write(f"Average tracking time (subsequent frames): {avg_frame_time:.4f} seconds\n")
        f.write(f"Average FPS (all frames): {avg_fps:.2f}\n")

    print(f"Saved runtime stats to: {stats_path}")

###########################################
# 5. Example main
###########################################
if __name__ == "__main__":
    # Adjust these paths to your files
    data_folder = "/home/martyn/Thesis/pose-tracking/data/frames/frames_part/scene_01/"

    mesh_for_sam6d  = "/home/martyn/Thesis/pose-estimation/data/cad-models/part.ply"   # or .ply, as a mesh
    pc_for_icp      = "/home/martyn/Thesis/pose-estimation/data/point-clouds/A6544132042_003_point_cloud_scaled.ply"    # point cloud for tracking

    cam_sam6d_json  = "/home/martyn/Thesis/pose-tracking/data/frames/cam_K.json"       # JSON used by SAM-6D
    cam_icp_txt     = "/home/martyn/Thesis/pose-tracking/data/frames/cam_K.txt"          # .txt used by your tracking pipeline

    seg_json_first_frame = os.path.join(data_folder, "first_frame_detection.json")
    tem_path            = "/home/martyn/Thesis/pose-estimation/SAM-6D/SAM-6D/Data/output_sam6d/templates/"
    output_dir          = "/home/martyn/Thesis/pose-tracking/results/part/sam6d/scene_01/"

    run_pose_estimation_tracking(
        data_folder=data_folder,
        mesh_for_sam6d=mesh_for_sam6d,
        pc_for_icp=pc_for_icp,
        cam_sam6d_json=cam_sam6d_json,
        cam_icp_txt=cam_icp_txt,
        seg_json_first_frame=seg_json_first_frame,
        tem_path=tem_path,
        output_dir=output_dir,
        scale_to_meters=True
    )
