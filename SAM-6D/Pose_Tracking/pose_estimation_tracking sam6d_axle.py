import os
import cv2
import numpy as np
import tempfile
import json
import sys
import time

###########################################
# 1. Ensure we can import sam6d_inference_api
###########################################
pose_estimation_model_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "Pose_Estimation_Model")
)
if pose_estimation_model_path not in sys.path:
    sys.path.append(pose_estimation_model_path)

# 1) Import Sam6DEstimator from external file
from pem_inference_api import Sam6DEstimator

# Helper imports
from ultralytics import YOLO
from helpers import *
from segmentation import generate_combined_mask

# If needed, adjust these paths according to your directory structure.
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) #/Pose_Tracking
ROOT_DIR = os.path.join(BASE_DIR, "..", "Pose_Estimation_Model") #/Pose_Estimation_Model
sys.path.append(os.path.join(ROOT_DIR, "provider"))
sys.path.append(os.path.join(ROOT_DIR, "utils"))
sys.path.append(os.path.join(ROOT_DIR, "model"))
sys.path.append(os.path.join(ROOT_DIR, "model", "pointnet2"))

# ------------------------------------------------------------------
# 1. Constants / placeholders used by SAM-6D
# ------------------------------------------------------------------
CONFIG_PATH      = os.path.join(ROOT_DIR, "config", "base.yaml")
MODEL_NAME       = "pose_estimation_model"   # e.g., "pose_estimation_model.py" minus the ".py"
CHECKPOINT_PATH  = os.path.join(ROOT_DIR, "checkpoints", "sam-6d-pem-base.pth") # e.g., "log/.../model_latest.pth"
EXP_ID           = 0                         # Some experiment ID
GPU              = "0"                       # GPU ID
ITERATION        = 600000                      # The iteration you want to test
DET_SCORE_THRESH = 0.2                       # Detection threshold used in your code


def run_pose_estimation_tracking(
    data_folder,
    mesh_for_sam6d,
    pc_for_overlay,
    cam_sam6d_json,
    cam_icp_txt,
    tem_path,
    output_dir,
    seg_model
):
    os.makedirs(output_dir, exist_ok=True)
    overlay_frames_dir = os.path.join(output_dir, "overlay_frames")
    os.makedirs(overlay_frames_dir, exist_ok=True)

    # 1) Gather frames
    rgb_folder = os.path.join(data_folder, "rgb")
    depth_folder = os.path.join(data_folder, "depth")

    rgb_files = sorted([
        os.path.join(rgb_folder, f) 
        for f in os.listdir(rgb_folder) 
        if f.endswith(".png")
    ])
    depth_files = sorted([
        os.path.join(depth_folder, f)
        for f in os.listdir(depth_folder)
        if f.endswith(".png")
    ])
    assert len(rgb_files) == len(depth_files), "Mismatch in frames."

    # 2) Intrinsics for overlay
    intrinsics_icp = load_camera_intrinsics(cam_icp_txt)
    fx, fy, cx, cy = (
        intrinsics_icp["fx"],
        intrinsics_icp["fy"],
        intrinsics_icp["cx"],
        intrinsics_icp["cy"],
    )

    # 3) Load a point cloud for overlay
    cad_o3d_dense = load_cad_model(pc_for_overlay)
    cad_points_dense = np.asarray(cad_o3d_dense.points)  # Nx3

    # 4) Setup video writer
    test_img = cv2.imread(rgb_files[0])
    H, W = test_img.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_path = os.path.join(output_dir, "pose_estimation_sam6d_each_frame.mp4")
    video_out = cv2.VideoWriter(video_path, fourcc, 30.0, (W, H))

    # ---------------------------------------------------------
    # 5) Create the Sam6DEstimator (ONE-TIME)
    # ---------------------------------------------------------
    sam6d_estimator = Sam6DEstimator(
        config_path=CONFIG_PATH,
        model_name=MODEL_NAME,
        checkpoint_path=CHECKPOINT_PATH,
        gpu=GPU,
        exp_id=EXP_ID,
        iteration=ITERATION,
        det_score_thresh=DET_SCORE_THRESH,
        tem_path=tem_path
    )

    frame_times = []

    # ---------------------------------------------------------
    # 6) Main loop: run inference per frame
    # ---------------------------------------------------------
    for i, (rgb_path, depth_path) in enumerate(zip(rgb_files, depth_files)):
        print(f"[Frame {i+1}/{len(rgb_files)}]")
        frame_start_time = time.time()

        # A) Load frame
        rgb = cv2.imread(rgb_path)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

        # B) Generate segmentation mask
        combined_mask, seg_json = generate_combined_mask(
            rgb_image=rgb,
            model=seg_model,
            frame_index=i,
            first_frame=True
        )

        # Prepare seg.json path for SAM-6D
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as temp_json_file:
            json.dump(seg_json, temp_json_file)
            seg_json_path = temp_json_file.name

        # C) Run SAM-6D inference on this frame
        R_3x3, t_3x1_mm, inference_time = sam6d_estimator.run_inference_on_frame(
            output_dir=output_dir,
            cad_path=mesh_for_sam6d,
            rgb_path=rgb_path,
            depth_path=depth_path,
            cam_path=cam_sam6d_json,
            seg_path=seg_json_path
        )
        # Cleanup
        os.remove(seg_json_path)

        # D) Overlay
        T_current = np.eye(4, dtype=np.float32)
        T_current[:3, :3] = R_3x3
        T_current[:3, 3] = t_3x1_mm

        R_mat = T_current[:3, :3]
        t_vec = T_current[:3, 3]

        processing_time = time.time() - frame_start_time
        frame_times.append(processing_time)

        transformed_points = (R_mat @ cad_points_dense.T).T + t_vec

        projected_2d = project_point_cloud_to_2d(transformed_points, intrinsics_icp)
        overlay_frame = overlay_points_on_frame(rgb.copy(), projected_2d, color=(0,0,255))

        # E) Write to video
        video_out.write(overlay_frame)
        if i == 0:
            cv2.imwrite(os.path.join(output_dir, "first_frame_estimation.png"), overlay_frame)
        cv2.imwrite(os.path.join(overlay_frames_dir, f"frame_{i:06d}.png"), overlay_frame)

        cv2.imshow("SAM-6D Overlay", overlay_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    video_out.release()
    cv2.destroyAllWindows()

    # F) Statistics
    if len(frame_times) > 0:
        avg_time = sum(frame_times) / len(frame_times)
        avg_fps = 1.0 / avg_time
        print(f"Average frame time: {avg_time:.4f}s, FPS: {avg_fps:.2f}")

        with open(os.path.join(output_dir, "runtime.txt"), "w") as f:
            f.write(f"Average frame time: {avg_time:.4f}s\n")
            f.write(f"Average FPS: {avg_fps:.2f}\n")

if __name__ == "__main__":
    # Example usage
    data_folder = "/home/martyn/Thesis/pose-tracking/data/frames/frames_axle/scene_03/"
    mesh_for_sam6d  = "/home/martyn/Thesis/pose-estimation/data/cad-models/sdit01888D53e5s6_Meshed_Decimated_Scaled.ply"
    pc_for_overlay  = "/home/martyn/Thesis/pose-estimation/data/point-clouds/point_cloud_medium.ply"
    cam_sam6d_json  = "/home/martyn/Thesis/pose-tracking/data/frames/cam_K.json"
    cam_icp_txt     = "/home/martyn/Thesis/pose-tracking/data/frames/cam_K.txt"
    tem_path        = "/home/martyn/Thesis/pose-estimation/methods/SAM-6D/SAM-6D/Data/output_sam6d_axle/templates/"
    output_dir      = "/home/martyn/Thesis/pose-tracking/results/axle/methods/sam6d/scene_03/"

    seg_model = YOLO("/home/martyn/Thesis/YOLOv8/axle_seg.pt")

    run_pose_estimation_tracking(
        data_folder=data_folder,
        mesh_for_sam6d=mesh_for_sam6d,
        pc_for_overlay=pc_for_overlay,
        cam_sam6d_json=cam_sam6d_json,
        cam_icp_txt=cam_icp_txt,
        tem_path=tem_path,
        output_dir=output_dir,
        seg_model=seg_model
    )