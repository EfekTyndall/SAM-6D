import os
import sys
import random
import torch
import numpy as np
import time

import gorilla  # Ensure gorilla is installed in your environment

# If needed, adjust these paths according to your directory structure.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, "..", "Pose_Estimation_Model")
sys.path.append(os.path.join(ROOT_DIR, "provider"))
sys.path.append(os.path.join(ROOT_DIR, "utils"))
sys.path.append(os.path.join(ROOT_DIR, "model"))
sys.path.append(os.path.join(BASE_DIR, "model", "pointnet2"))

# Import your utility functions for data loading
from run_inference_custom import get_test_data, get_templates


##########################
# HARD-CODED PARAMETERS
##########################
# You said these do not change:
GPU = "0"
MODEL_NAME = "pose_estimation_model"
CONFIG_PATH = os.path.join(BASE_DIR, "config", "base.yaml")
ITERATION = 600000
EXP_ID = 0
CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoints", "sam-6d-pem-base.pth")
DET_SCORE_THRESH = 0.2


def sam6d_inference_single_frame(
    output_dir: str,
    cad_path: str,
    rgb_path: str,
    depth_path: str,
    cam_path: str,
    seg_path: str,
    tem_path: str
):
    """
    Runs SAM-6D inference on a single frame, returning the best R, t (in mm),
    with certain parameters (GPU, model, config, iteration, checkpoint, etc.)
    hardcoded.
    """

    # ------------------------------------------------------------
    # 1) Build a gorilla Config object from the hardcoded config
    # ------------------------------------------------------------
    cfg = gorilla.Config.fromfile(CONFIG_PATH)

    # Build experiment name and log_dir from the hardcoded model/config/exp_id
    exp_name = f"{MODEL_NAME}_{os.path.splitext(os.path.basename(CONFIG_PATH))[0]}_id{EXP_ID}"
    log_dir = os.path.join("log", exp_name)

    cfg.exp_name = exp_name
    cfg.gpus = GPU
    cfg.model_name = MODEL_NAME
    cfg.log_dir = log_dir
    cfg.test_iter = ITERATION

    # ------------------------------------------------------------
    # 2) Fill in the user-supplied paths
    # ------------------------------------------------------------
    cfg.output_dir = output_dir
    cfg.cad_path = cad_path
    cfg.rgb_path = rgb_path
    cfg.depth_path = depth_path
    cfg.cam_path = cam_path
    cfg.seg_path = seg_path
    cfg.det_score_thresh = DET_SCORE_THRESH

    # Set the GPU device
    gorilla.utils.set_cuda_visible_devices(gpu_ids=cfg.gpus)

    # ------------------------------------------------------------
    # 3) Set random seeds if needed
    # ------------------------------------------------------------
    random.seed(cfg.rd_seed)
    torch.manual_seed(cfg.rd_seed)

    # ------------------------------------------------------------
    # 4) Create and load the SAM-6D model
    # ------------------------------------------------------------
    print("=> creating model ...")
    import importlib
    MODEL = importlib.import_module(cfg.model_name)  # e.g., pose_estimation_model.py
    model = MODEL.Net(cfg.model)
    model.cuda()
    model.eval()

    print("=> loading checkpoint ...")
    gorilla.solver.load_checkpoint(model=model, filename=CHECKPOINT_PATH)

    # ------------------------------------------------------------
    # 5) Load templates
    # ------------------------------------------------------------
    print("=> extracting templates ...")
    all_tem, all_tem_pts, all_tem_choose = get_templates(tem_path, cfg.test_dataset)

    with torch.no_grad():
        all_tem_pts, all_tem_feat = model.feature_extraction.get_obj_feats(
            all_tem, all_tem_pts, all_tem_choose
        )

    # ------------------------------------------------------------
    # 6) Load single-frame data
    # ------------------------------------------------------------
    print("=> loading input data ...")
    input_data, img, whole_pts, model_points, detections = get_test_data(
        cfg.rgb_path,
        cfg.depth_path,
        cfg.cam_path,
        cfg.cad_path,
        cfg.seg_path,
        cfg.det_score_thresh,
        cfg.test_dataset
    )
    ninstance = input_data["pts"].size(0)

    # ------------------------------------------------------------
    # 7) Forward pass (inference)
    # ------------------------------------------------------------
    print("=> running model inference ...")
    # Measure SAM-6D inference time
    start_estimation = time.time()
    with torch.no_grad():
        input_data["dense_po"] = all_tem_pts.repeat(ninstance, 1, 1)
        input_data["dense_fo"] = all_tem_feat.repeat(ninstance, 1, 1)
        out = model(input_data)

    # ------------------------------------------------------------
    # 8) Extract best pose
    # ------------------------------------------------------------
    if "pred_pose_score" in out.keys():
        pose_scores = out["pred_pose_score"] * out["score"]
    else:
        pose_scores = out["score"]
    pose_scores = pose_scores.detach().cpu().numpy()

    pred_rot = out["pred_R"].detach().cpu().numpy()          # shape: (ninstance, 3, 3)
    pred_trans = out["pred_t"].detach().cpu().numpy() * 1000  # in mm

    idx_best = np.argmax(pose_scores)
    R_best = pred_rot[idx_best]    # (3,3)
    t_best = pred_trans[idx_best]  # (3,)

    estimation_time = time.time() - start_estimation

    print(
        "=> inference done. "
        f"Best pose has score={pose_scores[idx_best]:.4f}\n"
        f"R:\n{R_best}\nt (mm): {t_best}"
    )

    return R_best, t_best, estimation_time