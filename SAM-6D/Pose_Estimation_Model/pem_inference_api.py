import os
import random
import numpy as np
import torch
import importlib
import gorilla

from run_inference_custom import get_test_data, get_templates  # example placeholder

# ------------------------------------------------------------------
# 3. Sam6DEstimator class: loads model + templates once
# ------------------------------------------------------------------
class Sam6DEstimator:
    """
    Initializes and holds a loaded SAM-6D model for repeated use.
    """
    def __init__(self,
                 config_path,
                 model_name,
                 checkpoint_path,
                 gpu,
                 exp_id,
                 iteration,
                 det_score_thresh,
                 tem_path):
        """
        1) Loads config
        2) Builds and loads the model
        3) Extracts templates
        """
        self.config_path = config_path
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.gpu = gpu
        self.exp_id = exp_id
        self.iteration = iteration
        self.det_score_thresh = det_score_thresh
        self.tem_path = tem_path

        # 1) Build gorilla Config
        self.cfg = gorilla.Config.fromfile(config_path)
        exp_name = f"{model_name}_{os.path.splitext(os.path.basename(config_path))[0]}_id{exp_id}"
        log_dir = os.path.join("log", exp_name)

        self.cfg.exp_name = exp_name
        self.cfg.gpus = gpu
        self.cfg.model_name = model_name
        self.cfg.log_dir = log_dir
        self.cfg.test_iter = iteration

        # 2) Set GPU
        gorilla.utils.set_cuda_visible_devices(gpu_ids=self.cfg.gpus)
        torch.cuda.empty_cache()

        # 3) Random seeds
        random.seed(self.cfg.rd_seed)
        torch.manual_seed(self.cfg.rd_seed)

        # 4) Create the model
        print("=> creating SAM-6D model ...")
        MODEL = importlib.import_module(self.cfg.model_name)  # e.g., pose_estimation_model.py (minus .py)
        self.model = MODEL.Net(self.cfg.model)
        self.model.cuda()
        self.model.eval()

        print(f"=> loading checkpoint from {checkpoint_path} ...")
        gorilla.solver.load_checkpoint(model=self.model, filename=self.checkpoint_path)

        # 5) Extract templates once
        print("=> extracting templates ...")
        self.all_tem, self.all_tem_pts, self.all_tem_choose = get_templates(self.tem_path, self.cfg.test_dataset)
        with torch.no_grad():
            (self.all_tem_pts,
             self.all_tem_feat) = self.model.feature_extraction.get_obj_feats(
                self.all_tem,
                self.all_tem_pts,
                self.all_tem_choose
            )

class Sam6DEstimator:
    def __init__(self,
                 config_path,
                 model_name,
                 checkpoint_path,
                 gpu,
                 exp_id,
                 iteration,
                 det_score_thresh,
                 tem_path):

        self.config_path = config_path
        self.model_name  = model_name
        self.checkpoint_path = checkpoint_path
        self.gpu = gpu
        self.exp_id = exp_id
        self.iteration = iteration
        self.det_score_thresh = det_score_thresh
        self.tem_path = tem_path

        # Build gorilla Config
        self.cfg = gorilla.Config.fromfile(self.config_path)
        exp_name = f"{self.model_name}_{os.path.splitext(os.path.basename(self.config_path))[0]}_id{self.exp_id}"
        log_dir = os.path.join("log", exp_name)
        self.cfg.exp_name = exp_name
        self.cfg.gpus = self.gpu
        self.cfg.model_name = self.model_name
        self.cfg.log_dir = log_dir
        self.cfg.test_iter = self.iteration

        # Set GPU
        gorilla.utils.set_cuda_visible_devices(gpu_ids=self.cfg.gpus)
        torch.cuda.empty_cache()

        # Random seeds
        random.seed(self.cfg.rd_seed)
        torch.manual_seed(self.cfg.rd_seed)

        # Create and load model
        MODEL = importlib.import_module(self.cfg.model_name)  # e.g. pose_estimation_model
        self.model = MODEL.Net(self.cfg.model)
        self.model.cuda()
        self.model.eval()
        gorilla.solver.load_checkpoint(model=self.model, filename=self.checkpoint_path)

        # Extract templates once
        print("=> extracting templates ...")
        self.all_tem, self.all_tem_pts, self.all_tem_choose = get_templates(self.tem_path, self.cfg.test_dataset)
        with torch.no_grad():
            (self.all_tem_pts,
             self.all_tem_feat) = self.model.feature_extraction.get_obj_feats(
                self.all_tem,
                self.all_tem_pts,
                self.all_tem_choose
            )

    def run_inference_on_frame(self,
                               output_dir,
                               cad_path,
                               rgb_path,
                               depth_path,
                               cam_path,
                               seg_path):
        """
        Runs the forward pass on a single frame, returning:
        R_best (3x3), t_best (3,), and inference_time in seconds.
        """
        import time

        # Update config for current frame
        self.cfg.output_dir = output_dir
        self.cfg.cad_path   = cad_path
        self.cfg.rgb_path   = rgb_path
        self.cfg.depth_path = depth_path
        self.cfg.cam_path   = cam_path
        self.cfg.seg_path   = seg_path
        self.cfg.det_score_thresh = self.det_score_thresh

        # Load single-frame data
        input_data, img, whole_pts, model_points, detections = get_test_data(
            self.cfg.rgb_path,
            self.cfg.depth_path,
            self.cfg.cam_path,
            self.cfg.cad_path,
            self.cfg.seg_path,
            self.cfg.det_score_thresh,
            self.cfg.test_dataset
        )
        ninstance = input_data["pts"].size(0)

        # Forward pass
        start_estimation = time.time()
        with torch.no_grad():
            input_data["dense_po"] = self.all_tem_pts.repeat(ninstance, 1, 1)
            input_data["dense_fo"] = self.all_tem_feat.repeat(ninstance, 1, 1)
            out = self.model(input_data)
        inference_time = time.time() - start_estimation

        # Extract best pose
        if "pred_pose_score" in out.keys():
            pose_scores = out["pred_pose_score"] * out["score"]
        else:
            pose_scores = out["score"]
        pose_scores = pose_scores.detach().cpu().numpy()

        pred_rot   = out["pred_R"].detach().cpu().numpy()            # (ninstance, 3, 3)
        pred_trans = out["pred_t"].detach().cpu().numpy() * 1000.0   # mm
        idx_best   = np.argmax(pose_scores)

        R_best = pred_rot[idx_best]
        t_best = pred_trans[idx_best]

        return R_best, t_best, inference_time