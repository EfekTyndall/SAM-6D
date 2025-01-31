import os, sys
import numpy as np
import shutil
from tqdm import tqdm
import time
import torch
from PIL import Image
import logging
import os, sys
import os.path as osp
from hydra import initialize, compose
# set level logging
logging.basicConfig(level=logging.INFO)
import logging
import trimesh
import numpy as np
from hydra.utils import instantiate
import argparse
import glob
from omegaconf import DictConfig, OmegaConf
from torchvision.utils import save_image
import torchvision.transforms as T
import cv2
import imageio.v2 as imageio
import distinctipy
from skimage.feature import canny
from skimage.morphology import binary_dilation
from segment_anything.utils.amg import rle_to_mask
import time

from utils.poses.pose_utils import get_obj_poses_from_template_level, load_index_level_in_level2
from utils.bbox_utils import CropResizePad
from model.utils import Detections, convert_npz_to_json
from model.loss import Similarity
from utils.inout import load_json, save_json_bop23

inv_rgb_transform = T.Compose(
        [
            T.Normalize(
                mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
            ),
        ]
    )

def visualize(rgb, detections, save_path="tmp.png"):
    img = rgb.copy()
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    colors = distinctipy.get_colors(len(detections))
    alpha = 0.33

    best_score = 0.
    for mask_idx, det in enumerate(detections):
        if best_score < det['score']:
            best_score = det['score']
            best_det = detections[mask_idx]

    mask = rle_to_mask(best_det["segmentation"])
    edge = canny(mask)
    edge = binary_dilation(edge, np.ones((2, 2)))
    obj_id = best_det["category_id"]
    temp_id = obj_id - 1

    r = int(255*colors[temp_id][0])
    g = int(255*colors[temp_id][1])
    b = int(255*colors[temp_id][2])
    img[mask, 0] = alpha*r + (1 - alpha)*img[mask, 0]
    img[mask, 1] = alpha*g + (1 - alpha)*img[mask, 1]
    img[mask, 2] = alpha*b + (1 - alpha)*img[mask, 2]   
    img[edge, :] = 255
    
    img = Image.fromarray(np.uint8(img))
    img.save(save_path)
    prediction = Image.open(save_path)
    
    # concat side by side in PIL
    img = np.array(img)
    concat = Image.new('RGB', (img.shape[1] + prediction.size[0], img.shape[0]))
    concat.paste(rgb, (0, 0))
    concat.paste(prediction, (img.shape[1], 0))
    return concat


def visualize_all_detections(rgb, detections, save_path="all_detections.png"):
    """
    Overlays *all* detections on the given image, each in a distinct color,
    and saves the result to save_path.
    """
    img = np.array(rgb.copy())  # keep the original
    # convert to grayscale for background style
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # convert back to 3-channel
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    # Get N distinct colors
    colors = distinctipy.get_colors(len(detections))
    alpha = 0.35
    
    # Loop over all detections
    for i, det in enumerate(detections):
        # Convert RLE -> mask
        mask = rle_to_mask(det["segmentation"])
        edge = canny(mask)
        edge = binary_dilation(edge, np.ones((2, 2)))
        
        # Distinct color for each detection
        color = colors[i]
        r = int(255 * color[0])
        g = int(255 * color[1])
        b = int(255 * color[2])
        
        # Apply alpha-blending within the mask
        img[mask, 0] = alpha * r + (1 - alpha) * img[mask, 0]
        img[mask, 1] = alpha * g + (1 - alpha) * img[mask, 1]
        img[mask, 2] = alpha * b + (1 - alpha) * img[mask, 2]

        # For boundary
        img[edge, :] = 255

    # Convert to PIL for saving
    viz_img = Image.fromarray(img.astype(np.uint8))
    viz_img.save(save_path)
    return viz_img


def visualize_all_detections_with_scores(rgb, detections, save_path="all_detections_scored.png"):
    img = np.array(rgb.copy())  # in RGB
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    colors = distinctipy.get_colors(len(detections))
    alpha = 0.35

    for i, det in enumerate(detections):
        mask = rle_to_mask(det["segmentation"])
        edge = canny(mask)
        edge = binary_dilation(edge, np.ones((2, 2)))

        # Distinct color
        color = colors[i]
        r = int(255 * color[0])
        g = int(255 * color[1])
        b = int(255 * color[2])

        img[mask, 0] = alpha*r + (1 - alpha)*img[mask, 0]
        img[mask, 1] = alpha*g + (1 - alpha)*img[mask, 1]
        img[mask, 2] = alpha*b + (1 - alpha)*img[mask, 2]
        img[edge, :] = 255

        # Optionally, draw bounding box if it’s in the detection (BOP format might have it).
        # Suppose 'det["bbox"]' is [x, y, w, h]:
        x, y, w, h = det["bbox"]
        cv2.rectangle(img, (int(x), int(y)), (int(x+w), int(y+h)), (r, g, b), 2)

        # Overlay the *semantic* score or the *final* score
        # Make sure 'semantic_score' is either stored in det or in a separate place
        # If you only have "score" or something, adapt accordingly:
        score_text = f"Score: {det['score']:.3f}"
        cv2.putText(
            img,
            score_text,
            (int(x), int(y) - 5),  # above top-left corner of box
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,  # font scale
            (r, g, b),
            1,    # line thickness
            cv2.LINE_AA
        )

    viz_img = Image.fromarray(img.astype(np.uint8))
    viz_img.save(save_path)
    return viz_img


def batch_input_data(depth_path, cam_path, device):
    batch = {}
    cam_info = load_json(cam_path)
    depth = np.array(imageio.imread(depth_path)).astype(np.int32)
    cam_K = np.array(cam_info['cam_K']).reshape((3, 3))
    depth_scale = np.array(cam_info['depth_scale'])

    batch["depth"] = torch.from_numpy(depth).unsqueeze(0).to(device)
    batch["cam_intrinsic"] = torch.from_numpy(cam_K).unsqueeze(0).to(device)
    batch['depth_scale'] = torch.from_numpy(depth_scale).unsqueeze(0).to(device)
    return batch

def run_inference(segmentor_model, output_dir, cad_path, rgb_path, depth_path, cam_path, stability_score_thresh):
    total_start = time.time()  # Start total runtime measurement

    # Configuration initialization
    config_start = time.time()
    with initialize(version_base=None, config_path="configs"):
        cfg = compose(config_name='run_inference.yaml')

    if segmentor_model == "sam":
        with initialize(version_base=None, config_path="configs/model"):
            cfg.model = compose(config_name='ISM_sam.yaml')
        cfg.model.segmentor_model.stability_score_thresh = stability_score_thresh
    elif segmentor_model == "fastsam":
        with initialize(version_base=None, config_path="configs/model"):
            cfg.model = compose(config_name='ISM_fastsam.yaml')
    else:
        raise ValueError("The segmentor_model {} is not supported now!".format(segmentor_model))   
    
    config_end = time.time()
    logging.info(f"Configuration initialization time: {config_end - config_start:.4f} seconds")
    
    # Model initialization
    model_start = time.time()
    logging.info("Initializing model")
    model = instantiate(cfg.model)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.descriptor_model.model = model.descriptor_model.model.to(device)
    model.descriptor_model.model.device = device
    # if there is predictor in the model, move it to device
    if hasattr(model.segmentor_model, "predictor"):
        model.segmentor_model.predictor.model = (
            model.segmentor_model.predictor.model.to(device)
        )
    else:
        model.segmentor_model.model.setup_model(device=device, verbose=True)
    
    model_end = time.time()
    logging.info(f"Model initialization time: {model_end - model_start:.4f} seconds")
    logging.info(f"Moving models to {device} done!")
        
    
    # Template initialization
    template_start = time.time()
    logging.info("Initializing template")
    template_dir = "/home/martyn/Thesis/pose-estimation/methods/SAM-6D/SAM-6D/Data/output_sam6d/templates/"
    num_templates = len(glob.glob(f"{template_dir}/*.npy"))
    boxes, masks, templates = [], [], []
    for idx in range(num_templates):
        image = Image.open(os.path.join(template_dir, 'rgb_'+str(idx)+'.png'))
        mask = Image.open(os.path.join(template_dir, 'mask_'+str(idx)+'.png'))
        boxes.append(mask.getbbox())

        image = torch.from_numpy(np.array(image.convert("RGB")) / 255).float()
        mask = torch.from_numpy(np.array(mask.convert("L")) / 255).float()
        image = image * mask[:, :, None]
        templates.append(image)
        masks.append(mask.unsqueeze(-1))
        
    templates = torch.stack(templates).permute(0, 3, 1, 2)
    masks = torch.stack(masks).permute(0, 3, 1, 2)
    boxes = torch.tensor(np.array(boxes))
    
    processing_config = OmegaConf.create(
        {
            "image_size": 224,
        }
    )
    proposal_processor = CropResizePad(processing_config.image_size)
    templates = proposal_processor(images=templates, boxes=boxes).to(device)
    masks_cropped = proposal_processor(images=masks, boxes=boxes).to(device)

    model.ref_data = {}
    model.ref_data["descriptors"] = model.descriptor_model.compute_features(
                    templates, token_name="x_norm_clstoken"
                ).unsqueeze(0).data
    model.ref_data["appe_descriptors"] = model.descriptor_model.compute_masked_patch_feature(
                    templates, masks_cropped[:, 0, :, :]
                ).unsqueeze(0).data
    
    template_end = time.time()
    logging.info(f"Template initialization time: {template_end - template_start:.4f} seconds")
    
    # run inference segmentation
    seg_start = time.time()
    rgb = Image.open(rgb_path).convert("RGB")
    detections = model.segmentor_model.generate_masks(np.array(rgb))
    detections = Detections(detections)
    seg_end = time.time()
    logging.info(f"Segmentation runtime: {seg_end - seg_start:.4f} seconds")
    
    # Descriptor matching
    desc_start = time.time()
    query_decriptors, query_appe_descriptors = model.descriptor_model.forward(np.array(rgb), detections)
    
    desc_end = time.time()
    logging.info(f"Descriptor matching time: {desc_end - desc_start:.4f} seconds")

    # matching descriptors/Semantic matching
    semantic_start = time.time()
    (
        idx_selected_proposals,
        pred_idx_objects,
        semantic_score,
        best_template,
    ) = model.compute_semantic_score(query_decriptors)
    
    semantic_end = time.time()
    logging.info(f"Semantic matching time: {semantic_end - semantic_start:.4f} seconds")

    # update detections
    detections.filter(idx_selected_proposals)
    query_appe_descriptors = query_appe_descriptors[idx_selected_proposals, :]

    # compute the appearance score
    appe_scores, ref_aux_descriptor= model.compute_appearance_score(best_template, pred_idx_objects, query_appe_descriptors)

    # compute the geometric score
    geo_start = time.time()
    batch = batch_input_data(depth_path, cam_path, device)
    template_poses = get_obj_poses_from_template_level(level=2, pose_distribution="all")
    template_poses[:, :3, 3] *= 0.4
    poses = torch.tensor(template_poses).to(torch.float32).to(device)
    model.ref_data["poses"] =  poses[load_index_level_in_level2(0, "all"), :, :]

    mesh = trimesh.load_mesh(cad_path)
    model_points = mesh.sample(2048).astype(np.float32) / 1000.0
    model.ref_data["pointcloud"] = torch.tensor(model_points).unsqueeze(0).data.to(device)
    
    image_uv = model.project_template_to_image(best_template, pred_idx_objects, batch, detections.masks)

    geometric_score, visible_ratio = model.compute_geometric_score(
        image_uv, detections, query_appe_descriptors, ref_aux_descriptor, visible_thred=model.visible_thred
        )
    
    geo_end = time.time()
    logging.info(f"Geometric scoring time: {geo_end - geo_start:.4f} seconds")

    # final score
    finalize_start = time.time()
    final_score = (semantic_score + appe_scores + geometric_score*visible_ratio) / (1 + 1 + visible_ratio)

    detections.add_attribute("scores", final_score)
    detections.add_attribute("object_ids", torch.zeros_like(final_score))   
         
    detections.to_numpy()
    save_path = f"{output_dir}/detection_ism"
    detections.save_to_file(0, 0, 0, save_path, "Custom", return_results=False)
    detections = convert_npz_to_json(idx=0, list_npz_paths=[save_path+".npz"])
    save_json_bop23(save_path+".json", detections)
    vis_img = visualize(rgb, detections, f"{output_dir}/vis_ism.png")
    vis_img.save(f"{output_dir}/vis_ism.png")

    # 2) If you want *all* detections in one image:
    vis_all_img = visualize_all_detections_with_scores(rgb, detections, 
        f"{output_dir}/vis_all_detections.png")
    vis_all_img.save(f"{output_dir}/vis_all_detections.png")

    # Add these print statements before accessing best_template[i]
    print("Best template type:", type(best_template))
    print("Best template content:", best_template)

    if isinstance(best_template, torch.Tensor):
        print("Best template shape:", best_template.shape)

    for i, det in enumerate(detections):
        # Fetch the corresponding template using the index from best_template
        template_index = best_template[i].item()  # Convert index to Python int
        actual_template = templates[template_index]  # Get the template tensor [C, H, W]

        # Permute dimensions for visualization (convert [C, H, W] to [H, W, C])
        template_img = actual_template.permute(1, 2, 0).cpu().numpy()

        # Scale and convert to uint8 for saving
        template_img = np.clip(template_img * 255, 0, 255).astype(np.uint8)

    # Save the image
    Image.fromarray(template_img).save(f"{output_dir}/best_template_det_{i}.png")

    
    finalize_end = time.time()
    logging.info(f"Finalization time: {finalize_end - finalize_start:.4f} seconds")

    # Total runtime
    total_end = time.time()
    logging.info(f"Total inference runtime: {total_end - total_start:.4f} seconds")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--segmentor_model", default='sam', help="The segmentor model in ISM")
    parser.add_argument("--output_dir", nargs="?", help="Path to root directory of the output")
    parser.add_argument("--cad_path", nargs="?", help="Path to CAD(mm)")
    parser.add_argument("--rgb_path", nargs="?", help="Path to RGB image")
    parser.add_argument("--depth_path", nargs="?", help="Path to Depth image(mm)")
    parser.add_argument("--cam_path", nargs="?", help="Path to camera information")
    parser.add_argument("--stability_score_thresh", default=0.97, type=float, help="stability_score_thresh of SAM")
    args = parser.parse_args()
    run_inference(
        args.segmentor_model, args.output_dir, args.cad_path, args.rgb_path, args.depth_path, args.cam_path, 
        stability_score_thresh=args.stability_score_thresh,
    )