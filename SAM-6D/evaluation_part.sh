#!/usr/bin/env bash

# Common paths that are the same for all scenes/runs
export CAD_PATH="/home/martyn/Thesis/pose-estimation/data/cad-models/part.ply"
export CAMERA_PATH="/home/martyn/Thesis/pose-estimation/data/scenes/cam_K.json"

# Path to SAM-6D directory
SAM6D_DIR="/home/martyn/Thesis/pose-estimation/methods/SAM-6D/SAM-6D/"

# The base output directory for results
BASE_OUTPUT_DIR="/home/martyn/Thesis/pose-estimation/results/part/methods/sam6d"

# The root directory containing the 10 scenes
SCENES_ROOT="/home/martyn/Thesis/pose-estimation/data/scenes/scenes_part/"

# Number of scenes and runs per scene
NUM_SCENES=10
NUM_RUNS=5

# Loop over each scene
for scene_num in $(seq 1 $NUM_SCENES); do

  # Construct the scene folder name: scene_01, scene_02, etc.
  SCENE_NAME=$(printf "scene_%02d" ${scene_num})
  SCENE_DIR="${SCENES_ROOT}/${SCENE_NAME}"

  # The path to that scene's ground truth .txt
  GT_PATH="${SCENE_DIR}/tf_ground_truth.txt"

  # The RGB and Depth images for this scene
  SCENE_RGB_PATH="${SCENE_DIR}/rgb.png"
  SCENE_DEPTH_PATH="${SCENE_DIR}/depth.png"

  # Safety checks: skip if images are missing
  if [[ ! -f "${SCENE_RGB_PATH}" ]]; then
    echo "Warning: RGB file not found at ${SCENE_RGB_PATH}. Skipping."
    continue
  fi
  if [[ ! -f "${SCENE_DEPTH_PATH}" ]]; then
    echo "Warning: Depth file not found at ${SCENE_DEPTH_PATH}. Skipping."
    continue
  fi

  # For each scene, run 5 times
  for run_idx in $(seq 1 $NUM_RUNS); do
    echo "======================================="
    echo "Processing ${SCENE_NAME} - Run ${run_idx}"
    echo "======================================="

    # Create a unique output subfolder for each run
    RUN_NAME=$(printf "run_%02d" ${run_idx})  # Format with a leading zero
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/${SCENE_NAME}/${RUN_NAME}"
    mkdir -p "${OUTPUT_DIR}"

    # Export environment variables so demo.sh can read them
    export RGB_PATH="${SCENE_RGB_PATH}"
    export DEPTH_PATH="${SCENE_DEPTH_PATH}"
    export OUTPUT_DIR

    # Go to the SAM-6D directory and run the pipeline
    cd "${SAM6D_DIR}"

    # Run instance segmentation model
    export SEGMENTOR_MODEL=fastsam

    cd Instance_Segmentation_Model
    python run_ism_eval.py --segmentor_model $SEGMENTOR_MODEL --output_dir $OUTPUT_DIR --cad_path $CAD_PATH --rgb_path $RGB_PATH --depth_path $DEPTH_PATH --cam_path $CAMERA_PATH

    # Run pose estimation model
    export SEG_PATH=$OUTPUT_DIR/detection_ism.json

    cd ../Pose_Estimation_Model
    python run_pem_eval_part.py --output_dir $OUTPUT_DIR --cad_path $CAD_PATH --rgb_path $RGB_PATH --depth_path $DEPTH_PATH --cam_path $CAMERA_PATH --seg_path $SEG_PATH --gt_path $GT_PATH

    cd ..
    python aggregate_scene_metrics_part.py
  done
done

echo "Finished processing all scenes and runs!"

