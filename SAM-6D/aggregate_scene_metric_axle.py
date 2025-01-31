import os
import pandas as pd

# Define the base directory for the scenes
base_dir = "/home/martyn/Thesis/pose-estimation/results/axle/methods/sam6d/"

# Step 1: Calculate scene_metrics.csv for each scene
for scene_folder in sorted(os.listdir(base_dir)):
    scene_path = os.path.join(base_dir, scene_folder)

    # Check if it's a directory and follows the scene_* naming pattern
    if os.path.isdir(scene_path) and scene_folder.startswith("scene_"):
        all_runtime_values = []  # List to store runtime values from all runs

        # Iterate over each run folder in the scene
        for run_folder in sorted(os.listdir(scene_path)):
            run_path = os.path.join(scene_path, run_folder)

            # Check if it's a directory and contains evaluation_metrics.csv
            if os.path.isdir(run_path) and os.path.exists(os.path.join(run_path, "evaluation_metrics.csv")):
                # Read the evaluation_metrics.csv file
                metrics = pd.read_csv(
                    os.path.join(run_path, "evaluation_metrics.csv"),
                    header=None,  # No header in your CSV
                    names=["Metric", "Value"]  # Assign column names
                )
                
                # Extract the runtime value
                try:
                    runtime_value = float(metrics.loc[metrics["Metric"] == "Inference Runtime (seconds)", "Value"].values[0])
                    all_runtime_values.append(runtime_value)
                except (ValueError, IndexError) as e:
                    print(f"Error processing runtime value in {run_path}: {e}")

        # Calculate and save scene-level metrics if there are runtime values
        if all_runtime_values:
            mean_runtime = sum(all_runtime_values) / len(all_runtime_values)
            std_runtime = pd.Series(all_runtime_values).std()

            # Save as scene_metrics.csv
            scene_metrics = pd.DataFrame([{
                "Scene": scene_folder,
                "Inference Runtime (s) Mean": mean_runtime,
                "Inference Runtime (s) SD": std_runtime
            }])
            scene_metrics.to_csv(os.path.join(scene_path, "scene_metrics.csv"), index=False)

# Step 2: Combine all scene_metrics.csv into all_scenes_average_metrics.csv
all_scenes_metrics = []
for scene_folder in sorted(os.listdir(base_dir)):
    scene_path = os.path.join(base_dir, scene_folder)

    # Check if scene_metrics.csv exists
    if os.path.isdir(scene_path) and os.path.exists(os.path.join(scene_path, "scene_metrics.csv")):
        # Read the scene_metrics.csv
        scene_metrics = pd.read_csv(os.path.join(scene_path, "scene_metrics.csv"))
        all_scenes_metrics.append(scene_metrics)

# Combine and save
if all_scenes_metrics:
    combined_metrics_df = pd.concat(all_scenes_metrics, ignore_index=True)
    output_combined_path = os.path.join(base_dir, "all_scenes_average_metrics.csv")
    combined_metrics_df.to_csv(output_combined_path, index=False)
    print(f"Combined metrics saved to: {output_combined_path}")