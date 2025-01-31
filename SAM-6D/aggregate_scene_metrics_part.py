#!/usr/bin/env python

import os
import pandas as pd
import numpy as np

def aggregate_scene_metrics(scenes_root, num_scenes=10, num_runs=5):
    """
    Aggregates run-level evaluation_metrics.csv data into scene-level means and standard deviations.

    :param scenes_root: Path to directory containing scene_01, scene_02, ..., scene_10
    :param num_scenes:  How many scenes total (default 10)
    :param num_runs:    How many runs per scene (default 5)
    """
    # We’ll store all scene-level metrics in a list of dicts to create a final DataFrame
    all_scenes_data = []

    for scene_idx in range(1, num_scenes+1):
        scene_name = f"scene_{scene_idx:02d}"
        scene_dir = os.path.join(scenes_root, scene_name)

        if not os.path.isdir(scene_dir):
            print(f"[WARN] {scene_dir} does not exist. Skipping.")
            continue

        # Prepare lists to accumulate per-run values
        rotation_errors = []
        translation_errors = []
        add_metrics = []
        runtimes = []

        # Try reading each run’s CSV
        for run_idx in range(1, num_runs+1):
            run_name = f"run_{run_idx:02d}"
            run_dir = os.path.join(scene_dir, run_name)
            metrics_csv_path = os.path.join(run_dir, "evaluation_metrics.csv")

            if not os.path.isfile(metrics_csv_path):
                print(f"[WARN] Missing {metrics_csv_path}, skipping run {run_name} for {scene_name}")
                continue

            # Load the CSV
            df = pd.read_csv(metrics_csv_path)
            # We assume the CSV has "Metric" and "Value" columns
            # Convert them to a dictionary
            metrics_dict = dict(zip(df["Metric"], df["Value"]))

            # Extract relevant metrics by key
            if "Rotation Error (deg)" in metrics_dict:
                rotation_errors.append(metrics_dict["Rotation Error (deg)"])
            if "Translation Error (mm)" in metrics_dict:
                translation_errors.append(metrics_dict["Translation Error (mm)"])
            if "ADD Metric (mm)" in metrics_dict:
                add_metrics.append(metrics_dict["ADD Metric (mm)"])
            if "Inference Runtime (seconds)" in metrics_dict:
                runtimes.append(metrics_dict["Inference Runtime (seconds)"])

        # If no runs or missing data, skip
        if len(rotation_errors) == 0:
            print(f"[WARN] No valid runs found for {scene_name}. Skipping scene.")
            continue

        # Compute mean & std for each metric
        rot_mean = np.mean(rotation_errors)
        rot_std  = np.std(rotation_errors, ddof=1)  # sample std

        trans_mean = np.mean(translation_errors) if len(translation_errors) else np.nan
        trans_std  = np.std(translation_errors, ddof=1) if len(translation_errors) else np.nan

        add_mean = np.mean(add_metrics) if len(add_metrics) else np.nan
        add_std  = np.std(add_metrics, ddof=1) if len(add_metrics) else np.nan

        runtime_mean = np.mean(runtimes) if len(runtimes) else np.nan
        runtime_std  = np.std(runtimes, ddof=1) if len(runtimes) else np.nan

        # Print or store
        print(f"\nScene: {scene_name}")
        print(f"  Rotation Error (deg): mean={rot_mean:.4f}, std={rot_std:.4f}")
        print(f"  Translation Error (mm): mean={trans_mean:.4f}, std={trans_std:.4f}")
        print(f"  ADD Metric (mm): mean={add_mean:.4f}, std={add_std:.4f}")
        print(f"  Inference Runtime (seconds): mean={runtime_mean:.4f}, std={runtime_std:.4f}")

        # Build a scene-level dict
        scene_row = {
            "Scene": scene_name,
            "Rotation Error (deg) Mean": rot_mean,
            "Rotation Error (deg) SD": rot_std,
            "Translation Error (mm) Mean": trans_mean,
            "Translation Error (mm) SD": trans_std,
            "ADD Metric (mm) Mean": add_mean,
            "ADD Metric (mm) SD": add_std,
            "Inference Runtime (s) Mean": runtime_mean,
            "Inference Runtime (s) SD": runtime_std,
        }
        all_scenes_data.append(scene_row)

        # You can also write a scene-level CSV if you want, e.g.:
        scene_csv_path = os.path.join(scene_dir, "scene_metrics.csv")
        pd.DataFrame([scene_row]).to_csv(scene_csv_path, index=False)
        print(f"Saved scene-level metrics to {scene_csv_path}")

    # If you want an overall CSV for all scenes:
    if all_scenes_data:
        summary_csv_path = os.path.join(scenes_root, "all_scenes_average_metrics.csv")
        df_all = pd.DataFrame(all_scenes_data)
        df_all.to_csv(summary_csv_path, index=False)
        print(f"\nSaved all scenes metrics to {summary_csv_path}")


if __name__ == "__main__":
    # Example usage:
    # python aggregate_scene_metrics.py --root /path/to/scenes_results
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/home/martyn/Thesis/pose-estimation/results/part/methods/sam6d/", help="Root folder containing scene_01, scene_02, ...")
    parser.add_argument("--num_scenes", type=int, default=10, help="Number of scenes")
    parser.add_argument("--num_runs", type=int, default=5, help="Number of runs per scene")
    args = parser.parse_args()

    aggregate_scene_metrics(args.root, num_scenes=args.num_scenes, num_runs=args.num_runs)
