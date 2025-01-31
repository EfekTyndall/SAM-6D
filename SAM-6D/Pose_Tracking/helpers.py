import os
import numpy as np
import open3d as o3d
from glob import glob
from PIL import Image

def load_camera_intrinsics(file_path):
    """
    Load the camera intrinsic matrix from a .txt file.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
        intrinsic_matrix = np.array([list(map(float, line.split())) for line in lines])

    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]

    return {"fx": fx, "fy": fy, "cx": cx, "cy": cy}

def load_frames_from_folder(folder_path, extension="png"):
    """
    Load image frames from a folder.
    """
    file_paths = sorted(glob(os.path.join(folder_path, f"*.{extension}")))
    return [np.array(Image.open(fp)) for fp in file_paths]

def crop_depth_with_mask(depth_frame, mask):
    """Crop depth frame using the segmentation mask."""
    return np.where(mask, depth_frame, 0)

def depth_to_point_cloud_optimized(depth_frame, intrinsics, scale=0.001, downsample_factor=2):
    """
    Convert a depth frame to a point cloud using vectorized operations.
    Args:
        depth_frame: 2D array of depth values.
        intrinsics: Dictionary containing camera intrinsics.
        scale: Scale factor for depth values.
    Returns:
        Open3D point cloud.
    """
    height, width = depth_frame.shape
    fx, fy, cx, cy = intrinsics['fx'], intrinsics['fy'], intrinsics['cx'], intrinsics['cy']

    # Generate a grid of pixel coordinates
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    #z = depth_frame * scale

    # Downsample the grid and depth frame
    u = u[::downsample_factor, ::downsample_factor]
    v = v[::downsample_factor, ::downsample_factor]
    z = depth_frame[::downsample_factor, ::downsample_factor] * scale

    # Compute 3D points
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    # Filter out invalid points
    valid = z > 0
    points = np.stack((x[valid], y[valid], z[valid]), axis=-1)

    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))

def track_pose(reference_cloud, target_cloud, init_transform):
    """Perform pose tracking using ICP."""
    icp = o3d.pipelines.registration.registration_icp(
        reference_cloud, target_cloud,
        max_correspondence_distance=0.05,
        init=init_transform,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    return icp.transformation

def load_cad_model(cad_model_path):
    """
    Load the CAD model point cloud.
    """
    return o3d.io.read_point_cloud(cad_model_path)

def compute_relative_transform(T_prev, T_curr):
    """
    Compute the relative transformation from T_prev to T_curr.
    Args:
        T_prev: 4x4 numpy array, transformation matrix at previous frame.
        T_curr: 4x4 numpy array, transformation matrix at current frame.
    Returns:
        4x4 numpy array representing the relative transformation from T_prev to T_curr.
    """
    return np.linalg.inv(T_prev) @ T_curr

def project_point_cloud_to_2d(points, intrinsics):
    """
    Project a 3D point cloud into a 2D image plane using camera intrinsics.
    Args:
        points: Nx3 numpy array of 3D points (in camera coordinates).
        intrinsics: Dictionary containing camera intrinsics (fx, fy, cx, cy).
    Returns:
        Nx2 numpy array of 2D pixel coordinates.
    """
    fx, fy, cx, cy = intrinsics["fx"], intrinsics["fy"], intrinsics["cx"], intrinsics["cy"]
    projected = []
    for point in points:
        x, y, z = point
        if z > 0:  # Only project points in front of the camera
            u = int((x * fx / z) + cx)
            v = int((y * fy / z) + cy)
            projected.append((u, v))
    return np.vstack(projected)


def overlay_points_on_frame(frame, points, color=(0, 0, 255)):
    """
    Overlay projected points on an image frame using NumPy.
    Args:
        frame: RGB frame (HxWx3).
        points: Nx2 array of (u, v) pixel coordinates.
        color: Tuple for point color (default: red).
    """
    for u, v in points.astype(int):
        if 0 <= v < frame.shape[0] and 0 <= u < frame.shape[1]:
            frame[v, u] = color
    return frame
