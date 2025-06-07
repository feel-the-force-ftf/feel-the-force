import torch
import numpy as np
from scipy.interpolate import CubicSpline, interp1d
from scipy.spatial.transform import Rotation
from typing import List

camera2pixelkey = {
    "cam_1": "pixels1",
    "cam_2": "pixels2",
    "cam_3": "pixels3",
    "cam_4": "pixels4",
    "cam_51": "pixels51",
}
pixelkey2camera = {v: k for k, v in camera2pixelkey.items()}


import numpy as np
import torch

import numpy as np
import torch

def sliding_window_outlier_filter(points_list, window_size=10, std_threshold=1):
    """
    Frame-wise outlier detection using sliding window with mean/std.
    If a frame contains any zeros, it is skipped in calculations.
    If a frame is an outlier, the entire tensor is set to zero.
    """

    frame_norms = []
    valid_mask = np.ones(len(points_list), dtype=bool)  # True if frame is valid (no zeros)
    
    for i, tensor in enumerate(points_list):
        arr = tensor.cpu().numpy() if torch.is_tensor(tensor) else np.array(tensor)
        frame_norms.append(np.linalg.norm(arr))
        
        # Check if the frame contains any zeros
        if np.any(arr == 0):
            valid_mask[i] = False
    
    frame_norms = np.array(frame_norms)
    output_list = [tensor.clone() for tensor in points_list]
    
    for i in range(len(points_list)):
        if not valid_mask[i]:
            continue  # Skip if frame contains zeros
        
        # Get sliding window around current frame
        start = max(0, i - window_size // 2)
        end = min(len(points_list), i + window_size // 2 + 1)
        window_indices = np.arange(start, end)
        
        # Only consider valid frames in the window
        valid_window_indices = window_indices[valid_mask[window_indices]]
        if len(valid_window_indices) == 0:
            continue  # No valid frames to compare
        
        window_norms = frame_norms[valid_window_indices]
        mean_norm = np.mean(window_norms)
        std_norm = np.std(window_norms)
        
        # Detect if current frame is an outlier
        if std_norm > 1e-6:  
            z_score = abs((frame_norms[i] - mean_norm) / std_norm)
            if z_score > std_threshold:
                output_list[i] = torch.zeros_like(points_list[i])  
                valid_mask[i] = False  # Mark as invalid for future windows
    
    return output_list


def moving_average_filter(signal: np.ndarray, window_size: int, axis: int = 0):
    if window_size % 2 == 0:
        window_size += 1
    pad_width = window_size // 2
    padded_signal = np.pad(signal, ((pad_width, pad_width), (0, 0)), mode="reflect")
    kernel = np.ones(window_size) / window_size
    smoothed = np.apply_along_axis(
        lambda x: np.convolve(x, kernel, mode="valid"), axis=axis, arr=padded_signal
    )
    return smoothed


def interpolate_translations(indices: np.ndarray, t: np.ndarray):
    try:
        interp_func = interp1d(indices, t, axis=0, kind="linear", fill_value="extrapolate")
        indices_new = np.arange(indices.min(), indices.max() + 1)
        t = interp_func(indices_new)
    except:
        breakpoint()
    return t

def interpolate_rotations(indices: np.ndarray, r: np.ndarray):
    interp_func = CubicSpline(indices, r, axis=0)
    indices = np.arange(indices.min(), indices.max() + 1)
    r = interp_func(indices)
    return r

def filter_and_interpolate_fingertips(fingertips):
    all_fingertips, indices = [], []
    for i, fingertip in enumerate(fingertips):
        if not torch.any(fingertip == 0):
            indices.append(i)
            all_fingertips.append(fingertip.cpu().numpy())
        else:
            print(f"Frame {i} has zero fingertips")
    
    if indices[-1] != len(fingertips)-1: 
        all_fingertips.append(all_fingertips[-1])
        indices.append(len(fingertips)-1)
    if indices[0] != 0: 
        all_fingertips.insert(0, all_fingertips[0])
        indices.insert(0,0)


    indices = np.array(indices)
    all_fingertips = np.array(all_fingertips)
    all_fingertips = interpolate_translations(indices, all_fingertips)
    indices = np.arange(indices.min(), indices.max() + 1)
    all_fingertips = [torch.from_numpy(arr) for arr in all_fingertips]
    return indices, all_fingertips

def pixel2d_to_3d_torch(points2d, depths, intrinsic_matrix, extrinsic_matrix):
    intrinsic_matrix = torch.tensor(intrinsic_matrix).float().to(depths.device)
    extrinsic_matrix = torch.tensor(extrinsic_matrix).float().to(depths.device)
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]
    x = (points2d[:, 0] - cx) / fx
    y = (points2d[:, 1] - cy) / fy
    points3d = torch.stack((x * depths, y * depths, depths), dim=1)  # in camera frame
    points3d = torch.cat(
        (points3d, torch.ones((len(points2d), 1)).to(depths.device)), dim=1
    )
    points3d = (torch.linalg.inv(extrinsic_matrix) @ points3d.T).T  # world frame
    return points3d[..., :3]


def pixel2d_to_3d(points2d, depths, intrinsic_matrix, extrinsic_matrix):
    points2d = np.array(points2d)
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]
    x = (points2d[:, 0] - cx) / fx
    y = (points2d[:, 1] - cy) / fy
    points_3d = np.column_stack((x * depths, y * depths, depths))  # in camera frame
    points_3d = np.concatenate([points_3d, np.ones((len(points2d), 1))], axis=1)
    points_3d = (np.linalg.inv(extrinsic_matrix) @ points_3d.T).T  # world frame
    return points_3d[..., :3]


def pixel3d_to_2d(points3d, intrinsic_matrix, camera_projection_matrix):
    points3d = np.array(points3d)
    points3d = np.concatenate([points3d, np.ones((len(points3d), 1))], axis=1)
    points3d = (camera_projection_matrix @ points3d.T).T  # camera frame
    depth = points3d[:, 2]
    points2d = (intrinsic_matrix @ points3d.T).T
    points2d = points2d / points2d[:, 2][:, None]
    return points2d[..., :2], depth


def triangulate_points(P, points):
    """
    Triangulate a batch of points from a variable number of camera views.

    Parameters:
    P: list of 3x4 projection matrices for each camera (currently world2camera transform)
    points: list of Nx2 arrays of normalized image coordinates for each camera

    Returns:
    Nx4 array of homogeneous 3D points
    """
    num_views = len(P)
    assert num_views > 1, "At least 2 cameras are required for triangulation"

    num_points = points[0].shape[0]
    A = np.zeros((num_points, num_views * 2, 4))

    for idx in range(num_views):
        # Set up the linear system for each point
        A[:, idx * 2] = points[idx][:, 0, np.newaxis] * P[idx][2] - P[idx][0]
        A[:, idx * 2 + 1] = points[idx][:, 1, np.newaxis] * P[idx][2] - P[idx][1]

    # Solve the system using SVD
    _, _, Vt = np.linalg.svd(A)
    X = Vt[:, -1, :]

    # Normalize the homogeneous coordinates
    X = X / X[:, 3:]

    return X


def rigid_transform_3D(A, B):
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_cols != 3:
        raise Exception(f"matrix A is not Nx3, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_cols != 3:
        raise Exception(f"matrix B is not Nx3, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am.T @ Bm

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A.T + centroid_B.T

    return R, t


def rotation_6d_to_matrix(d6: np.ndarray) -> np.ndarray:
    """
    Converts 6D rotation representation to rotation matrix
    using Gram-Schmidt orthogonalization.

    Args:
        d6: 6D rotation representation, of shape (..., 6)

    Returns:
        Batch of rotation matrices of shape (..., 3, 3)
    """
    a1, a2 = d6[..., :3], d6[..., 3:]

    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2, axis=-1)

    return np.stack((b1, b2, b3), axis=-2)


def matrix_to_rotation_6d(matrix: np.ndarray) -> np.ndarray:
    """
    Converts rotation matrices to 6D rotation representation
    by dropping the last row.

    Args:
        matrix: Batch of rotation matrices of shape (..., 3, 3)

    Returns:
        6D rotation representation, of shape (..., 6)
    """
    batch_dim = matrix.shape[:-2]
    return matrix[..., :2, :].reshape(batch_dim + (6,))
