import sys

sys.path.append("../../")

import re
import yaml
import argparse
import pickle as pkl
from pathlib import Path
import cv2
import numpy as np
from pandas import read_csv
from scipy.spatial.transform import Rotation as R

from gripper_points import extrapoints, Tshift

import torch
from point_utils.points_class import PointsClass
from utils import (
    camera2pixelkey,
    pixel2d_to_3d_torch,
    triangulate_points,
)

# Create the parser
parser = argparse.ArgumentParser(
    description="Convert processed robot data into a pkl file"
)

# Add the arguments
parser.add_argument("--data_dir", type=str, help="Path to the data directory")
parser.add_argument("--calib_path", type=str, help="Path to the calibration file")
parser.add_argument("--task_names", nargs="+", type=str, help="List of task names")
parser.add_argument(
    "--num_demos", type=int, default=None, help="Number of demonstrations to process"
)
parser.add_argument(
    "--process_points", action="store_true", help="Process human hand points"
)
parser.add_argument(
    "--use_gt_depth", action="store_true", help="Use ground truth depth"
)

args = parser.parse_args()
DATA_DIR = Path(args.data_dir)
CALIB_PATH = Path(args.calib_path)
task_names = args.task_names
NUM_DEMOS = args.num_demos
process_points = args.process_points
use_gt_depth = args.use_gt_depth

camera_indices = [1, 2]
original_img_size = (640, 480)
crop_h, crop_w = (0.0, 1.0), (0.0, 1.0)
save_img_size = None
object_labels = [
    "objects",
]

PROCESSED_DATA_PATH = Path(DATA_DIR) / "processed_data"
SAVE_DATA_PATH = Path(DATA_DIR) / "processed_data_pkl" 

if save_img_size is None:
    save_img_size = (
        int(original_img_size[0] * (crop_w[1] - crop_w[0])),
        int(original_img_size[1] * (crop_h[1] - crop_h[0])),
    )


if task_names is None:
    task_names = [x.name for x in PROCESSED_DATA_PATH.iterdir() if x.is_dir()]

SAVE_DATA_PATH.mkdir(parents=True, exist_ok=True)

# Calibration data
calibration_data = np.load(CALIB_PATH, allow_pickle=True).item()

# Load p3po config for tracking
if process_points:
    with open("../../cfgs/suite/points_cfg.yaml") as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
        root_dir, dift_path, cotracker_checkpoint = (
            cfg["root_dir"],
            cfg["dift_path"],
            cfg["cotracker_checkpoint"],
        )
        cfg["dift_path"] = f"{root_dir}/{dift_path}"
        cfg["cotracker_checkpoint"] = f"{root_dir}/{cotracker_checkpoint}"
        cfg["task_name"] = task_names[0]
        cfg["pixel_keys"] = [
            camera2pixelkey[f"cam_{cam_idx}"] for cam_idx in camera_indices
        ]
        cfg["object_labels"] = object_labels
        cfg["use_gt_depth"] = use_gt_depth
    points_class = PointsClass(**cfg)


def extract_number(s):
    s = s.strip()
    # Remove any leading/trailing brackets or whitespace
    s = s.strip("[]")
    # Match 'np.float32(number)' or 'np.float64(number)'
    match = re.match(r"np\.float(?:32|64)\((-?\d+\.?\d*(?:[eE][-+]?\d+)?)\)", s)
    if match:
        return float(match.group(1))
    else:
        # Match plain numbers, including negatives and decimals
        match = re.match(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", s)
        if match:
            return float(match.group(0))
        else:
            raise ValueError(f"Cannot extract number from '{s}'")


for TASK_NAME in task_names:
    DATASET_PATH = Path(f"{PROCESSED_DATA_PATH}/{TASK_NAME}")

    if (SAVE_DATA_PATH / "expert_demos" / "franka_env" / f"{TASK_NAME}.pkl").exists():
        print(f"Data for {TASK_NAME} already exists. Appending to it...")
        input("Press Enter to continue...")
        data = pkl.load(open(SAVE_DATA_PATH / "expert_demos" / "franka_env" / f"{TASK_NAME}.pkl", "rb"))
        observations = data["observations"]
        max_cartesian = data["max_cartesian"]
        min_cartesian = data["min_cartesian"]
        max_gripper = data["max_gripper"]
        min_gripper = data["min_gripper"]
        max_sensor = data["max_sensor"]
        min_sensor = data["min_sensor"]
    else:
        observations = []
        max_cartesian, min_cartesian = None, None
        max_gripper, min_gripper = None, None
        max_sensor, min_sensor = None, None

    dirs = [x for x in DATASET_PATH.iterdir() if x.is_dir()]
    for i, data_point in enumerate(sorted(dirs)):
        print(f"Processing data point {i+1}/{len(dirs)}")

        if NUM_DEMOS is not None and int(str(data_point).split("_")[-1]) >= NUM_DEMOS:
            print(f"Skipping data point {data_point}")
            continue

        observation = {}

        # Process images
        image_dir = data_point / "videos"
        if not image_dir.exists():
            print(f"Data point {data_point} is incomplete")
            continue

        for save_idx, idx in enumerate(camera_indices):
            video_path = image_dir / f"camera{idx}.mp4"
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                print(f"Video {video_path} could not be opened")
                continue

            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # crop the image
                h, w, _ = frame.shape
                frame = frame[
                    int(h * crop_h[0]) : int(h * crop_h[1]),
                    int(w * crop_w[0]) : int(w * crop_w[1]),
                ]

                frame = cv2.resize(frame, save_img_size)
                frames.append(frame)

            observation[f"pixels{idx}"] = np.array(frames)

        # Process depth
        if use_gt_depth:
            depth_dir = data_point / "depth"
            if not depth_dir.exists():
                print(f"Data point {data_point} is incomplete (no depth)")
                continue

            depth_frames = {}
            for idx in camera_indices:
                depth_file = depth_dir / f"depth{idx}.pkl"
                with open(depth_file, "rb") as f:
                    depth = pkl.load(f)  # depth in meters
                depth_frames[idx] = depth

        state_csv_path = data_point / "states.csv"
        sensor_csv_path = data_point / "sensor.csv"
        state = read_csv(state_csv_path)

        try:
            # Parsing sensor data
            sensor_data = read_csv(sensor_csv_path)
            sensor_states = sensor_data["sensor_values"].values

            sensor_states = np.array(
                [
                    np.array([extract_number(x) for x in sensor.strip("[]").split(",")])
                    for sensor in sensor_states
                ],
                dtype=np.float32,
            )

            sensor_history = None
            if "sensor_history" in sensor_data.columns:
                sensor_history = sensor_data["sensor_history"].values
                sensor_history = np.array(
                    [
                        np.array(
                            [extract_number(x) for x in sensor.strip("[]").split(",")]
                        ).reshape((-1, 30))
                        for sensor in sensor_history
                    ],
                    dtype=np.float32,
                )

            baseline_sensor_state = np.mean(sensor_states[:5], axis=0)
            sensor_states -= baseline_sensor_state
            sensor_states = np.linalg.norm(sensor_states[:, :3], axis=1)
            # Update max and min sensor values for normalization
            if max_sensor is None:
                max_sensor = np.max(sensor_states, axis=0)
                min_sensor = np.min(sensor_states, axis=0)
            else:
                max_sensor = np.maximum(max_sensor, np.max(sensor_states, axis=0))
                min_sensor = np.minimum(min_sensor, np.min(sensor_states, axis=0))

            observation["sensor_states"] = sensor_states.astype(np.float32)
            use_sensor = True

        except FileNotFoundError:
            use_sensor = False
            print(f"Sensor data not found for {data_point}")
            sensor_states, sensor_history = None, None

        # Parsing cartesian pose data
        cartesian_states = state["pose_aa"].values
        cartesian_states = np.array(
            [
                np.array([extract_number(x) for x in pose.strip("[]").split(",")])
                for pose in cartesian_states
            ],
            dtype=np.float32,
        )

        gripper_states = state["gripper_state"].values.astype(np.float32)
        observation["cartesian_states"] = cartesian_states.astype(np.float32)
        observation["gripper_states"] = gripper_states.astype(np.float32)

        if process_points:
            # Robot Tracks
            for cam_idx in camera_indices:
                if cam_idx == 51:
                    continue
                camera_name = f"cam_{cam_idx}"
                pixel_key = camera2pixelkey[camera_name]
                observation[f"robot_tracks_{pixel_key}"] = []
                # if save_tracks_3d:
                observation[f"gt_robot_tracks_3d_{pixel_key}"] = []

            for timestep, state in enumerate(cartesian_states):
                pos = state[:3]
                ori = state[3:]
                T_g_b = np.eye(4)
                T_g_b[:3, :3] = R.from_rotvec(ori).as_matrix()
                T_g_b[:3, 3] = pos

                # shift the point
                T_g_b = T_g_b @ Tshift

                # add extra points
                points3d = [T_g_b[:3, 3]]
                for idx, Tp in enumerate(extrapoints):
                    if gripper_states[timestep] == 1 and idx in [0, 1]:
                        Tp = Tp.copy()
                        Tp[1, 3] = 0.015 if idx == 0 else -0.015
                    pt = T_g_b @ Tp
                    pt = pt[:3, 3]
                    points3d.append(pt)
                points3d = np.array(points3d)

                # for pixel_key, camera_name in pixelkey2camera.items():
                for cam_idx in camera_indices:
                    if cam_idx == 51:
                        continue
                    camera_name = f"cam_{cam_idx}"
                    pixel_key = camera2pixelkey[camera_name]

                    P = calibration_data[camera_name]["ext"]  # 4x4
                    K = calibration_data[camera_name]["int"]  # 3x3
                    D = calibration_data[camera_name]["dist_coeff"]  # 5

                    r, t = P[:3, :3], P[:3, 3]
                    r, _ = cv2.Rodrigues(r)
                    points2d, _ = cv2.projectPoints(points3d, r, t, K, D)
                    points2d = points2d[:, 0]

                    # consider cropping params
                    w, h = original_img_size
                    points2d[:, 0] = points2d[:, 0] - int(w * crop_w[0])
                    points2d[:, 1] = points2d[:, 1] - int(h * crop_h[0])
                    img_w, img_h = int(w * (crop_w[1] - crop_w[0])), int(
                        h * (crop_h[1] - crop_h[0])
                    )
                    points2d[:, 0] = points2d[:, 0] * save_img_size[0] / img_w
                    points2d[:, 1] = points2d[:, 1] * save_img_size[1] / img_h

                    observation[f"robot_tracks_{pixel_key}"].append(points2d)

                    # store 3D points in robot base frame
                    observation[f"gt_robot_tracks_3d_{pixel_key}"].append(points3d)

            # Human hand tracks
            mark_every = 8
            save = True
            for cam_idx in camera_indices:
                if save == False:
                    break
                camera_name = f"cam_{cam_idx}"
                pixel_key = camera2pixelkey[camera_name]

                frames = observation[pixel_key]
                # CV2 reads in BGR format, so we need to convert to RGB
                frames = [frame[..., ::-1] for frame in frames]
                points_class.add_to_image_list(frames[0], pixel_key)
                for object_label in object_labels:
                    points_class.find_semantic_similar_points(pixel_key, object_label)
                try:
                    points_class.track_points(
                        pixel_key, last_n_frames=mark_every, is_first_step=True
                    )
                except:
                    print(f"Error in tracking hand points for {pixel_key}")
                    save = False
                    continue

                points_class.track_points(
                    pixel_key, last_n_frames=mark_every, one_frame=(mark_every == 1)
                )

                points_list = []
                points = points_class.get_points_on_image(pixel_key)
                points_list.append(points[0])

                if use_gt_depth:
                    points_3d_list = []
                    if use_gt_depth:
                        depth = depth_frames[cam_idx][0]
                        points_class.set_depth(
                            depth,
                            pixel_key,
                            original_img_size,
                            save_img_size,
                            (crop_h, crop_w),
                        )
                    else:
                        points_class.get_depth()
                    points_with_depth = points_class.get_points(pixel_key)
                    depths = points_with_depth[:, :, -1]

                    P = calibration_data[camera_name]["ext"]  # 4x4
                    K = calibration_data[camera_name]["int"]  # 3x3
                    points3d = pixel2d_to_3d_torch(points[0], depths[0], K, P)
                    points_3d_list.append(points3d)

                for idx, image in enumerate(frames[1:]):
                    print(f"Traj: {i}, Frame: {idx}, Image: {pixel_key}")
                    points_class.add_to_image_list(image, pixel_key)

                    if use_gt_depth:
                        depth = depth_frames[cam_idx][idx]
                        points_class.set_depth(
                            depth,
                            pixel_key,
                            original_img_size,
                            save_img_size,
                            (crop_h, crop_w),
                        )

                    if (idx + 1) % mark_every == 0 or idx == (len(frames) - 2):
                        to_add = mark_every - (idx + 1) % mark_every
                        if to_add < mark_every:
                            for j in range(to_add):
                                points_class.add_to_image_list(image, pixel_key)
                        else:
                            to_add = 0

                        points_class.track_points(
                            pixel_key,
                            last_n_frames=mark_every,
                            one_frame=(mark_every == 1),
                        )

                        points = points_class.get_points_on_image(
                            pixel_key, last_n_frames=mark_every
                        )
                        for j in range(mark_every - to_add):
                            points_list.append(points[j])

                        if use_gt_depth:
                            if not use_gt_depth:
                                points_class.get_depth(last_n_frames=mark_every)

                            points_with_depth = points_class.get_points(
                                pixel_key, last_n_frames=mark_every
                            )
                            for j in range(mark_every - to_add):
                                depth = points_with_depth[j, :, -1]
                                # points3d = pixel2d_to_3d_camera_frame_torch(
                                #     points[j], depth, K
                                # )
                                points3d = pixel2d_to_3d_torch(points[j], depth, K, P)
                                points_3d_list.append(points3d)

                observation[f"object_tracks_{pixel_key}"] = torch.stack(
                    points_list
                ).numpy()
                if use_gt_depth:
                    observation[f"object_tracks_3d_{pixel_key}"] = torch.stack(
                        points_3d_list
                    ).numpy()
                points_class.reset_episode()

            if save == False:
                continue

            for cam_idx in camera_indices:
                if cam_idx == 51:
                    continue
                camera_name = f"cam_{cam_idx}"
                pixel_key = camera2pixelkey[camera_name]
                observation[f"robot_tracks_{pixel_key}"] = np.array(
                    observation[f"robot_tracks_{pixel_key}"]
                )
                # if save_tracks_3d:
                observation[f"gt_robot_tracks_3d_{pixel_key}"] = np.array(
                    observation[f"gt_robot_tracks_3d_{pixel_key}"]
                )

        # Update max and min cartesian values for normalization
        if max_cartesian is None:
            max_cartesian = np.max(cartesian_states, axis=0)
            min_cartesian = np.min(cartesian_states, axis=0)
        else:
            max_cartesian = np.maximum(max_cartesian, np.max(cartesian_states, axis=0))
            min_cartesian = np.minimum(min_cartesian, np.min(cartesian_states, axis=0))

        # Update max and min gripper values for normalization
        if max_gripper is None:
            max_gripper = np.max(gripper_states)
            min_gripper = np.min(gripper_states)
        else:
            max_gripper = np.maximum(max_gripper, np.max(gripper_states))
            min_gripper = np.minimum(min_gripper, np.min(gripper_states))

        if process_points and not use_gt_depth:
            """
            Triangulate 3D points from 2D points when gt_depth is not available
            """
            for cam_idx in camera_indices:
                camera_name = f"cam_{cam_idx}"
                pixel_key = camera2pixelkey[camera_name]
                observation[f"object_tracks_3d_{pixel_key}"] = []
                observation[f"robot_tracks_3d_{pixel_key}"] = []
            for t_idx in range(
                len(observation[f"object_tracks_{pixel_key}"])
            ):  # for each frame
                P, pts_obj, pts_robot = [], [], []
                for cam_idx in camera_indices:
                    camera_name = f"cam_{cam_idx}"
                    pixel_key = camera2pixelkey[camera_name]

                    extr = calibration_data[camera_name]["ext"]
                    intr = calibration_data[camera_name]["int"]
                    intr = np.concatenate([intr, np.zeros((3, 1))], axis=1)
                    P.append(intr @ extr)

                    # For object points
                    pt2d_obj = observation[f"object_tracks_{pixel_key}"][t_idx]
                    # compute point_h in original image
                    point_h = pt2d_obj[:, 1]
                    h_orig, h_curr = original_img_size[1], save_img_size[1]
                    h_orig_cropped = h_orig * (crop_h[1] - crop_h[0])
                    point_h_orig = (
                        point_h / h_curr
                    ) * h_orig_cropped + h_orig * crop_h[0]
                    point_h_orig = point_h_orig.astype(np.int32)
                    # compute point_w in original image
                    point_w = pt2d_obj[:, 0]
                    w_orig, w_curr = original_img_size[0], save_img_size[0]
                    w_orig_cropped = w_orig * (crop_w[1] - crop_w[0])
                    point_w_orig = (
                        point_w / w_curr
                    ) * w_orig_cropped + w_orig * crop_w[0]
                    point_w_orig = point_w_orig.astype(np.int32)
                    # Store points
                    pt2d_obj = np.column_stack((point_w_orig, point_h_orig))
                    pts_obj.append(pt2d_obj)

                    # For robot points
                    pt2d_robot = observation[f"robot_tracks_{pixel_key}"][t_idx]
                    # compute point_h in original image
                    point_h = pt2d_robot[:, 1]
                    h_orig, h_curr = original_img_size[1], save_img_size[1]
                    h_orig_cropped = h_orig * (crop_h[1] - crop_h[0])
                    point_h_orig = (
                        point_h / h_curr
                    ) * h_orig_cropped + h_orig * crop_h[0]
                    point_h_orig = point_h_orig.astype(np.int32)
                    # compute point_w in original image
                    point_w = pt2d_robot[:, 0]
                    w_orig, w_curr = original_img_size[0], save_img_size[0]
                    w_orig_cropped = w_orig * (crop_w[1] - crop_w[0])
                    point_w_orig = (
                        point_w / w_curr
                    ) * w_orig_cropped + w_orig * crop_w[0]
                    point_w_orig = point_w_orig.astype(np.int32)
                    # Store points
                    pt2d_robot = np.column_stack((point_w_orig, point_h_orig))
                    pts_robot.append(pt2d_robot)

                pts3d_obj = triangulate_points(P, pts_obj)
                pts3d_robot = triangulate_points(P, pts_robot)
                for cam_idx in camera_indices:
                    camera_name = f"cam_{cam_idx}"
                    pixel_key = camera2pixelkey[camera_name]
                    observation[f"object_tracks_3d_{pixel_key}"].append(
                        pts3d_obj[:, :3]
                    )
                    observation[f"robot_tracks_3d_{pixel_key}"].append(
                        pts3d_robot[:, :3]
                    )
            for cam_idx in camera_indices:
                camera_name = f"cam_{cam_idx}"
                pixel_key = camera2pixelkey[camera_name]
                observation[f"object_tracks_3d_{pixel_key}"] = np.array(
                    observation[f"object_tracks_3d_{pixel_key}"]
                )
                observation[f"robot_tracks_3d_{pixel_key}"] = np.array(
                    observation[f"robot_tracks_3d_{pixel_key}"]
                )

        observations.append(observation)

    if process_points:
        print("Creating visualization video...")
        # Create output directory for videos
        video_dir = SAVE_DATA_PATH / "point_overlay_videos"
        video_dir.mkdir(exist_ok=True)
        video_dir = video_dir / TASK_NAME
        video_dir.mkdir(exist_ok=True)
        for i, observation in enumerate(observations):
            for pixel_key in ['pixels1', 'pixels2', 'pixels3', 'pixels4']:
                if pixel_key not in observation:
                    continue
                frames = observation[pixel_key]
                points = observation[f"robot_tracks_{pixel_key}"]
          
                # Initialize video writer
                base_path = str(video_dir / f"{TASK_NAME}_demo_{i}_{pixel_key}")
                height, width = frames[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                
                def write_video(frames, points, output_path, include_sensor=False):
                    out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))
                    
                    for frame_idx, (frame, frame_points) in enumerate(zip(frames, points)):
                        frame = frame.copy()
                        
                        # Draw robot points in green
                        for point in frame_points:
                            x, y = point.astype(int)
                            cv2.circle(frame, (x, y), radius=3, color=(0, 255, 0), thickness=-1)
                        
                        # Draw object points in blue if they exist
                        if f"object_tracks_{pixel_key}" in observation:
                            object_points = observation[f"object_tracks_{pixel_key}"][frame_idx]
                            for point in object_points:
                                x, y = point.astype(int)
                                cv2.circle(frame, (x, y), radius=3, color=(255, 0, 0), thickness=-1)
                        
                        # Add sensor overlay if requested
                        if include_sensor:
                            sensor_norm = observation["sensor_states"][frame_idx]
                            gripper_value = observation["gripper_states"][frame_idx]
                            
                            # Display force sensor value
                            cv2.putText(frame,
                                    f"Force: {sensor_norm:.3f}",
                                    (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,  # Smaller font scale
                                    (0, 0, 255),
                                    1)    # Thinner line
                            
                            # Display gripper value
                            cv2.putText(frame,
                                    f"Gripper: {gripper_value:.3f}",
                                    (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,  # Smaller font scale
                                    (0, 255, 0),
                                    1)    # Thinner line
                        
                        out.write(frame)
                        
                    out.release()
                    print(f"Saved visualization to {output_path}")
                    
                # Write base video
                write_video(frames, points, f"{base_path}.mp4")
                
                # Write video with sensor overlay if sensor data exists
                if "sensor_states" in observation:
                    write_video(frames, points, f"{base_path}.mp4", include_sensor=True)

    # Save data to a pickle file
    data = {
        "observations": observations,
        "max_cartesian": max_cartesian,
        "min_cartesian": min_cartesian,
        "max_gripper": max_gripper,
        "min_gripper": min_gripper,
        "max_sensor": max_sensor if max_sensor is not None else None,
        "min_sensor": min_sensor if min_sensor is not None else None,
    }
    with open(SAVE_DATA_PATH / "expert_demos" / "franka_env" / f"{TASK_NAME}.pkl", "wb") as f:
        pkl.dump(data, f)
    print(f"Saved data for {TASK_NAME} to {SAVE_DATA_PATH / 'expert_demos' / 'franka_env' / f'{TASK_NAME}.pkl'}")

print("Processing complete.")
