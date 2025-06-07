import sys

sys.path.append("../../")

import re
import yaml
import argparse
import pickle as pkl
from pathlib import Path
import cv2
import torch
import numpy as np
from pandas import read_csv
from point_utils.points_class import PointsClass
from utils import (
    camera2pixelkey,
    sliding_window_outlier_filter,
    filter_and_interpolate_fingertips,
    pixel2d_to_3d_torch,
    triangulate_points,

)

# Create the parser
parser = argparse.ArgumentParser(
    description="Convert processed human data into a pkl file"
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
parser.add_argument(
    "--calibrate", action="store_true", help="Calibrate the camera"
)

parser.add_argument(
    '--camera', type=int, nargs='+', default=[1, 2],help='Camera ID'
)
parser.add_argument(
    '--smooth_tracks', action='store_true', help='Smooth the tracks'
)

args = parser.parse_args()
DATA_DIR = Path(args.data_dir)
CALIB_PATH = Path(args.calib_path)
task_names = args.task_names
NUM_DEMOS = args.num_demos
process_points = args.process_points
use_gt_depth = args.use_gt_depth

camera_indices = args.camera
original_img_size = (640, 480)
crop_h, crop_w = (0.0, 1.0), (0.0, 1.0)
save_img_size = None
object_labels = [
    "human_hand",
    "objects",
]

PROCESSED_DATA_PATH = Path(DATA_DIR) / "processed_data"
SAVE_DATA_PATH = Path(DATA_DIR) / "processed_data_pkl"

if save_img_size is None:
    save_img_size = (
        int(original_img_size[0] * (crop_w[1] - crop_w[0])),
        int(original_img_size[1] * (crop_h[1] - crop_h[0])),
    )

assert len(task_names) == 1, "Only one task name is supported for now"

if task_names is None:
    task_names = [x.name for x in PROCESSED_DATA_PATH.iterdir() if x.is_dir()]

SAVE_DATA_PATH.mkdir(parents=True, exist_ok=True)

# Calibration data
if args.calibrate:
    calibration_data = None
else:
    calibration_data = np.load(CALIB_PATH, allow_pickle=True).item()

episode_list = {}
for cam_idx in camera_indices:
    pixel_key = f"pixels{cam_idx}"
    episode_list[pixel_key] = []


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
        cfg["smooth_tracks"] = args.smooth_tracks
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
    if use_gt_depth:
        TASK_NAME = f"{TASK_NAME}_gt_depth"

    if (SAVE_DATA_PATH / f"{TASK_NAME}.pkl").exists() and not process_points:
        print(f"Data for {TASK_NAME} already exists. Appending to it...")
        input("Press Enter to continue...")
        data = pkl.load(open(SAVE_DATA_PATH / f"{TASK_NAME}.pkl", "rb"))
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
    # detect_hand_rates = []
    for i, data_point in enumerate(sorted(dirs, key=lambda x: int(str(x).split("_")[-1]))):
        print(f"Processing data point {i+1}/{len(dirs)}")

        if NUM_DEMOS is not None and int(str(data_point).split("_")[-1]) >= NUM_DEMOS:
            print(f"Skipping data point {data_point}")
            continue

        observation = {}
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

            for idx in camera_indices:
                depth_file = depth_dir / f"depth{idx}.pkl"
                with open(depth_file, "rb") as f:
                    depth = pkl.load(f)  # depth in meters
                camera_name = f"cam_{idx}"
                pixel_key = camera2pixelkey[camera_name]
                observation[f"depth_{pixel_key}"] = depth

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
                points_class.add_to_image_list(frames[0], pixel_key) # only deal with the first frame
                for object_label in object_labels:
                    points_class.find_semantic_similar_points(pixel_key, object_label)
                try:
                    points_class.track_points(
                        pixel_key, last_n_frames=mark_every, is_first_step=True
                    )
                except Exception as e:
                    print(str(e))
                    print(f"Error in tracking hand points for {pixel_key}")
                    points_class.reset_episode()
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
                        depth = observation[f"depth_{pixel_key}"][0]
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

                for idx, image in enumerate(frames[1:]): # skip the first frame
                    print(f"Traj: {i}, Frame: {idx}, Image: {pixel_key}")
                    points_class.add_to_image_list(image, pixel_key) #add this frame to image list
                    if use_gt_depth:
                        depth = observation[f"depth_{pixel_key}"][idx]
                        points_class.set_depth(
                            depth,
                            pixel_key,
                            original_img_size,
                            save_img_size,
                            (crop_h, crop_w),
                        )

                    if (idx + 1) % mark_every == 0 or idx == (len(frames) - 2):
                        to_add = mark_every - (idx + 1) % mark_every
                        if to_add < mark_every: # which is when idx == (len(frames) - 2)
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
                        ) # torch.Size([8, 16, 2])
                        # points[3]  which is the frame
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
                                points3d = pixel2d_to_3d_torch(points[j], depth, K, P)
                                points_3d_list.append(points3d)
                
                if args.smooth_tracks:
                    cleaned_tensors = sliding_window_outlier_filter(points_list)
                    indices, interpolated_points = filter_and_interpolate_fingertips(cleaned_tensors)
                    final_points = interpolated_points
                else:
                    final_points = points_list
                observation[f"human_tracks_{pixel_key}"] = torch.stack(
                    final_points
                ).numpy()
                if use_gt_depth:
                    observation[f"human_tracks_3d_{pixel_key}"] = torch.stack(
                        points_3d_list
                    ).numpy()
                points_class.reset_episode()

            if save == False: 
                continue
        
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
                observation[f"human_tracks_3d_{pixel_key}"] = []
            for t_idx in range(
                len(observation[f"human_tracks_{pixel_key}"])
            ):  # for each frame
                P, pts = [], []

                for cam_idx in camera_indices:
                    camera_name = f"cam_{cam_idx}"
                    pixel_key = camera2pixelkey[camera_name]

                    extr = calibration_data[camera_name]["ext"]
                    intr = calibration_data[camera_name]["int"]
                    intr = np.concatenate([intr, np.zeros((3, 1))], axis=1)
                    P.append(intr @ extr)
                    pt2d = observation[f"human_tracks_{pixel_key}"][t_idx]
                    # compute point_h in original image
                    point_h = pt2d[:, 1]
                    h_orig, h_curr = original_img_size[1], save_img_size[1]
                    h_orig_cropped = h_orig * (crop_h[1] - crop_h[0])
                    point_h_orig = (
                        point_h / h_curr
                    ) * h_orig_cropped + h_orig * crop_h[0]
                    point_h_orig = point_h_orig.astype(np.int32)

                    # compute point_w in original image
                    point_w = pt2d[:, 0]
                    w_orig, w_curr = original_img_size[0], save_img_size[0]
                    w_orig_cropped = w_orig * (crop_w[1] - crop_w[0])
                    point_w_orig = (
                        point_w / w_curr
                    ) * w_orig_cropped + w_orig * crop_w[0]
                    point_w_orig = point_w_orig.astype(np.int32)

                    pt2d = np.column_stack((point_w_orig, point_h_orig))

                    pts.append(pt2d)

                pts3d = triangulate_points(P, pts)
     
                for cam_idx in camera_indices:
                    camera_name = f"cam_{cam_idx}"
                    pixel_key = camera2pixelkey[camera_name]
                    observation[f"human_tracks_3d_{pixel_key}"].append(pts3d[:, :3])
            for cam_idx in camera_indices:
                camera_name = f"cam_{cam_idx}"
                pixel_key = camera2pixelkey[camera_name]
                observation[f"human_tracks_3d_{pixel_key}"] = np.array(
                    observation[f"human_tracks_3d_{pixel_key}"]
                )

        observations.append(observation)
    # Create video with overlaid points
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
                points = observation[f"human_tracks_{pixel_key}"]
          
                # Initialize video writer
                base_path = str(video_dir / f"{TASK_NAME}_demo_{i}_{pixel_key}")
                height, width = frames[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                
                def write_video(frames, points, output_path, include_sensor=False):
                    out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))
                    
                    for frame_idx, (frame, frame_points) in enumerate(zip(frames, points)):
                        frame = frame.copy()
                        
                        # Draw points
                        for point in frame_points:
                            x, y = point.astype(int)
                            cv2.circle(frame, (x, y), radius=3, color=(0, 255, 0), thickness=-1)
                        
                        # Add sensor overlay if requested
                        if include_sensor:
                            sensor_norm = observation["sensor_states"][frame_idx]
                            cv2.putText(frame,
                                    f"Sensor Norm: {sensor_norm:.3f}",
                                    (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1,
                                    (0, 0, 255),
                                    2)
                        
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
    with open(SAVE_DATA_PATH / f"{TASK_NAME}.pkl", "wb") as f:
        pkl.dump(data, f)
    print(f"Saved data for {TASK_NAME} to {SAVE_DATA_PATH / f'{TASK_NAME}.pkl'}")

print("Processing complete.")
