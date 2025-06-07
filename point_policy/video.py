import cv2
import imageio
import numpy as np
from pathlib import Path


class VideoRecorder:
    def __init__(self, root_dir, calib_path, overlay_keypoints=False, render_size=(640, 480), fps=20):
        if root_dir is not None:
            self.save_dir = root_dir / "eval_video"
            self.save_dir.mkdir(exist_ok=True)
        else:
            self.save_dir = None

        self.fps = fps
        self.frames = []
        self.original_image_size = (640, 480)
        self.render_size = render_size
        self.overlay_keypoints = overlay_keypoints
        self.calib_path = calib_path
        CALIB_PATH = Path(calib_path)
        self.calibration_data = np.load(CALIB_PATH, allow_pickle=True).item()

    def init(self, env, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled
        self.record(env)

    def record(self, env, observation=None):
        if self.enabled:
            if hasattr(env, "physics"):
                frame = env.physics.render(
                    height=self.render_size[1], width=self.render_size[0], camera_id=0
                )
            else:
                frame = env.render(width=self.render_size[0], height=self.render_size[1])

            if observation is not None:
                if isinstance(observation, tuple):
                    observation, time_step_observation = observation
                else:
                    time_step_observation = observation
                    observation = None
            else:
                time_step_observation = None
            points = []
            if observation is not None:
                if self.overlay_keypoints:
                    for pixel_key in ['pixels1', 'pixels2', 'pixels3', 'pixels4']:
                        if not any(pixel_key in key for key in observation):
                            continue
                        points.append(observation[f'point_tracks_{pixel_key}'])
                    num_pixel_keys = len(points)
                    if num_pixel_keys > 0:
                        subframe_size = self.render_size[0] // num_pixel_keys

                        overlayed_frame = frame.copy()
                        for i in range(num_pixel_keys):
                            subframe = frame[:, i*subframe_size:(i+1)*subframe_size, :]
                            subframe_points = points[i]
                            camera_name = f'cam_{i+1}'
                            # project 3d point to 2d image coordinates
                            P = self.calibration_data[camera_name]["ext"]
                            K = self.calibration_data[camera_name]["int"]
                            D = self.calibration_data[camera_name]["dist_coeff"]
                            r, t = P[:3, :3], P[:3, 3]
                            r, _ = cv2.Rodrigues(r)
                            for point_num, point in enumerate(subframe_points):
                                point_2d = cv2.projectPoints(point, r, t, K, D)[0].squeeze()
                                point_2d[0] = int(point_2d[0] * subframe.shape[1] / self.original_image_size[0])
                                point_2d[1] = int(point_2d[1] * subframe.shape[0] / self.original_image_size[1])
                                color = (0, 255, 0) if point_num >= 9 else (255, 0, 0)
                                cv2.circle(subframe, tuple(point_2d.astype(int)), 2, color, -1)
                            overlayed_frame[:, i*subframe_size:(i+1)*subframe_size, :] = subframe
                        frame = overlayed_frame
                if 'force' in observation:
                    force = observation['force']
                    if isinstance(force, np.ndarray):
                        force = force.item()
                    force_text = f'Force: {force:.2f}'
                    frame_width = frame.shape[1]
                    text_x = frame_width - 200
                    cv2.rectangle(frame, (frame_width - 205, 5), (frame_width - 5, 40), (0, 0, 0), -1)
                    cv2.putText(frame, force_text, (text_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            
            points = []
            if time_step_observation is not None:
                if self.overlay_keypoints:
                    for pixel_key in ['pixels1', 'pixels2', 'pixels3', 'pixels4']:
                        if not any(pixel_key in key for key in time_step_observation):
                            continue
                        points.append(time_step_observation[f'point_tracks_{pixel_key}'])
                    num_pixel_keys = len(points)
                    subframe_size = self.render_size[0] // num_pixel_keys

                    overlayed_frame = frame.copy()
                    for i in range(num_pixel_keys):
                        subframe = frame[:, i*subframe_size:(i+1)*subframe_size, :]
                        subframe_points = points[i]
                        camera_name = f'cam_{i+1}'
                        P = self.calibration_data[camera_name]["ext"]
                        K = self.calibration_data[camera_name]["int"]
                        D = self.calibration_data[camera_name]["dist_coeff"]
                        r, t = P[:3, :3], P[:3, 3]
                        r, _ = cv2.Rodrigues(r)
                        for point_num, point in enumerate(subframe_points):
                            point_2d = cv2.projectPoints(point, r, t, K, D)[0].squeeze()
                            point_2d[0] = int(point_2d[0] * subframe.shape[1] / self.original_image_size[0])
                            point_2d[1] = int(point_2d[1] * subframe.shape[0] / self.original_image_size[1])
                            color = (255, 0, 255) if point_num >= 9 else (0, 0, 255)
                            cv2.circle(subframe, tuple(point_2d.astype(int)), 2, color, -1)
                        overlayed_frame[:, i*subframe_size:(i+1)*subframe_size, :] = subframe
                    frame = overlayed_frame
                if 'force' in time_step_observation:
                    force = time_step_observation['force']
                    force_text = f'Force: {force:.2f}'
                    cv2.rectangle(frame, (5, 5), (200, 40), (0, 0, 0), -1)  
                    cv2.putText(frame, force_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            path = self.save_dir / file_name
            imageio.mimsave(str(path), self.frames, fps=self.fps)


class TrainVideoRecorder:
    def __init__(self, root_dir, render_size=256, fps=20):
        if root_dir is not None:
            self.save_dir = root_dir / "train_video"
            self.save_dir.mkdir(exist_ok=True)
        else:
            self.save_dir = None

        self.render_size = render_size
        self.fps = fps
        self.frames = []

    def init(self, obs, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled
        self.record(obs)

    def record(self, obs):
        if self.enabled:
            frame = cv2.resize(
                obs[-3:].transpose(1, 2, 0),
                dsize=(self.render_size, self.render_size),
                interpolation=cv2.INTER_CUBIC,
            )
            self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            path = self.save_dir / file_name
            imageio.mimsave(str(path), self.frames, fps=self.fps)
