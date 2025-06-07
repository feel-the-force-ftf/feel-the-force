import cv2
import gym
import numpy as np
import time
import pickle

from frankateach.constants import (
    CAM_PORT,
    GRIPPER_OPEN,
    HOST,
    CONTROL_PORT,
)
from frankateach.messages import FrankaAction, FrankaState
from frankateach.network import (
    ZMQCameraSubscriber,
    create_request_socket,
)
from frankateach.sensors.reskin import ReskinSensorPublisher
try:
    from frankateach.sensors.reskin import ReskinSensorSubscriber
except ImportError:
    print("ReskinSensorSubscriber not found")
    ReskinSensorSubscriber = None
import matplotlib.pyplot as plt
import os 
import csv
from datetime import datetime

class FrankaEnv(gym.Env):
    def __init__(
        self,
        width=640,
        height=480,
        use_robot=True,
        use_gt_depth=False,
        crop_h=None,
        crop_w=None,
        force_controller=False,
        read_force=False,
        desired_force=50,
        force_match_tolerance=5,
        variable_desired_force=False,
        sensor_type="reskin",
        sensor_params={},
    ):
        super(FrankaEnv, self).__init__()
        self.cumulative_gripper_positions = []
        self.cumulative_desired_forces = []
        self.cumulative_actual_forces = []

        self.width = width
        self.height = height
        self.crop_h = crop_h
        self.crop_w = crop_w

        self.channels = 3
        self.feature_dim = 8
        self.action_dim = 7  # (pos, axis angle, gripper)
        self.readings_per_pd_step = 200

        self.use_robot = use_robot
        self.sensor_type = sensor_type
        if sensor_type is not None and read_force:
            assert sensor_type in ["reskin"]
            assert (
                ReskinSensorSubscriber is not None
            ), "ReskinSensorSubscriber not found"
            if sensor_type == "reskin":
                self.n_sensors = 2
                self.sensor_dim = 3    # only use data from center sensor
        self.sensor_params = sensor_params
        self.use_robot = use_robot
        self.use_gt_depth = use_gt_depth
        self.force_controller = force_controller
        self.read_force = read_force
        self.desired_force = desired_force
        self.force_match_tolerance = force_match_tolerance
        self.variable_desired_force = variable_desired_force
        self.n_channels = 3
        self.reward = 0

        self.franka_state = None
        self.curr_images = None

        self.action_space = gym.spaces.Box(
            low=-float("inf"), high=float("inf"), shape=(self.action_dim,)
        )
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(height, width, self.n_channels), dtype=np.uint8
        )

        if self.use_robot:
            self.cam_ids = [1, 2]
            self.image_subscribers = {}
            if self.use_gt_depth:
                self.depth_subscribers = {}
            for cam_idx in self.cam_ids:
                port = CAM_PORT + cam_idx
                self.image_subscribers[cam_idx] = ZMQCameraSubscriber(
                    host=HOST,
                    port=port,
                    topic_type="RGB",
                )

                if self.use_gt_depth:
                    depth_port = CAM_PORT + cam_idx + 1000  # depth offset =1000
                    self.depth_subscribers[cam_idx] = ZMQCameraSubscriber(
                        host=HOST,
                        port=depth_port,
                        topic_type="Depth",
                    )
            if self.sensor_type == "reskin" and self.read_force:
                self.sensor_subscriber = ReskinSensorSubscriber()

                self.sensor_prev_state = None
                self.subtract_sensor_baseline = sensor_params.get("subtract_sensor_baseline", 1)
                self.sensor_baseline, self.sensor_prev_state = self.set_baseline()
            self.action_request_socket = create_request_socket(HOST, CONTROL_PORT)

    def set_baseline(self):
        baseline_meas = []
        while len(baseline_meas) < 5:
            sensor_state = self.sensor_subscriber.get_sensor_state()
            sensor_values = np.array(sensor_state["sensor_values"], dtype=np.float32)
            baseline_meas.append(sensor_values)
        return np.mean(baseline_meas, axis=0),  sensor_values - np.mean(baseline_meas, axis=0)

    def get_state(self):
        self.action_request_socket.send(b"get_state")
        franka_state: FrankaState = pickle.loads(self.action_request_socket.recv())
        self.franka_state = franka_state
        return franka_state

    def reskin_state_to_force(self, reskin_state):
        force = np.linalg.norm(reskin_state["sensor0"])
        return force

    def force_step(self, pos, quat, gripper, desired_force, force_match_tolerance):
        franka_state = self.get_state()
        reskin_state = self.get_reskin_state()  
        curr_force = self.reskin_state_to_force(reskin_state)
        force_history = [curr_force]
        gripper_history = [gripper]  
        timestep = 0
        while not np.count_nonzero(np.abs(desired_force - np.array(force_history[-1:])) < force_match_tolerance) >= 1:
            gripper_action = (
                0.001 * (desired_force - curr_force) 
            )
            timestep += 1
            gripper = self.get_state().gripper
            gripper = gripper + gripper_action
            print("gripper", gripper)
            gripper = np.clip(gripper, -1, 1)
            franka_action = FrankaAction(
                pos=pos,
                quat=quat,
                gripper=gripper,
                reset=False,
                timestamp=time.time(),
            )
            self.action_request_socket.send(bytes(pickle.dumps(franka_action, protocol=-1)))
            franka_state: FrankaState = pickle.loads(self.action_request_socket.recv())

            gripper = self.get_state().gripper

            curr_force = 0
            for _ in range(self.readings_per_pd_step):
                reskin_state = self.get_reskin_state()
                curr_force += self.reskin_state_to_force(reskin_state)
            curr_force = curr_force / self.readings_per_pd_step
            print()
            print("force", curr_force)
            print("desired force", desired_force)
            force_history.append(curr_force)
            gripper_history.append(gripper)
        
        if len(force_history) == 1:
            gripper_action = (
                0.001 * (desired_force - curr_force) 
            )
            gripper = self.get_state().gripper
            gripper = gripper + gripper_action
            print("gripper", gripper)
            gripper = np.clip(gripper, -1, 1)
            franka_action = FrankaAction(
                pos=pos,
                quat=quat,
                gripper=gripper,
                reset=False,
                timestamp=time.time(),
            )
            self.action_request_socket.send(bytes(pickle.dumps(franka_action, protocol=-1)))
            franka_state: FrankaState = pickle.loads(self.action_request_socket.recv())

        self.franka_state = franka_state
        self.cumulative_gripper_positions.extend(gripper_history[1:])
        self.cumulative_desired_forces.extend([desired_force]*len(gripper_history[1:]))
        self.cumulative_actual_forces.extend(force_history[1:])
        return franka_state

    def plot_force_and_gripper(self, force_history, gripper_history, desired_force_history, work_dir):
        """Plots force magnitude (actual and desired) and gripper position over time and saves data for future use."""
        time_steps = np.arange(len(force_history))

        fig, ax1 = plt.subplots(figsize=(10, 5))

        # Plot Actual Force on left y-axis
        ax1.set_xlabel("Time Step")
        ax1.set_ylabel("Force Magnitude", color="tab:red")
        ax1.plot(time_steps, force_history, color="tab:red", label="Actual Force", linestyle="-")  # Red solid line
        ax1.plot(time_steps, desired_force_history, color="tab:red", label="Desired Force", linestyle="--")  # Red dashed line
        ax1.tick_params(axis="y", labelcolor="tab:red")

        force_min = min(min(force_history), min(desired_force_history)) * 0.9 
        force_max = max(max(force_history), max(desired_force_history)) * 1.1  
        ax1.set_ylim(force_min, force_max)

        # Plot Gripper Position on right y-axis
        ax2 = ax1.twinx()
        ax2.set_ylabel("Gripper Position", color="tab:blue")
        ax2.plot(time_steps, gripper_history, color="tab:blue", label="Gripper Position", linestyle="-")  # Blue solid line
        ax2.tick_params(axis="y", labelcolor="tab:blue")
        ax2.set_ylim(-1, 1)

        # Add title and legend
        fig.suptitle("Force Magnitude (Actual & Desired) & Gripper Position Over Time")
        fig.legend(loc="upper right")

        # Create directories
        save_dir = work_dir / "eval_plot"
        os.makedirs(save_dir, exist_ok=True)

        # Generate filename
        timestamp = datetime.now().strftime("%m-%d_%H-%M-%S")
        filename = f"force_gripper_plot_{timestamp}"

        # Save plot
        plot_path = os.path.join(save_dir, f"{filename}.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()

        # Save raw data
        data_path = os.path.join(save_dir, f"{filename}.csv")
        with open(data_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Time Step", "Actual Force", "Gripper Position", "Desired Force"])
            for i in range(len(force_history)):
                writer.writerow([i, force_history[i], gripper_history[i], desired_force_history[i]])

        print(f"Plot saved at: {plot_path}")
        print(f"Raw data saved at: {data_path}")

    def step(self, abs_action):
        pos = abs_action[:3]
        quat = abs_action[3:7]
        if len(abs_action) == 8:
            gripper = abs_action[-1]
        elif len(abs_action) == 9:
            gripper = abs_action[-2]
            if self.variable_desired_force:
                self.desired_force = abs_action[-1]
        else:
            raise ValueError("Invalid action length")
        
        if self.force_controller and gripper == 1:
            franka_state = self.force_step(pos, quat, gripper, desired_force=self.desired_force, force_match_tolerance=self.force_match_tolerance)
        else:
            # Send action to the robot
            franka_action = FrankaAction(
                pos=pos,
                quat=quat,
                gripper=gripper,
                reset=False,
                timestamp=time.time(),
            )

            self.action_request_socket.send(bytes(pickle.dumps(franka_action, protocol=-1)))
            franka_state: FrankaState = pickle.loads(self.action_request_socket.recv())
            self.cumulative_gripper_positions.append(franka_state.gripper)
            if self.force_controller:
                reskin_state = self.get_reskin_state()
                curr_force = self.reskin_state_to_force(reskin_state)
            else:
                curr_force = 0
            self.cumulative_desired_forces.append(curr_force)
            self.cumulative_actual_forces.append(curr_force)
        self.franka_state = franka_state

        image_list = {}
        for cam_idx, subscriber in self.image_subscribers.items():
            image, _ = subscriber.recv_rgb_image()

            # crop the image
            if self.crop_h is not None and self.crop_w is not None:
                h, w, _ = image.shape
                image = image[
                    int(h * self.crop_h[0]) : int(h * self.crop_h[1]),
                    int(w * self.crop_w[0]) : int(w * self.crop_w[1]),
                ]

            image_list[cam_idx] = image

        if self.use_gt_depth:
            depth_list = {}
            for cam_idx, subscriber in self.depth_subscribers.items():
                depth, _ = subscriber.recv_depth_image()

                if self.crop_h is not None and self.crop_w is not None:
                    h, w = depth.shape
                    depth = depth[
                        int(h * self.crop_h[0]) : int(h * self.crop_h[1]),
                        int(w * self.crop_w[0]) : int(w * self.crop_w[1]),
                    ]

                depth_list[cam_idx] = depth

        self.curr_images = image_list

        obs = {
            "features": np.concatenate(
                (franka_state.pos, franka_state.quat, [franka_state.gripper])
            ),
        }

        for cam_idx, image in image_list.items():
            obs[f"pixels{cam_idx}"] = cv2.resize(image, (self.width, self.height))
        if self.use_gt_depth:
            for cam_idx, depth in depth_list.items():
                obs[f"depth{cam_idx}"] = cv2.resize(depth, (self.width, self.height))
        
        if self.sensor_type == "reskin" and self.read_force:
            try:
                reskin_state = self.get_reskin_state()
                force = {'force': self.reskin_state_to_force(reskin_state)}
                obs.update(force)
            except KeyError:
                pass

        return obs, self.reward, False, None

    def reset(self, franka_state=None):
        if self.use_robot:
            if franka_state is None:
                print("resetting")
                franka_action = FrankaAction(
                    pos=np.zeros(3),
                    quat=np.zeros(4),
                    gripper=GRIPPER_OPEN,
                    reset=True,
                    timestamp=time.time(),
                )

                self.action_request_socket.send(
                    bytes(pickle.dumps(franka_action, protocol=-1))
                )
                franka_state: FrankaState = pickle.loads(self.action_request_socket.recv())
                self.franka_state = franka_state
        
                print("reset done: ", franka_state)
            else:
                self.franka_state = franka_state

            image_list = {}
            for cam_idx, subscriber in self.image_subscribers.items():
                image, _ = subscriber.recv_rgb_image()

                # crop the image
                if self.crop_h is not None and self.crop_w is not None:
                    h, w, _ = image.shape
                    image = image[
                        int(h * self.crop_h[0]) : int(h * self.crop_h[1]),
                        int(w * self.crop_w[0]) : int(w * self.crop_w[1]),
                    ]

                image_list[cam_idx] = image

            if self.use_gt_depth:
                depth_list = {}
                for cam_idx, subscriber in self.depth_subscribers.items():
                    depth, _ = subscriber.recv_depth_image()

                    if self.crop_h is not None and self.crop_w is not None:
                        h, w = depth.shape
                        depth = depth[
                            int(h * self.crop_h[0]) : int(h * self.crop_h[1]),
                            int(w * self.crop_w[0]) : int(w * self.crop_w[1]),
                        ]

                    depth_list[cam_idx] = depth

            self.curr_images = image_list

            obs = {
                "features": np.concatenate(
                    (franka_state.pos, franka_state.quat, [franka_state.gripper])
                ),
            }
            for cam_idx, image in image_list.items():
                obs[f"pixels{cam_idx}"] = cv2.resize(image, (self.width, self.height))
            if self.use_gt_depth:
                for cam_idx, depth in depth_list.items():
                    obs[f"depth{cam_idx}"] = cv2.resize(
                        depth, (self.width, self.height)
                    )
            if self.sensor_type == "reskin" and self.read_force:
                try:
                    reskin_state = self.get_reskin_state(update_baseline=True)
                    force = {'force': self.reskin_state_to_force(reskin_state)}
                    obs.update(force)
                except KeyError:
                    pass

            return obs

        else:
            obs = {}
            obs["features"] = np.zeros(self.feature_dim)
            obs["pixels"] = np.zeros((self.height, self.width, self.n_channels))
            if self.use_gt_depth:
                obs["depth"] = np.zeros((self.height, self.width))

            return obs

    def get_reskin_state(self, update_baseline=False):
        sensor_state = self.sensor_subscriber.get_sensor_state()
        sensor_values = np.array(sensor_state["sensor_values"], dtype=np.float32)
        if update_baseline:
            self.sensor_baseline, self.sensor_prev_state = self.set_baseline()
        if self.subtract_sensor_baseline:
            sensor_values = sensor_values - self.sensor_baseline
        sensor_diff = sensor_values - self.sensor_prev_state
        self.sensor_prev_state = sensor_values 
        sensor_keys = [f"sensor{sensor_idx}" for sensor_idx in range(self.n_sensors)]
        reskin_state = {}
        for sidx, sensor_key in enumerate(sensor_keys):
            reskin_state[sensor_key] = sensor_values[
                sidx * self.sensor_dim : (sidx + 1) * self.sensor_dim
            ]
            reskin_state[f"{sensor_key}_diffs"] = sensor_diff[
                sidx * self.sensor_dim : (sidx + 1) * self.sensor_dim
            ]
        return reskin_state
    
    def render(self, mode="rgb_array", width=640, height=480):
        assert self.curr_images is not None, "Must call reset() before render()"
        if mode == "rgb_array":
            image_list = []
            for key, im in self.curr_images.items():
                image_list.append(cv2.resize(im, (width, height)))

            return np.concatenate(image_list, axis=1)
        else:
            raise NotImplementedError
