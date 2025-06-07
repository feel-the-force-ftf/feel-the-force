#!/usr/bin/env python3

import warnings
import os
import signal
import sys

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "egl"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pathlib import Path

import hydra
import torch
import numpy as np

import utils
from logger import Logger
from replay_buffer import make_expert_replay_loader
from video import VideoRecorder
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from robot_utils.franka.utils import matrix_to_rotation_6d
warnings.filterwarnings("ignore", category=DeprecationWarning)
torch.backends.cudnn.benchmark = True


def make_agent(obs_spec, action_spec, cfg):
    obs_shape = {}
    for key in cfg.suite.pixel_keys:
        obs_shape[key] = obs_spec[key].shape
    if cfg.use_proprio:
        obs_shape[cfg.suite.proprio_key] = obs_spec[cfg.suite.proprio_key].shape
    obs_shape[cfg.suite.feature_key] = obs_spec[cfg.suite.feature_key].shape
    cfg.agent.obs_shape = obs_shape
    cfg.agent.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg.agent)


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f"workspace: {self.work_dir}")

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        # load data
        dataset_iterable = hydra.utils.call(self.cfg.expert_dataset)
        self.expert_replay_loader = make_expert_replay_loader(
            dataset_iterable, self.cfg.batch_size
        )
        self.expert_replay_iter = iter(self.expert_replay_loader)

        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)
        # create envs
        self.cfg.suite.task_make_fn.max_episode_len = (
            self.expert_replay_loader.dataset._max_episode_len
        )
        self.cfg.suite.task_make_fn.max_state_dim = (
            self.expert_replay_loader.dataset._max_state_dim
        )

        try:
            if self.cfg.suite.use_object_points:
                import yaml

                cfg_path = f"{cfg.root_dir}/point_policy/cfgs/suite/points_cfg.yaml"
                with open(cfg_path) as stream:
                    try:
                        points_cfg = yaml.safe_load(stream)
                    except yaml.YAMLError as exc:
                        print(exc)
                    root_dir, dift_path, cotracker_checkpoint = (
                        points_cfg["root_dir"],
                        points_cfg["dift_path"],
                        points_cfg["cotracker_checkpoint"],
                    )
                    points_cfg["dift_path"] = f"{root_dir}/{dift_path}"
                    points_cfg[
                        "cotracker_checkpoint"
                    ] = f"{root_dir}/{cotracker_checkpoint}"
                self.cfg.suite.task_make_fn.points_cfg = points_cfg
        except:
            pass

        self.env, self.task_descriptions = hydra.utils.call(self.cfg.suite.task_make_fn)

        # create agent
        self.agent = make_agent(
            self.env[0].observation_spec(), self.env[0].action_spec(), cfg
        )

        self.envs_till_idx = len(self.env)
        self.expert_replay_loader.dataset.envs_till_idx = self.envs_till_idx
        self.expert_replay_iter = iter(self.expert_replay_loader)

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None,
            overlay_keypoints=self.cfg.overlay_keypoints,
            calib_path=self.cfg.suite.task_make_fn.calib_path,
        )

        # Set up signal handler for graceful exit
        signal.signal(signal.SIGINT, self.signal_handler)
        self.current_env_idx = 0  # Track current environment for video saving

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.suite.action_repeat

    def signal_handler(self, signum, frame):
        print("\nCaught Ctrl+C, saving video and plot and exiting...")
        
        # Visualize the force and gripper history
        gripper_history = self.env[self.current_env_idx].cumulative_gripper_positions
        desired_force_history = self.env[self.current_env_idx].cumulative_desired_forces
        force_history = self.env[self.current_env_idx].cumulative_actual_forces
        self.env[self.current_env_idx].plot_force_and_gripper(force_history, gripper_history, desired_force_history, work_dir = self.work_dir)

        # Convert BGR to RGB before saving
        if hasattr(self.video_recorder, 'frames') and len(self.video_recorder.frames) > 0:
            self.video_recorder.frames = [frame[..., ::-1] for frame in self.video_recorder.frames]  # BGR to RGB
        self.video_recorder.save(f"{self.global_frame}_env{self.current_env_idx}_interrupted.mp4")
        sys.exit(0)

    def eval(self):
        self.agent.train(False)
        episode_rewards = []
        successes = []
        for env_idx in range(self.envs_till_idx):
            self.current_env_idx = env_idx  # Update current environment index
            print(f"evaluating env {env_idx}")
            episode, total_reward = 0, 0
            eval_until_episode = utils.Until(self.cfg.suite.num_eval_episodes)
            success = []

            while eval_until_episode(episode):
                print("episode", episode)
                time_step = self.env[env_idx].reset()
                self.agent.buffer_reset()
                step = 0

                if episode == 0:
                    self.video_recorder.init(self.env[env_idx], enabled=True)

                while not time_step.last():
                    with torch.no_grad(), utils.eval_mode(self.agent):
                        action = self.agent.act(
                            time_step.observation,
                            self.expert_replay_loader.dataset.stats,
                            step,
                            self.global_step,
                            eval_mode=True,
                        )

                    time_step = self.env[env_idx].step(action)
                    for pixel_key in ['pixels1', 'pixels2', 'pixels3', 'pixels4']:
                        if not any(pixel_key in key for key in action):
                            continue
                        action[f'point_tracks_{pixel_key}'] = action[f'future_tracks_{pixel_key}']
                        if 'future_force_states' in action:
                            action['force'] = action['future_force_states']
                    self.video_recorder.record(self.env[env_idx], (action, time_step.observation))
                    total_reward += time_step.reward
                    step += 1

                episode += 1
                success.append(time_step.observation["goal_achieved"])
                # Convert BGR to RGB before saving
                if hasattr(self.video_recorder, 'frames') and len(self.video_recorder.frames) > 0:
                    self.video_recorder.frames = [frame[..., ::-1] for frame in self.video_recorder.frames]  # BGR to RGB
                self.video_recorder.save(f"{self.global_frame}_env{env_idx}.mp4")
            episode_rewards.append(total_reward / episode)
            successes.append(np.mean(success))

        for _ in range(len(self.env) - self.envs_till_idx):
            episode_rewards.append(0)
            successes.append(0)

        with self.logger.log_and_dump_ctx(self.global_frame, ty="eval") as log:
            for env_idx, reward in enumerate(episode_rewards):
                log(f"episode_reward_env{env_idx}", reward)
                log(f"success_env{env_idx}", successes[env_idx])
            log("episode_reward", np.mean(episode_rewards[: self.envs_till_idx]))
            log("success", np.mean(successes))
            log("episode_length", step * self.cfg.suite.action_repeat / episode)
            log("episode", self.global_episode)
            log("step", self.global_step)

        self.agent.train(True)

    def save_snapshot(self):
        snapshot = self.work_dir / "snapshot.pt"
        self.agent.clear_buffers()
        keys_to_save = ["timer", "_global_step", "_global_episode"]
        payload = {k: self.__dict__[k] for k in keys_to_save}
        payload.update(self.agent.save_snapshot())
        with snapshot.open("wb") as f:
            torch.save(payload, f)

        self.agent.buffer_reset()

    def load_snapshot(self, snapshots):
        # bc
        with snapshots["bc"].open("rb") as f:
            payload = torch.load(f)
        agent_payload = {}
        for k, v in payload.items():
            if k not in self.__dict__:
                agent_payload[k] = v
        self.agent.load_snapshot(agent_payload, eval=True)

    def replay_demo(self):
        """Replay an expert demonstration."""
        self.video_recorder.init(self.env[0], enabled=True)
        
        # Get a batch from expert replay loader
        try:
            import pickle as pkl
            from pathlib import Path
            
            data_path = Path(self.cfg.expert_dataset.path)
            task_name = self.cfg.expert_dataset.tasks # Tasks is a list based on config
            
            demo_path = data_path / Path(task_name + '.pkl')
            with open(demo_path, 'rb') as f:
                dataset_pkl = pkl.load(f)
            first_demo = dataset_pkl['observations'][0]
            # Extract tracks and gripper states from first demo
            robot_tracks1 = []
            robot_tracks2 = []
            gripper = []
            object_tracks1 = []
            object_tracks2 = []
            force = []
            cartesian_states = []
            for i in range(0, len(first_demo[f"robot_tracks_3d_pixels1"]), self.cfg.expert_dataset.subsample):
                robot_tracks1.append(first_demo[f"robot_tracks_3d_pixels1"][i])
                robot_tracks2.append(first_demo[f"robot_tracks_3d_pixels2"][i])
                gripper.append(first_demo["gripper_states"][i])
                object_tracks1.append(first_demo[f"object_tracks_3d_pixels1"][i])
                object_tracks2.append(first_demo[f"object_tracks_3d_pixels2"][i])
                if 'cartesian_states' in first_demo:
                    cartesian_states.append(first_demo["cartesian_states"][i])
                if 'sensor_states' in first_demo:
                    force.append(first_demo["sensor_states"][i])

            # Reset environment to initial state
            time_step = self.env[0].reset()
            step = 0

            # Replay the expert actions
            executed_forces = []
            for i in range(len(robot_tracks1)):
                action = {}
                if self.cfg.agent == "p3po":
                    pos = cartesian_states[i][:3]
                    rot = cartesian_states[i][3:6]
                    rot = R.from_rotvec(rot).as_matrix()
                    rot = matrix_to_rotation_6d(rot[None])[0]
                    action["action"] = np.concatenate([pos, rot, np.expand_dims(np.array(gripper[i]), axis=0)], axis=-1)
                else:
                    action[f"future_tracks_pixels1"] = robot_tracks1[i]
                    action[f"future_tracks_pixels2"] = robot_tracks2[i]
                    action[f"gripper"] = np.expand_dims(np.array(gripper[i]), axis=0)
                if len(force) > 0 and self.cfg.suite.predict_force:
                    action["future_force_states"] = np.expand_dims(np.expand_dims(np.array(force[i]), axis=0), axis=0)
                time_step = self.env[0].step(action)

                # recording data for video
                dataset_observation = {
                    "point_tracks_pixels1": np.concatenate([robot_tracks1[i], object_tracks1[i]], axis=0),
                    "point_tracks_pixels2": np.concatenate([robot_tracks2[i], object_tracks2[i]], axis=0),
                }
                dataset_observation['pixels1'] = []
                dataset_observation['pixels2'] = []
                if len(force) > 0:
                    dataset_observation['force'] = force[i]
                self.video_recorder.record(self.env[0], (dataset_observation, time_step.observation))
                if 'force' in time_step.observation:
                    executed_forces.append(time_step.observation['force'])
                step += 1
                
                if time_step.last():
                    break
            
            # Save the video
            if hasattr(self.video_recorder, 'frames') and len(self.video_recorder.frames) > 0:
                self.video_recorder.frames = [frame[..., ::-1] for frame in self.video_recorder.frames]  # BGR to RGB
            self.video_recorder.save(f"expert_demo_{self.global_frame}.mp4")

            if len(executed_forces) > 0:    
                plt.plot(executed_forces)
                plt.title("executed forces for target force: " + str(self.cfg.suite.desired_force))
                plt.xlabel("time step")
                plt.ylabel("force")
                plt.savefig(f"{self.work_dir}/executed_forces.png")
                plt.close()

        except StopIteration:
            print("No more demonstrations available")
        except Exception as e:
            print(f"Error during demo replay: {e}")


@hydra.main(config_path="cfgs", config_name="config_eval")
def main(cfg):
    workspace = Workspace(cfg)

    if cfg.replay_demo:
        # Just replay the expert demonstration
        workspace.replay_demo()
    else:
        # Load weights and evaluate
        snapshots = {}
        # bc
        bc_snapshot = Path(cfg.bc_weight)
        if not bc_snapshot.exists():
            raise FileNotFoundError(f"bc weight not found: {bc_snapshot}")
        print(f"loading bc weight: {bc_snapshot}")
        snapshots["bc"] = bc_snapshot
        workspace.load_snapshot(snapshots)
        workspace.eval()


if __name__ == "__main__":
    main()
