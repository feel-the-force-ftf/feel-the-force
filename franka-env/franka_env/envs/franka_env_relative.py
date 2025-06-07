import numpy as np
from scipy.spatial.transform import Rotation as R

from franka_env.envs.franka_env import FrankaEnv


class FrankaEnvRelative(FrankaEnv):
    def step(self, rel_action):
        current_state = self.franka_state

        pos_curr = current_state.pos
        ori_curr = current_state.quat
        r_curr = R.from_quat(ori_curr).as_matrix()
        matrix_curr = np.eye(4)
        matrix_curr[:3, :3] = r_curr
        matrix_curr[:3, 3] = pos_curr

        # find transformation matrix
        pos_delta = rel_action[:3]
        ori_delta = rel_action[3:6]
        r_delta = R.from_rotvec(ori_delta).as_matrix()
        matrix_delta = np.eye(4)
        matrix_delta[:3, :3] = r_delta
        matrix_delta[:3, 3] = pos_delta

        # find desired matrix
        matrix_desired = matrix_curr @ matrix_delta

        pos_desired = pos_curr + pos_delta
        r_desired = matrix_desired[:3, :3]
        ori_desired = R.from_matrix(r_desired).as_quat()
        desired_cartesian_pose = np.concatenate([pos_desired, ori_desired])

        action = np.concatenate([desired_cartesian_pose, rel_action[6:]])
        return super().step(action)
