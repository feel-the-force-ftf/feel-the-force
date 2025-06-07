#!/usr/bin/env python3
import gym
import numpy as np
import franka_env
import time

env = gym.make(
    "Franka-v1",
    height=480,
    width=640,
    use_robot=True,
    use_gt_depth=False,
    force_controller=True,
    read_force=True,
    desired_force=80,
    force_match_tolerance=5,
)
state = env.get_state()
env.reset(franka_state=state)
action = np.concatenate([state.pos, state.quat, [-1.0]])
state = env.step(action)
action = np.concatenate([state[0]['features'], [1.0]])
state = env.step(action)


