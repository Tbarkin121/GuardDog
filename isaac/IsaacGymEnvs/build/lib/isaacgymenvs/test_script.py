from isaacgym import gymutil, gymtorch, gymapi
from isaacgym.torch_utils import *
from isaacgym.terrain_utils import *

import torch
import yaml

from tasks.dynasoar_env import DynasoarEnv

with open("cfg/task/Dynasoar.yaml", "r") as cfg:
    try:
        cfg = yaml.safe_load(cfg)
    except yaml.YAMLError as exc:
        print(exc)
# print(cfg)


env = DynasoarEnv(cfg)
print(env.kinematic_states[0,...])
env.step(env.actions_buf)
print(env.kinematic_states[0,...])
