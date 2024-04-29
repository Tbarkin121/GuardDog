"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.


DOF control methods example
---------------------------
An example that demonstrates various DOF control methods:
- Load cartpole asset from an urdf
- Get/set DOF properties
- Set DOF position and velocity targets
- Get DOF positions
- Apply DOF efforts
"""

import math
from isaacgym import gymutil, gymtorch, gymapi
import time
import numpy as np
from joystick import Joystick
from keyboard import Keyboard
import stm32_comms
import torch

import onnx
import onnxruntime as ort

from torch_jit_utils import to_torch, get_axis_params, torch_rand_float, quat_rotate, quat_rotate_inverse

# comm_obj = stm32_comms.MCU_Comms(0)
max_push_effort = 2.0


def compute_quadwalker_observations(root_states,
                                commands,
                                dof_pos,
                                dof_vel,
                                gravity_vec,
                                actions,
                                lin_vel_scale,
                                ang_vel_scale,
                                dof_vel_scale
                                ):

    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, float, float) -> Tensor
    base_quat = root_states[:, 3:7]
    base_lin_vel = quat_rotate_inverse(base_quat, root_states[:, 7:10]) 
    base_ang_vel = quat_rotate_inverse(base_quat, root_states[:, 10:13])
    projected_gravity = quat_rotate(base_quat, gravity_vec)
    

    commands_scaled = commands*torch.tensor([lin_vel_scale, lin_vel_scale, ang_vel_scale], requires_grad=False)


    sin_encode, cos_encode, motor_angle = convert_angle(dof_pos.squeeze())
    obs = torch.cat((sin_encode.reshape([1,-1]), #12 (0:12)
                     cos_encode.reshape([1,-1]), #12 (12:24)
                     dof_vel * dof_vel_scale, #12 (24:36)
                     base_lin_vel * lin_vel_scale, #3 (36:39)
                     base_ang_vel * ang_vel_scale, #3 (39:42)
                     projected_gravity, #3 (42:45)
                     commands_scaled.reshape([1,-1]), #3 (45:48)
                     actions.reshape([1,-1]) #12 (48:60)
                     ), dim=-1)

    return obs


def convert_angle(angle):
    # Apply sine and cosine functions
    sin_component = torch.sin(angle)
    cos_component = torch.cos(angle)

    #  Normalize angle to [-pi, pi]
    normalized_angle = torch.remainder(angle + np.pi, 2 * np.pi) - np.pi
    # Apply offset
    normalized_angle += np.pi
    # Normalize again if needed
    normalized_angle = torch.remainder(normalized_angle + np.pi, 2 * np.pi) - np.pi
    #  Normalize angle to [-1, 1]
    normalized_angle /= torch.pi

    return sin_component, cos_component, normalized_angle

# initialize gym
gym = gymapi.acquire_gym()

# parse arguments
args = gymutil.parse_arguments(description="Joint control Methods Example")

# create a simulator
sim_params = gymapi.SimParams()
sim_params.substeps = 2
sim_params.dt = 1.0 / 100.0


sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 4 #8
sim_params.physx.num_velocity_iterations = 0 #2

sim_params.physx.num_threads = args.num_threads
sim_params.physx.use_gpu = args.use_gpu

sim_params.use_gpu_pipeline = False
if args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)

if sim is None:
    print("*** Failed to create sim")
    quit()



# add ground plane
plane_params = gymapi.PlaneParams()
# set the normal force to be z dimension
plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
gym.add_ground(sim, plane_params)

# set up the env grid
num_envs = 1
spacing = 1.5
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, 0.0, spacing)

# add cartpole urdf asset
asset_root = "../../IsaacGymEnvs/assets"
asset_file = "urdf/Biped/urdf/Biped.urdf"
# asset_file = "urdf/WalkBot_3DOF_330/urdf/WalkBot_3DOF.urdf"

# Load asset with default control type of position for all joints
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
asset_options.angular_damping = 0.0
asset_options.max_angular_velocity = 10000
asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
cubebot_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
num_dof = gym.get_asset_dof_count(cubebot_asset)

# initial root pose for cartpole actorsto_torch
initial_pose = gymapi.Transform()
initial_pose.p = gymapi.Vec3(0.0, 0.0, 2.0)
initial_pose.r = gymapi.Quat.from_euler_zyx(0.0, 0.0, 0.0)

# Create environment 0
# Cart held steady using position target mode.
# Pole held at a 45 degree angle using position target mode.
env0 = gym.create_env(sim, env_lower, env_upper, 2)
cubebot0 = gym.create_actor(env0, cubebot_asset, initial_pose, 'CubeBot', 0, 1)
# Configure DOF properties
props = gym.get_actor_dof_properties(env0, cubebot0)
props["driveMode"].fill(gymapi.DOF_MODE_EFFORT) 
props["stiffness"].fill(0.0)
props['damping'].fill(0.0)
props['velocity'].fill(25.0)
props['effort'].fill(0.0)
props['friction'].fill(0.02)

props['velocity'][6:8] = 1000.0
props['friction'][6:8] = 0.001

gym.set_actor_dof_properties(env0, cubebot0, props)
# Set DOF drive targets
dof_dict = gym.get_actor_dof_dict(env0, cubebot0)
joint_dict = gym.get_actor_joint_dict(env0, cubebot0)
dof_keys = list(dof_dict.keys())
actor_root_state = gym.acquire_actor_root_state_tensor(sim)
root_states = gymtorch.wrap_tensor(actor_root_state)

# targets = torch.tensor([1000, 0, 0, 0, 0, 0])
# gym.set_dof_velocity_target_tensor(env0, gymtorch.unwrap_tensor(targets))

# create viewer using the default camera properties
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise ValueError('*** Failed to create viewer')

print('Tester')
# Look at the first env
cam_pos = gymapi.Vec3(2, 1, 1)
cam_target = initial_pose.p
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# Simulate
joint_idx = 0
control_idx = 0
loop_counter = 1
max_loops = 250

# joy = Joystick()
key = Keyboard(3)


dof_state_tensor = gym.acquire_dof_state_tensor(sim)
dof_state = gymtorch.wrap_tensor(dof_state_tensor)
print(dof_state)
dof_pos = dof_state.view(num_envs, num_dof, 2)[..., 0]
dof_vel = dof_state.view(num_envs, num_dof, 2)[..., 1]

# positions =  0.0 * (torch.rand((1,num_dof)) - 0.5) - np.pi
positions =  torch.zeros((1,num_dof))

# velocities = 2.0 * (torch.rand((1)) - 0.5)
dof_pos[0, :] = positions[:]
# dof_vel[0, :] = velocities[:]

env_ids = torch.tensor([0])
env_ids_int32 = env_ids.to(dtype=torch.int32)
gym.set_dof_state_tensor_indexed(sim,
                                gymtorch.unwrap_tensor(dof_state),
                                gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

onnx_model = onnx.load("QuadWalker.onnx")
ort_model = ort.InferenceSession("QuadWalker.onnx")
actions = torch.zeros([12], dtype=torch.float32)
gravity_vec = to_torch(get_axis_params(-1., 1), device='cpu').reshape(1,-1)
lin_vel_scale = 2.0
ang_vel_scale = 0.25
dof_vel_scale = 0.05
while not gym.query_viewer_has_closed(viewer):
    gym.refresh_actor_root_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_dof_force_tensor(sim)

    # print(root_states)

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # a = joy.get_axis()
    # a = key.get_keys()
    # scale = torch.tensor([3., 2., 1.])
    # commands = a*scale
    
    # obs = compute_quadwalker_observations(root_states,
    #                             commands,
    #                             dof_pos,
    #                             dof_vel,
    #                             gravity_vec,
    #                             actions,
    #                             lin_vel_scale,
    #                             ang_vel_scale,
    #                             dof_vel_scale
    #                             )

    # output = ort_model.run(
    #     None,
    #     {"obs": obs.reshape([1,-1]).numpy().astype(np.float32)},
    #     )
    #     # print(outputs[0])
    # actions[:] = torch.tensor(output[0])*max_push_effort

    a = key.get_keys()
    scale = torch.ones(12)
    print(a)
    print(scale)
    actions = a[0]*scale
    gym.apply_dof_effort(env0, 0, actions[0])
    gym.apply_dof_effort(env0, 1, actions[1])
    gym.apply_dof_effort(env0, 2, actions[2])
    gym.apply_dof_effort(env0, 3, actions[3])
    gym.apply_dof_effort(env0, 4, actions[4])
    gym.apply_dof_effort(env0, 5, actions[5])
    gym.apply_dof_effort(env0, 6, actions[6])
    gym.apply_dof_effort(env0, 7, actions[7])
    # gym.apply_dof_effort(env0, 8, actions[8])
    # gym.apply_dof_effort(env0, 9, actions[9])
    # gym.apply_dof_effort(env0, 10, actions[10])
    # gym.apply_dof_effort(env0, 11, actions[11])
    

 
    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

print('Done')

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)

