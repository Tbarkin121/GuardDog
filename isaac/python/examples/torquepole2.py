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

comm_obj = stm32_comms.MCU_Comms()

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
asset_file = "urdf/TorquePole/urdf/TorquePole.urdf"
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

# initial root pose for cartpole actors
initial_pose = gymapi.Transform()
initial_pose.p = gymapi.Vec3(0.0, 0.0, 1.0)
initial_pose.r = gymapi.Quat.from_euler_zyx(1.5708, 0.0, 0.0)

# Create environment 0
# Cart held steady using position target mode.
# Pole held at a 45 degree angle using position target mode.
env0 = gym.create_env(sim, env_lower, env_upper, 2)
cubebot0 = gym.create_actor(env0, cubebot_asset, initial_pose, 'CubeBot', 0, 0)
# Configure DOF properties
props = gym.get_actor_dof_properties(env0, cubebot0)
props["driveMode"].fill(gymapi.DOF_MODE_EFFORT) 
props["stiffness"].fill(0.0)
props['damping'].fill(0.0)
props['velocity'].fill(60.0)
props['effort'].fill(0.0)
props['friction'].fill(0.001)


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
key = Keyboard()


dof_state_tensor = gym.acquire_dof_state_tensor(sim)
dof_state = gymtorch.wrap_tensor(dof_state_tensor)
print(dof_state)
dof_pos = dof_state.view(num_envs, num_dof, 2)[..., 0]
dof_vel = dof_state.view(num_envs, num_dof, 2)[..., 1]

positions =  6.0 * (torch.rand((1)) - 0.5) - np.pi
velocities = 2.0 * (torch.rand((1)) - 0.5)
dof_pos[0, :] = positions[:]
dof_vel[0, :] = velocities[:]
env_ids = torch.tensor([0])
env_ids_int32 = env_ids.to(dtype=torch.int32)
gym.set_dof_state_tensor_indexed(sim,
                                gymtorch.unwrap_tensor(dof_state),
                                gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
onnx_model = onnx.load("pendulum.onnx")
ort_model = ort.InferenceSession("pendulum.onnx")
while not gym.query_viewer_has_closed(viewer):
    gym.refresh_actor_root_state_tensor(sim)
    # print(root_states)

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    # a = joy.get_axis()
    a = key.get_keys()
    enc_sin, enc_cos, pole_pos = convert_angle(dof_pos)
    pole_vel = dof_vel/20.0
    # print('~~~~~~~~~~~~~~~~~~~')
    # print(pole_pos)
    # print(pole_vel)
    if(1):
        comm_obj.out_data = np.array([enc_sin, enc_cos, pole_vel, 0.0])
        comm_obj.write_data()
        comm_obj.read_data()
        max_push_effort = 0.10    
        action = comm_obj.in_data[0] * max_push_effort
    else:
        outputs = ort_model.run(
        None,
        {"obs": comm_obj.out_data[0:3].reshape(1,-1).astype(np.float32)},
        )
        print(outputs[0])
        action=outputs[0]* max_push_effort
    # print(comm_obj.in_data)
    # gym.apply_dof_effort(env0, joint_idx, a[0]/20.0)
    if(a[0]):
        action = 0.0
    gym.apply_dof_effort(env0, joint_idx, action)
    

 
    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)

print('Done')

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)


