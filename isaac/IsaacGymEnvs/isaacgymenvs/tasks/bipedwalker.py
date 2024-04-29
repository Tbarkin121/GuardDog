# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import os
import torch

from isaacgym import gymutil, gymtorch, gymapi
from .base.vec_task import VecTask
from .keyboard import Keyboard

from isaacgymenvs.utils.torch_jit_utils import to_torch, get_axis_params, torch_rand_float, quat_rotate, quat_rotate_inverse

class BipedWalker(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg
        # normalization
        self.lin_vel_scale = self.cfg["env"]["learn"]["linearVelocityScale"]
        self.ang_vel_scale = self.cfg["env"]["learn"]["angularVelocityScale"]
        self.dof_vel_scale = 1/self.cfg["env"]["control"]["maxVelocity"]
        self.max_dof_effort = self.cfg["env"]['control']["maxEffort"]
        self.max_dof_velocity = self.cfg["env"]['control']["maxVelocity"]
        self.dof_friction = self.cfg["env"]['control']["friction"]
        
        # reward scales
        self.rew_scales = {}
        self.rew_scales["lin_vel_xy"] = self.cfg["env"]["learn"]["linearVelocityXYRewardScale"]
        self.rew_scales["ang_vel_z"] = self.cfg["env"]["learn"]["angularVelocityZRewardScale"]
        self.rew_scales["torque"] = self.cfg["env"]["learn"]["torqueRewardScale"]
        self.rew_scales["toe_force"] = self.cfg["env"]["learn"]["toeForceRewardScale"]
        self.rew_scales["joints_speed"] = self.cfg["env"]["learn"]["jointSpeedRewardScale"]
        self.reset_dist = self.cfg["env"]["resetDist"]
        
        # randomization
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.randomize = self.cfg["task"]["randomize"]

        # command ranges
        self.command_x_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_x"]
        self.command_y_range = self.cfg["env"]["randomCommandVelocityRanges"]["linear_y"]
        self.command_yaw_range = self.cfg["env"]["randomCommandVelocityRanges"]["yaw"]

        # plane params
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        # base init state
        pos = self.cfg["env"]["baseInitState"]["pos"]
        rot = self.cfg["env"]["baseInitState"]["rot"]
        v_lin = self.cfg["env"]["baseInitState"]["vLinear"]
        v_ang = self.cfg["env"]["baseInitState"]["vAngular"]
        state = pos + rot + v_lin + v_ang

        self.base_init_state = state

        # default joint positions
        self.named_default_joint_angles = self.cfg["env"]["defaultJointAngles"]

        self.cfg["env"]["numObservations"] = 40
        self.cfg["env"]["numActions"] = 8

        self.Kp = self.cfg["env"]["control"]["stiffness"]
        self.Kd = self.cfg["env"]["control"]["damping"]

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # other
        self.dt = self.sim_params.dt
        self.max_episode_length_s = self.cfg["env"]["learn"]["episodeLength_s"]
        self.max_episode_length = int(self.max_episode_length_s / self.dt + 0.5)

        

        for key in self.rew_scales.keys():
            self.rew_scales[key] *= self.dt
        
        if self.viewer != None:
            p = self.cfg["env"]["viewer"]["pos"]
            lookat = self.cfg["env"]["viewer"]["lookat"]
            cam_pos = gymapi.Vec3(p[0], p[1], p[2])
            cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

            # cam_pos = gymapi.Vec3(10.0, 9.95, 0.5)
            # cam_target = gymapi.Vec3(10.0, -20.0, 0.5)
            # self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        
        # get gym state tensors
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        torques = self.gym.acquire_dof_force_tensor(self.sim)
        
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)  # shape: num_envs, num_bodies, xyz axis
        self.torques = gymtorch.wrap_tensor(torques).view(self.num_envs, self.num_dof)

        self.up_axis_idx = 1 # index of up axis: Y=1, Z=2
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))

        self.commands = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.commands_x = self.commands.view(self.num_envs, 3)[..., 0]
        self.commands_y = self.commands.view(self.num_envs, 3)[..., 1]
        self.commands_yaw = self.commands.view(self.num_envs, 3)[..., 2]
        self.default_dof_pos = torch.zeros_like(self.dof_pos, dtype=torch.float, device=self.device, requires_grad=False)

        for i in range(self.cfg["env"]["numActions"]):
            name = self.dof_names[i]
            angle = self.named_default_joint_angles[name]
            self.default_dof_pos[:, i] = angle


            

        self.initial_root_states = self.root_states.clone()
        self.initial_root_states[:] = to_torch(self.base_init_state, device=self.device, requires_grad=False)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.keys = Keyboard(3)

        self.reset_idx(torch.arange(self.num_envs, device=self.device))


    def create_sim(self):
        # set the up axis to be z-up given that assets are y-up by default
        self.up_axis = self.cfg["sim"]["up_axis"]

        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        # set the normal force to be z dimension
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0) if self.up_axis == 'z' else gymapi.Vec3(0.0, 1.0, 0.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        # define plane on which environments are initialized
        lower = gymapi.Vec3(-spacing, -spacing, 0.0) if self.up_axis == 'z' else gymapi.Vec3(0.5 * -spacing, 0.0, -spacing)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        asset_file = "urdf/Biped/urdf/Biped.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
            asset_file = self.cfg["env"]["asset"].get("assetFileName", asset_file)

        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        asset_options.collapse_fixed_joints = True
        asset_options.replace_cylinder_with_capsule = True
        asset_options.fix_base_link = False
        asset_options.angular_damping = 0.0
        asset_options.linear_damping = 0.0
        asset_options.max_angular_velocity = 10000
        asset_options.armature = 0.0
        asset_options.thickness = 0.01
        asset_options.disable_gravity = False


        bipedwalker_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(bipedwalker_asset)
        self.body_names = self.gym.get_asset_rigid_body_names(bipedwalker_asset)
        self.dof_names = self.gym.get_asset_dof_names(bipedwalker_asset)

        print('self.num_dof')
        print(self.num_dof)
        print('self.body_names')
        print(self.body_names)
        print('self.dof_names')
        print(self.dof_names)

        hip_names = [s for s in self.body_names if "Hip" in s]
        thigh_names = [s for s in self.body_names if "Thigh" in s]
        shin_names = [s for s in self.body_names if "Shin" in s]
        foot_names = [s for s in self.body_names if "Foot" in s]

        self.hip_indices = torch.zeros(len(hip_names), dtype=torch.long, device=self.device, requires_grad=False)
        self.thigh_indices = torch.zeros(len(thigh_names), dtype=torch.long, device=self.device, requires_grad=False)
        self.shin_indices = torch.zeros(len(shin_names), dtype=torch.long, device=self.device, requires_grad=False)
        self.foot_indices = torch.zeros(len(foot_names), dtype=torch.long, device=self.device, requires_grad=False)

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(*self.base_init_state[:3])
        

        self.bipedwalker_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            bipedwalker_handle = self.gym.create_actor(env_ptr, bipedwalker_asset, pose, "bipedwalker", i, 1, 0)
            
            rand_color = torch.rand((3), device=self.device)
            self.gym.set_rigid_body_color(env_ptr, bipedwalker_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(rand_color[0],rand_color[1],rand_color[2]))
            rand_color = torch.rand((3), device=self.device)
            self.gym.set_rigid_body_color(env_ptr, bipedwalker_handle, 1, gymapi.MESH_VISUAL, gymapi.Vec3(rand_color[0],rand_color[1],rand_color[2]))
            
            dof_props = self.gym.get_actor_dof_properties(env_ptr, bipedwalker_handle)
            dof_props['driveMode'][:] = gymapi.DOF_MODE_EFFORT
            dof_props['stiffness'][:] = self.Kp
            dof_props['damping'][:] = self.Kd
            dof_props['velocity'][:] = self.max_dof_velocity
            dof_props['effort'].fill(0.0)
            dof_props['friction'][:] = self.dof_friction

            dof_props['velocity'][6:8] = 200.0
            dof_props['friction'][6:8] = 0.001

            self.gym.set_actor_dof_properties(env_ptr, bipedwalker_handle, dof_props)
            self.gym.enable_actor_dof_force_sensors(env_ptr, bipedwalker_handle)

            self.envs.append(env_ptr)
            self.bipedwalker_handles.append(bipedwalker_handle)
        
        for i in range(len(hip_names)):
            self.hip_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.bipedwalker_handles[0], hip_names[i])
        for i in range(len(thigh_names)):
            self.thigh_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.bipedwalker_handles[0], thigh_names[i])
        for i in range(len(shin_names)):
            self.shin_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.bipedwalker_handles[0], shin_names[i])
        for i in range(len(foot_names)):
            self.foot_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.bipedwalker_handles[0], foot_names[i])
        print(self.hip_indices)
        print(self.thigh_indices)
        print(self.shin_indices)
        print(self.foot_indices)



    def compute_reward(self):
        base_quat = self.root_states[:, 3:7]
        base_lin_vel = quat_rotate_inverse(base_quat, self.root_states[:, 7:10])
        base_ang_vel = quat_rotate_inverse(base_quat, self.root_states[:, 10:13])

        # velocity tracking reward
        # lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - base_lin_vel[:, :2]), dim=1)
        # ang_vel_error = torch.square(self.commands[:, 2] - base_ang_vel[:, 2])
        # print(base_lin_vel[0, :2])
        # print('!!!')
        # print(self.commands[0, :2])
        # print(base_lin_vel[0, [0,2]])

        # print(self.progress_buf)
        self.rew_buf[:], self.reset_buf[:] = compute_bipedwalker_reward(self.root_states,
                                                                        self.commands,
                                                                        self.torques,
                                                                        self.dof_vel,
                                                                        self.contact_forces,
                                                                        self.reset_buf, 
                                                                        self.progress_buf, 
                                                                        self.hip_indices,
                                                                        self.thigh_indices,
                                                                        self.shin_indices,
                                                                        self.rew_scales,
                                                                        self.reset_dist,
                                                                        self.max_episode_length)
        # print(self.rew_buf[0])
        # print(self.reset_buf)
        # print(torch.norm(torch.norm(self.contact_forces[:,[3,6]],dim=1),dim=1))
        toe_force = torch.norm(torch.norm(self.contact_forces[:,[3,6]],dim=1),dim=1)
        rew_toe_force = torch.where(toe_force>1.0, toe_force, torch.zeros_like(toe_force))
        # print(toe_force)
        # print(rew_toe_force)
    
        
    def compute_observations(self, env_ids=None):
        if env_ids is None:
            env_ids = np.arange(self.num_envs)

        self.gym.refresh_dof_state_tensor(self.sim)  # done in step
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        self.obs_buf[:] = compute_bipedwalker_observations(  # tensors
                                                        self.root_states,
                                                        self.commands,
                                                        self.dof_pos,
                                                        self.default_dof_pos,
                                                        self.dof_vel,
                                                        self.gravity_vec,
                                                        self.actions_tensor,
                                                        # scales
                                                        self.lin_vel_scale,
                                                        self.ang_vel_scale,
                                                        self.dof_vel_scale)
        # print(self.obs_buf[0,0:6])
        # print(self.obs_buf[0, 26:29])
        

    # obs = torch.cat((
    #                  base_lin_vel * lin_vel_scale, #3 (0:3)
    #                  base_ang_vel * ang_vel_scale, #3 (3:6)
    #                  sin_encode[:, 0:6], #6 (6:12)
    #                  cos_encode[:, 0:6], #6 (12:18)
    #                  dof_vel * dof_vel_scale, #8 (18:26)
    #                  projected_gravity, #3 (26:29)
    #                  commands_scaled, #3 (29:32)
    #                  actions #8 (32:40)
    #                  ), dim=-1)


        return self.obs_buf
    

    def reset_idx(self, env_ids):
        # Randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)
        
        positions_offset = torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        # positions_offset = torch.ones((len(env_ids), self.num_dof), device=self.device)
        # velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)
        velocities = torch.zeros((len(env_ids), self.num_dof), device=self.device)

        self.dof_pos[env_ids] = self.default_dof_pos[env_ids] * positions_offset[:]
        self.dof_vel[env_ids, :] = velocities[:]

        env_ids_int32 = env_ids.to(dtype=torch.int32)        
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                    gymtorch.unwrap_tensor(self.initial_root_states),
                                                    gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                            gymtorch.unwrap_tensor(self.dof_state),
                                            gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        
        self.reset_commands(env_ids)

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def reset_commands(self, env_ids):
        self.commands_x[env_ids] = torch_rand_float(self.command_x_range[0], self.command_x_range[1], (len(env_ids), 1), device=self.device).squeeze()
        self.commands_y[env_ids] = torch_rand_float(self.command_y_range[0], self.command_y_range[1], (len(env_ids), 1), device=self.device).squeeze()
        self.commands_yaw[env_ids] = torch_rand_float(self.command_yaw_range[0], self.command_yaw_range[1], (len(env_ids), 1), device=self.device).squeeze()
        




    def pre_physics_step(self, actions):
        self.actions_tensor = torch.zeros( [self.num_envs, self.num_dof], device=self.device, dtype=torch.float)
        self.actions_tensor[:, 0:self.num_dof] = actions.to(self.device) * self.max_dof_effort

        # a = self.keys.get_keys()
        # scale = torch.tensor([10, self.max_dof_effort, self.max_dof_effort])
        # self.actions_tensor[0,0:3] = a*scale

        forces = gymtorch.unwrap_tensor(self.actions_tensor)
        self.gym.set_dof_actuation_force_tensor(self.sim, forces)

        # print(actions_tensor[0])

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        env_ids = torch.where(self.progress_buf % 100 == 0,  torch.ones_like(self.progress_buf),  torch.zeros_like(self.progress_buf)).nonzero(as_tuple=False).squeeze(-1)
       
        if len(env_ids) > 0:
            self.reset_commands(env_ids)

        self.compute_observations()
        a = self.keys.get_keys()
        scale = torch.tensor([5., 1., 0.5])
        self.obs_buf[0, 29:32] = a*scale
        # print(self.obs_buf[0,29:32])
        self.compute_reward()

#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def convert_angle(angle):
        # Apply sine and cosine functions
        sin_component = torch.sin(angle)
        cos_component = torch.cos(angle)

        #  Normalize angle to [-pi, pi]
        normalized_angle = torch.remainder(angle + np.pi, 2 * np.pi) - np.pi
        # Apply offset
        # normalized_angle += np.pi
        # Normalize again if needed
        # normalized_angle = torch.remainder(normalized_angle + np.pi, 2 * np.pi) - np.pi

        #  Normalize angle to [-1, 1]
        normalized_angle /= torch.pi

        return sin_component, cos_component, normalized_angle


@torch.jit.script
def compute_bipedwalker_reward(
                            # tensors
                            root_states, 
                            commands,
                            torques,  
                            dof_vel,
                            contact_forces, 
                            reset_buf, 
                            progress_buf, 
                            hip_idx, 
                            thigh_idx, 
                            shin_idx, 
                            # Dict
                            rew_scales,
                            # other
                            reset_dist,
                            max_episode_length):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Dict[str, float], float, float) -> Tuple[Tensor, Tensor]

    # prepare quantities (TODO: return from obs ?)
    height = root_states[:,2]
    base_quat = root_states[:, 3:7]
    base_lin_vel = quat_rotate_inverse(base_quat, root_states[:, 7:10])
    base_ang_vel = quat_rotate_inverse(base_quat, root_states[:, 10:13])

    # velocity tracking reward
    lin_vel_error = torch.sum(torch.square(commands[:, :2] - base_lin_vel[:, :2]), dim=1)
    ang_vel_error = torch.square(commands[:, 2] - base_ang_vel[:, 2])
    rew_lin_vel_xy = torch.exp(-lin_vel_error/0.25) * rew_scales["lin_vel_xy"]
    rew_ang_vel_z = torch.exp(-ang_vel_error/0.25) * rew_scales["ang_vel_z"]


    # torque penalty
    rew_torque = torch.sum(torch.square(torques), dim=1)  * rew_scales["torque"]
    # joint speed penalty
    rew_joint_speed = torch.sum(torch.square(dof_vel[:, 0:6]), dim=1)  * rew_scales["joints_speed"]
    
    # stubbed toes penalty 
    toe_force = torch.norm(torch.norm(contact_forces[:,[3,6]],dim=1),dim=1)
    rew_toe_force = torch.where(toe_force>5.0, toe_force * rew_scales["toe_force"], torch.zeros_like(toe_force))
                                


    total_reward = rew_lin_vel_xy + rew_ang_vel_z + rew_torque + rew_toe_force + rew_joint_speed
    total_reward = torch.clip(total_reward, 0., None)

    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)

    # This is a hacky fix, the contact forces sometimes don't update when an environment resets causing a double reset. 
    # This waits 10 environment steps before factoring in contact forces
    check_forces = torch.where(progress_buf >= 10, torch.ones_like(reset_buf), reset_buf)

    reset = reset | ((torch.norm(contact_forces[:, 0, :], dim=1) > 1.) & check_forces) # Body Collision 
    reset = reset | ((torch.any(torch.norm(contact_forces[:, hip_idx, :], dim=2) > 1., dim=1)) & check_forces)
    reset = reset | ((torch.any(torch.norm(contact_forces[:, thigh_idx, :], dim=2) > 1., dim=1)) & check_forces)
    reset = reset | (height<0.25)
    # reset = reset | (torch.any(torch.norm(contact_forces[:, shin_idx, :], dim=2) > 1., dim=1))
    
    return total_reward.detach(), reset


@torch.jit.script
def compute_bipedwalker_observations(root_states,
                                commands,
                                dof_pos,
                                default_dof_pos,
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
    

    commands_scaled = commands*torch.tensor([lin_vel_scale, lin_vel_scale, ang_vel_scale], requires_grad=False, device=commands.device)


    sin_encode, cos_encode, motor_angle = convert_angle(dof_pos.squeeze())

    obs = torch.cat((
                     base_lin_vel * lin_vel_scale, #3 (0:3)
                     base_ang_vel * ang_vel_scale, #3 (3:6)
                     sin_encode[:, 0:6], #6 (6:12)
                     cos_encode[:, 0:6], #6 (12:18)
                     dof_vel * dof_vel_scale, #8 (18:26)
                     projected_gravity, #3 (26:29)
                     commands_scaled, #3 (29:32)
                     actions #8 (32:40)
                     ), dim=-1)

    return obs

