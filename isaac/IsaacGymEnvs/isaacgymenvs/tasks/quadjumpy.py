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

class QuadJumpy(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        self.reset_dist = self.cfg["env"]["resetDist"]

        self.max_push_effort = self.cfg["env"]["maxEffort"]
        self.max_episode_length = self.cfg["env"]["maxEpisodeLen"]

        self.cfg["env"]["numObservations"] = 49
        self.cfg["env"]["numActions"] = 12

        # randomization
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.randomize = self.cfg["task"]["randomize"]


        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

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
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.initial_root_states = self.root_states.clone()
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)  # shape: num_envs, num_bodies, xyz axis
        self.torques = gymtorch.wrap_tensor(torques).view(self.num_envs, self.num_dof)


        self.keys = Keyboard(3)
        self.max_height_reached = torch.zeros((self.num_envs), device=self.device)
        self.air_time = torch.zeros((self.num_envs), device=self.device)

        cam_pos = gymapi.Vec3(10.0, 9.95, 0.5)
        cam_target = gymapi.Vec3(10.0, -20.0, 0.5)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        

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
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        # define plane on which environments are initialized
        lower = gymapi.Vec3(-spacing, -spacing, 0.0) if self.up_axis == 'z' else gymapi.Vec3(0.5 * -spacing, 0.0, -spacing)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        asset_file = "urdf/Quad_Foot/urdf/Quad_Foot.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", asset_root))
            asset_file = self.cfg["env"]["asset"].get("assetFileName", asset_file)

        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.angular_damping = 0.0
        asset_options.max_angular_velocity = 10000
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT

        torquepole_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(torquepole_asset)
        self.body_names = self.gym.get_asset_rigid_body_names(torquepole_asset)
        self.dof_names = self.gym.get_asset_dof_names(torquepole_asset)

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
        if self.up_axis == 'z':
            pose.p.z = 1.0
            # asset is rotated z-up by default, no additional rotations needed
            # pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
            pose.r = gymapi.Quat.from_euler_zyx(1.5708, 0.0, 0.0)
        else:
            pose.p.y = 0.0
            pose.r = gymapi.Quat(-np.sqrt(2)/2, 0.0, 0.0, np.sqrt(2)/2)

        self.torquepole_handles = []
        self.envs = []
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            torquepole_handle = self.gym.create_actor(env_ptr, torquepole_asset, pose, "torquepole", i, 1, 0)
            
            rand_color = torch.rand((3), device=self.device)
            self.gym.set_rigid_body_color(env_ptr, torquepole_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(rand_color[0],rand_color[1],rand_color[2]))
            rand_color = torch.rand((3), device=self.device)
            self.gym.set_rigid_body_color(env_ptr, torquepole_handle, 1, gymapi.MESH_VISUAL, gymapi.Vec3(rand_color[0],rand_color[1],rand_color[2]))
            
            dof_props = self.gym.get_actor_dof_properties(env_ptr, torquepole_handle)
            dof_props['driveMode'][:] = gymapi.DOF_MODE_EFFORT
            dof_props['stiffness'][:] = 0.0
            dof_props['damping'][:] = 0.0
            dof_props['velocity'].fill(25.0)
            dof_props['effort'].fill(0.0)
            dof_props['friction'].fill(0.01)

            self.gym.set_actor_dof_properties(env_ptr, torquepole_handle, dof_props)

            self.envs.append(env_ptr)
            self.torquepole_handles.append(torquepole_handle)
        
        for i in range(len(hip_names)):
            self.hip_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.torquepole_handles[0], hip_names[i])
        for i in range(len(thigh_names)):
            self.thigh_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.torquepole_handles[0], thigh_names[i])
        for i in range(len(shin_names)):
            self.shin_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.torquepole_handles[0], shin_names[i])
        for i in range(len(foot_names)):
            self.foot_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.torquepole_handles[0], foot_names[i])
        print(self.hip_indices)
        print(self.thigh_indices)
        print(self.shin_indices)
        print(self.foot_indices)



    def compute_reward(self):
        # retrieve environment observations from buffer
        height = self.obs_buf[:, 4*self.num_dof]        
        self.max_height_reached = torch.max(self.max_height_reached, height)
        self.max_height_reached = torch.where((torch.any(torch.norm(self.contact_forces[:, self.foot_indices, :], dim=1) > 0.1, dim=1)), torch.zeros_like(self.max_height_reached), self.max_height_reached)

        self.air_time += 1
        self.air_time = torch.where((torch.any(torch.norm(self.contact_forces[:, self.foot_indices, :], dim=1) > 0.1, dim=1)), torch.zeros_like(self.air_time), self.air_time)

        # print(self.max_height_reached)

        # print(self.contact_forces.shape)
        # print(self.contact_forces[0, :, :])

        # print(torch.norm(self.contact_forces[0:3, 0, :], dim=1) > 1.)

        # print(torch.any(torch.norm(self.contact_forces[:, self.hip_indices, :], dim=1) > 1., dim=1).shape)
        
        self.rew_buf[:], self.reset_buf[:] = compute_torquepole_reward(height,
                                                                        self.max_height_reached,
                                                                        self.air_time,
                                                                        self.contact_forces,
                                                                        self.hip_indices,
                                                                        self.thigh_indices,
                                                                        self.shin_indices,
                                                                        self.reset_dist, 
                                                                        self.reset_buf, 
                                                                        self.progress_buf, 
                                                                        self.max_episode_length)
        # print(self.rew_buf[0])
        # print(self.reset_buf)

    def convert_angle(self, angle):
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
        
    def compute_observations(self, env_ids=None):
        if env_ids is None:
            env_ids = np.arange(self.num_envs)

        self.gym.refresh_dof_state_tensor(self.sim)  # done in step
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        
        sin_encode, cos_encode, motor_angle = self.convert_angle(self.dof_pos[env_ids, 0:self.num_dof].squeeze())
        self.obs_buf[env_ids, 0:self.num_dof] = sin_encode
        self.obs_buf[env_ids, self.num_dof:2*self.num_dof] = cos_encode
        self.obs_buf[env_ids, 2*self.num_dof:3*self.num_dof] = self.dof_vel[env_ids, :]/20.0            # Motor 0, Velocity


        self.obs_buf[env_ids, 3*self.num_dof:4*self.num_dof] = self.actions_tensor[env_ids, :]          # Actions
        self.obs_buf[env_ids, 4*self.num_dof] = self.root_states[env_ids, 2]                                # Height

        # print('!!!')
        # print(self.obs_buf[0,...])
        # print(self.obs_buf[0, 2*self.num_dof:3*self.num_dof])
        # print(self.obs_buf[0, 3*self.num_dof:4*self.num_dof])
        return self.obs_buf

    def reset_idx(self, env_ids):
        # Randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)
        positions =  torch.ones((len(env_ids), self.num_dof), device=self.device)*0.0

        positions[:, 1] = 0.5
        positions[:, 4] = -0.5
        positions[:, 7] = -0.5
        positions[:, 10] = 0.5
        positions[:, 2] = -2.0
        positions[:, 5] = 2.0
        positions[:, 8] = 2.0
        positions[:, 11] = -2.0
        velocities = torch.zeros((len(env_ids), self.num_dof), device=self.device)

        self.dof_pos[env_ids, :] = positions[:]
        self.dof_vel[env_ids, :] = velocities[:]

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.initial_root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        
        
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        self.max_height_reached[env_ids] = 0
        self.air_time[env_ids] = 0

    def pre_physics_step(self, actions):
        self.actions_tensor = torch.zeros( [self.num_envs, self.num_dof], device=self.device, dtype=torch.float)
        self.actions_tensor[:, 0:self.num_dof] = actions.to(self.device) * self.max_push_effort

        a = self.keys.get_keys()
        scale = torch.tensor([10, self.max_push_effort, self.max_push_effort])
        self.actions_tensor[0,0:3] = a*scale

        forces = gymtorch.unwrap_tensor(self.actions_tensor)
        self.gym.set_dof_actuation_force_tensor(self.sim, forces)

        # print(actions_tensor[0])

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward()

#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_torquepole_reward(height, max_height_reached, air_time, contact_forces, hip_idx, thigh_idx, shin_idx, reset_dist, reset_buf, progress_buf, max_episode_length):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor, float) -> Tuple[Tensor, Tensor]

    # reward = height**3
    # reward = max_height_reached**2
    reward = height*0.1 + max_height_reached*0.05
    reward += air_time/100.0
    # reward = torch.where((torch.norm(contact_forces[:, 3, :], dim=1) > 0.1), torch.zeros_like(reward), reward)

    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)

    # This is a hacky fix, the contact forces sometimes don't update when an environment resets causing a double reset. 
    # This waits 10 environment steps before factoring in contact forces
    check_forces = torch.where(progress_buf >= 10, torch.ones_like(reset_buf), reset_buf)

    reset = reset | ((torch.norm(contact_forces[:, 0, :], dim=1) > 1.) & check_forces)
    reset = reset | ((torch.any(torch.norm(contact_forces[:, hip_idx, :], dim=2) > 1., dim=1)) & check_forces)
    reset = reset | ((torch.any(torch.norm(contact_forces[:, thigh_idx, :], dim=2) > 1., dim=1)) & check_forces)
    # reset = reset | (torch.any(torch.norm(contact_forces[:, shin_idx, :], dim=2) > 1., dim=1))

    
    
    
    return reward, reset
