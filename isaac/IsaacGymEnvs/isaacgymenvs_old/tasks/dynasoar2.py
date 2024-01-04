# Copyright (c) 2018-2022, NVIDIA Corporation
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


from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *
from isaacgym.terrain_utils import *
# from tasks.base.vec_task import VecTask
from tasks.base.vec_task_glider import VecTaskGlider

import numpy as np
import os
import torch

from typing import Tuple, Dict
from tasks.dynasoar_env import DynasoarEnv
from .joystick import Joystick
import time



class Dynasoar2(VecTaskGlider):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        
        self.debug_flags = self.cfg["env"]["debug_flags"]
        
        # randomization
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.randomize = self.cfg["task"]["randomize"]

        # plane params
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]


        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # other
        self.dt = self.sim_params.dt
        self.env = DynasoarEnv(cfg)



        if self.viewer != None:
            p = self.cfg["env"]["viewer"]["pos"]
            lookat = self.cfg["env"]["viewer"]["lookat"]
            cam_pos = gymapi.Vec3(p[0], p[1], p[2])
            cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)

        self.gym.refresh_actor_root_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        # *2 because we are loading 2 assets
        self.glider_states = self.root_states.view(self.num_envs*2, 13)[0::2, :]
        self.goal_states = self.root_states.view(self.num_envs*2, 13)[1::2, :]
        
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)

        self.env = DynasoarEnv(self.cfg)


        # self.reset_goals(torch.arange(self.num_envs, device=self.device))
        if(self.debug_flags['joystick']):
            self.joy = Joystick()

        self.camera_setup()
        self.input_delay = 10
        self.input_delay_counter = 0


    def create_sim(self):
        self.up_axis_idx = 2 # index of up axis: Y=1, Z=2
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        # self._create_ground_plane()
        self._create_custom_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))
        # If randomizing, apply once immediately on startup before the fist sim step
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        self.gym.add_ground(self.sim, plane_params)
        
    def _create_custom_ground_plane(self):
        terrain_width = 2000.
        terrain_length = 2000.
        horizontal_scale = 3.0  # [m]
        vertical_scale = 1.0  # [m]
        num_rows = int(terrain_width/horizontal_scale)
        num_cols = int(terrain_length/horizontal_scale)
        sub_terrain = SubTerrain(width=num_rows, length=num_cols, vertical_scale=vertical_scale, horizontal_scale=horizontal_scale)
        heightfield = discrete_obstacles_terrain(sub_terrain, max_height=vertical_scale, min_size=10., max_size=100., num_rects=10000).height_field_raw
        
        # add the terrain as a triangle mesh
        vertices, triangles = convert_heightfield_to_trimesh(heightfield, horizontal_scale=horizontal_scale, vertical_scale=vertical_scale, slope_threshold=1.5)
        vertices = vertices - vertical_scale
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = vertices.shape[0]
        tm_params.nb_triangles = triangles.shape[0]
        tm_params.transform.p.x = -terrain_width/2.0
        tm_params.transform.p.y = -terrain_width/2.0
        self.gym.add_triangle_mesh(self.sim, vertices.flatten(), triangles.flatten(), tm_params)
    
    def _create_envs(self, num_envs, spacing, num_per_row):
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../assets')
        glider_asset_file = self.cfg['env']['glider']['urdf_path']
        goal_asset_file = self.cfg['env']['goal']['urdf_path']
        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.collapse_fixed_joints = True
        asset_options.replace_cylinder_with_capsule = True
        asset_options.density = 0.001
        asset_options.angular_damping = 0.0
        asset_options.linear_damping = 0.0
        asset_options.armature = 0.0
        asset_options.thickness = 0.01
        asset_options.disable_gravity = False

        glider_asset = self.gym.load_asset(self.sim, asset_root, glider_asset_file, asset_options)
        goal_asset = self.gym.load_asset(self.sim, asset_root, goal_asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(glider_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(glider_asset)

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        body_names = self.gym.get_asset_rigid_body_names(glider_asset)
        self.dof_names = self.gym.get_asset_dof_names(glider_asset)
        self.base_index = 0

        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        self.glider_handles = []
        self.goal_handles = []
        self.envs = []

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, env_lower, env_upper, num_per_row)
            glider_handle = self.gym.create_actor(env_ptr, glider_asset, start_pose, "glider", i, 1, 0)
            goal_handle = self.gym.create_actor(env_ptr, goal_asset, start_pose, "goal", i, 1, 0)
            
            rand_color = torch.rand((3), device=self.device)
            self.gym.set_rigid_body_color(env_ptr, glider_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(rand_color[0],rand_color[1],rand_color[2]))
            self.gym.set_rigid_body_color(env_ptr, goal_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(rand_color[0],rand_color[1],rand_color[2]))

            self.envs.append(env_ptr)
            self.glider_handles.append(glider_handle)
            self.goal_handles.append(goal_handle)

        self.base_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.glider_handles[0], "base")

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)


    def physics_step(self): 
        self.gym.refresh_actor_root_state_tensor(self.sim)
        # print('!!')
        # print(self.root_states)
        # print(self.glider_states)
        # print(self.goal_states)
        actions = self.actions

        if(self.debug_flags['joystick']):
            a = self.joy.get_axis()
            b = self.joy.get_button()
            d = self.joy.get_dpad()
            self.input_delay_counter += 1
            if(self.debug_flags['control_en']):
                a[2] = a[2] - 1 # The triggers are [0,2] instead of [-1,1]
                a[5] = a[5] - 1 
                # print(a)
                # actions[self.env_2_watch, :] = torch.tensor((a[3], a[1], a[2], a[5]), device=self.device)
                actions[self.env_2_watch, :] = torch.tensor((a[4], a[1], a[2], a[5]), device=self.device)
            

            if(b[0] and self.input_delay_counter > self.input_delay):
                if(self.debug_flags['cam_follow']):
                    self.debug_flags['cam_follow'] = False
                else :
                    self.debug_flags['cam_follow'] = True
                self.input_delay_counter = 0

            if(b[5] and self.input_delay_counter > self.input_delay):
                self.debug_flags['control_en'] = ~self.debug_flags['control_en']
                self.input_delay_counter = 0

        
        new_root_states, new_obs, new_acts, new_rews, goal_pos = self.env.step(actions)
        self.glider_states[:] = new_root_states
        self.goal_states[:,0:3] = goal_pos


        if(self.debug_flags['plots_enable']):
            pass
            # self.draw_all_line_graphs()

        if(self.debug_flags['cam_follow']):
            if(self.debug_flags['joystick']):
                self.camera_follow(d)
            else:
                self.camera_follow([0,0])

        if(self.debug_flags['print']):
            #Contain printouts to debug function calls so they aren't scattered all over the place
            self.physics.debug()

        # self.print_all_tensors()
        # print(self.obs_buf[0],...)

    def post_physics_step(self):
        self.progress_buf += 1

        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
        self.compute_observations()

        if(self.debug_flags['control_en']):
            self.obs_buf[self.env_2_watch, ]
            pass

        self.compute_reward()


    def compute_reward(self):
        self.rew_buf[:], self.reset_buf[:], self.goal_reached[:] = self.env.get_reward()


    def compute_observations(self):
        self.obs_buf[:] = self.env.get_observation()


    def camera_setup(self):
        self.env_2_watch = self.debug_flags['cam_target']

    def camera_follow(self, d_pad):
        if(self.input_delay_counter > self.input_delay):
            if(d_pad[1] == 1):
                self.env_2_watch = int(torch.randint(0, self.num_envs, (1,)))
                self.input_delay_counter = 0
                self.debug_flags['control_en'] = False
            if(d_pad[1] == -1):
                self.env_2_watch = 0
                self.input_delay_counter = 0
                self.debug_flags['control_en'] = False
            if(d_pad[0] == 1):
                self.env_2_watch += 1
                self.input_delay_counter = 0
                self.debug_flags['control_en'] = False
            if(d_pad[0] == -1):
                self.env_2_watch -= 1
                self.input_delay_counter = 0
                self.debug_flags['control_en'] = False
            
        # Wrap Around
        if(self.env_2_watch > self.num_envs-1):
            self.env_2_watch = 0
        if(self.env_2_watch < 0):
            self.env_2_watch = int(self.num_envs-1)

        
        num_per_row = int(np.sqrt(self.num_envs))
        spacing = 2
        row_num = int(self.env_2_watch/num_per_row)*spacing
        col_num = (self.env_2_watch % num_per_row)*spacing        
        env_offset = gymapi.Vec3(col_num, row_num, 0.0)
        cam_offset = torch.tensor([[5.0], [0.0], [0.5]], device=self.device)


        Quat = torch.unsqueeze(self.glider_states[self.env_2_watch, 3:7], 0)
        Quat = torch.unsqueeze(quaternion_invert(self.glider_states[self.env_2_watch, 3:7]), 0)
        Quat = torch.roll(Quat, 1, 1)
        # Quat = torch.tensor([1.0, 0.0, 0.0, 0.0])

        cam_offset_rot = quaternion_apply(Quat, torch.permute(cam_offset,(1,0)))
        cam_o = gymapi.Vec3(cam_offset_rot[0,0],cam_offset_rot[0,1],cam_offset_rot[0,2])

        # cam_pos = gymapi.Vec3(self.glider_states[self.env_2_watch,0], self.glider_states[self.env_2_watch,1], self.glider_states[self.env_2_watch,2])
        cam_pos = gymapi.Vec3(self.glider_states[self.env_2_watch,0], self.glider_states[self.env_2_watch,1], 5.0)

        # cam_target = gymapi.Vec3(10, self.glider_states[self.env_2_watch,1], self.glider_states[self.env_2_watch,2])
        cam_target = gymapi.Vec3(self.glider_states[self.env_2_watch,0], self.glider_states[self.env_2_watch,1], self.glider_states[self.env_2_watch,2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos+cam_o+env_offset, cam_target+env_offset)
        # self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)