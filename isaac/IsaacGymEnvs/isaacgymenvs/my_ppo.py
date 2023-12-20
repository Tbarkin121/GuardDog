from isaacgym import gymutil, gymtorch, gymapi
from isaacgym.torch_utils import *
from isaacgym.terrain_utils import *

from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_quaternion, quaternion_to_matrix, quaternion_apply, quaternion_invert

import numpy as np
import os
import torch
import yaml

# from tasks.base.vec_task_glider import VecTaskGlider
from tasks.dynasoar_env import DynasoarEnv
from tasks.rl_algs.TinyPPO import PPO_Agent
from tasks.joystick import Joystick

import pygame
import time

# from pytorch3d.transforms import quaternion_to_matrix, quaternion_apply, euler_angles_to_matrix, quaternion_invert
# from tasks.csv_logger import CSVLogger
device = "cuda:0"
torch.set_default_device(device)


class State_Check():
    def __init__(self):
        with open("cfg/task/Dynasoar2.yaml", "r") as cfg:
            try:
                self.cfg = yaml.safe_load(cfg)
            except yaml.YAMLError as exc:
                print(exc)
        # print(self.cfg)
        self.env = DynasoarEnv(self.cfg)
        self.agent = PPO_Agent(self.cfg)
        self.ctrlFreqInv = self.cfg['env']['control']['controlFrequencyInv']
        # optimization flags for pytorch JIT
        # torch._C._jit_set_profiling_mode(False)
        # torch._C._jit_set_profiling_executor(False)

        # # Look at the first env
        # cam_pos = gymapi.Vec3(2, 1, 1)
        # cam_target = gymapi.Vec3(0, 0, 0)
        # self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # self.vel_loc = torch.zeros((self.num_envs, 3))
        # self.goal_pos = torch.zeros((self.num_envs, 3))

        self.joy=Joystick()

        self.create_sim()
        self.get_state_tensors()

        
    def create_sim(self):
        self.up_axis_idx = 2 # index of up axis: Y=1, Z=2
        

        # initialize gym
        self.gym = gymapi.acquire_gym()
        # parse arguments
        args = gymutil.parse_arguments(description="Dynasoar Playground")

        self.sim_params = gymapi.SimParams()
        self.sim_params.physx.num_threads = args.num_threads
        self.sim_params.physx.use_gpu = args.use_gpu
        self.sim_params.use_gpu_pipeline = False
        self.sim_params.up_axis = gymapi.UP_AXIS_Z

        self.sim_params.physx.use_gpu = True
        self.sim_params.physx.solver_type = 1  # default = 1 (Temporal Gauss-Seidel)
        self.sim_params.physx.max_gpu_contact_pairs = 1024 ** 2  # default = 1024^2
        self.sim_params.physx.default_buffer_size_multiplier = 8  # default = 1

        self.sim = self.gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, self.sim_params)

        self._create_custom_ground_plane()
        self.num_envs=16
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))
        
        

        # create viewer using the default camera properties
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        if self.viewer is None:
            raise ValueError('*** Failed to create viewer')
        
        # Look at the first env
        cam_pos = gymapi.Vec3(2, 1, 1)
        cam_target = gymapi.Vec3(0, 0, 0)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        self.vel_loc = torch.zeros((self.num_envs, 3))
        self.goal_pos = torch.zeros((self.num_envs, 3))

        if self.sim is None:
            print("*** Failed to create sim")
            quit()

        

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
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
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
        start_pose.p = gymapi.Vec3(*self.env.base_init_state[:3])

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
            
            rand_color = torch.rand((3))
            self.gym.set_rigid_body_color(env_ptr, glider_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(rand_color[0],rand_color[1],rand_color[2]))
            self.gym.set_rigid_body_color(env_ptr, goal_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(rand_color[0],rand_color[1],rand_color[2]))

            self.envs.append(env_ptr)
            self.glider_handles.append(glider_handle)
            self.goal_handles.append(goal_handle)

        self.base_index = self.gym.find_actor_rigid_body_handle(self.envs[0], self.glider_handles[0], "base")

    def get_state_tensors(self):
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(actor_root_state).clone()
       
        self.num_actors=2
        # *2 because we are loading 2 assets
        
        self.glider_states = self.root_states.view(self.num_envs, self.num_actors, 13)[:, 0, :].detach()
        self.goal_states = self.root_states.view(self.num_envs, self.num_actors, 13)[:, 1, :]

        self.pos = self.root_states.view(self.num_envs, self.num_actors, 13)[:, 0, 0:3] #num_envs, num_actors, 13 (pos,ori,Lvel,Avel)
        self.ori = self.root_states.view(self.num_envs, self.num_actors, 13)[:, 0, 3:7] #num_envs, num_actors, 13 (pos,ori,Lvel,Avel)
        self.linvel = self.root_states.view(self.num_envs, self.num_actors, 13)[:, 0, 7:10] #num_envs, num_actors, 13 (pos,ori,Lvel,Avel)
        self.angvel = self.root_states.view(self.num_envs, self.num_actors, 13)[:, 0, 10:13] #num_envs, num_actors, 13 (pos,ori,Lvel,Avel)

        self.goal_pos = self.root_states.view(self.num_envs, self.num_actors, 13)[:, 1, 0:3]
        self.gym.refresh_actor_root_state_tensor(self.sim)


    def simulation_loop(self):
        # actions = torch.zeros((self.num_envs, self.env.num_acts))
        a = self.joy.get_axis()
        b = self.joy.get_button()
        d = self.joy.get_dpad()
    
        a[2] = a[2] - 1 # The triggers are [0,2] instead of [-1,1]
        a[5] = a[5] - 1 
        # print(a)
        # actions[self.env_2_watch, :] = torch.tensor((a[3], a[1], a[2], a[5]))
        

        # actions[:,1] = actions[:,1] * self.action_scale_tail #Scale Both The Same For AWing
        # actions[:,0] = actions[:,0] * self.action_scale_tail 
        self.agent.training_step()

        for _ in range(self.ctrlFreqInv): 
            actions, log_probs = self.agent.get_actions()
            actions[0, :] = torch.tensor((a[3], a[1], a[2], a[5]))
            new_obs, old_obs, new_rews, new_dones, new_root_states, goal_pos = self.env.step(actions)
            new_vals = self.agent.get_vals(new_obs)
            # print(self.agent.buffer.returns[0:2, 0:3])
            # print(actions[0:2])
            
 
            if(1):
                self.glider_states[:] = new_root_states[0:self.num_envs, ...]
                self.goal_pos[:] = goal_pos[0:self.num_envs, ...]
                self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
                # update the viewer
                self.env_2_watch = 0
                self.camera_follow()
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                # Wait for dt to elapse in real time.
                # This synchronizes the physics simulation with the rendering rate.
                # self.gym.sync_frame_time(self.sim)

        self.agent.buffer_store(old_obs, actions, log_probs, new_rews, new_obs, new_dones, new_vals)

        # self.gym.destroy_viewer(self.viewer)
        # self.gym.destroy_sim(self.sim)
        



    def camera_follow(self):
        num_per_row = int(np.sqrt(self.num_envs))
        spacing = 0
        row_num = int(self.env_2_watch/num_per_row)*spacing
        col_num = (self.env_2_watch % num_per_row)*spacing
        env_offset = gymapi.Vec3(col_num, row_num, 0.0)
        cam_offset = torch.tensor([[5.0], [0.0], [0.5]]).to('cpu')


        Quat = torch.unsqueeze(self.glider_states[self.env_2_watch, 3:7], 0)
        Quat = torch.unsqueeze(quaternion_invert(self.glider_states[self.env_2_watch, 3:7]), 0)
        Quat = torch.roll(Quat, 1, 1)
        # Quat = torch.tensor([1.0, 0.0, 0.0, 0.0])
        cam_offset_rot = quaternion_apply(Quat, torch.permute(cam_offset,(1,0)))
        cam_o = gymapi.Vec3(cam_offset_rot[0,0],cam_offset_rot[0,1],cam_offset_rot[0,2])

        cam_pos = gymapi.Vec3(self.glider_states[self.env_2_watch,0], self.glider_states[self.env_2_watch,1], self.glider_states[self.env_2_watch,2])
        # cam_pos = gymapi.Vec3(self.glider_states[self.env_2_watch,0], self.glider_states[self.env_2_watch,1], 5.0)

        # cam_target = gymapi.Vec3(10, self.glider_states[self.env_2_watch,1], self.glider_states[self.env_2_watch,2])
        cam_target = gymapi.Vec3(self.glider_states[self.env_2_watch,0], self.glider_states[self.env_2_watch,1], self.glider_states[self.env_2_watch,2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos+cam_o, cam_target)
        # self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

#%%
sc = State_Check()
#%%
import time
for it  in range(1000):
    time_start = time.perf_counter()
    sc.simulation_loop()
    time_end = time.perf_counter()
    print('iter:{}. fps : {}'.format(it,sc.cfg['env']['horizon']*sc.cfg['env']['num_envs']/(time_end-time_start)))
    # print(sc.agent.buffer.d[0,:])
    # print(sc.agent.buffer.s1[0,:,:])



