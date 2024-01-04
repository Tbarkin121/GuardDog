from isaacgym import gymutil, gymtorch, gymapi
from isaacgym.torch_utils import *
from isaacgym.terrain_utils import *

from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_quaternion, quaternion_to_matrix, quaternion_apply, quaternion_invert

import numpy as np
import os
import torch
import yaml

# from tasks.base.vec_task_glider import VecTaskGlider
from tasks.glider_physics import LiftingLine
from tasks.joystick import Joystick

import pygame
import time

# from pytorch3d.transforms import quaternion_to_matrix, quaternion_apply, euler_angles_to_matrix, quaternion_invert
# from tasks.csv_logger import CSVLogger


class State_Check():
    def __init__(self):
        with open("cfg/task/Dynasoar.yaml", "r") as cfg:
            try:
                self.cfg = yaml.safe_load(cfg)
            except yaml.YAMLError as exc:
                print(exc)
        print(self.cfg)

        self.num_envs = 2
        self.num_actions = 4
        self.num_observations = 13
        self.device = "cuda"
        self.dt = 0.02

        # # optimization flags for pytorch JIT
        # torch._C._jit_set_profiling_mode(False)
        # torch._C._jit_set_profiling_executor(False)

        # # Look at the first env
        # cam_pos = gymapi.Vec3(2, 1, 1)
        # cam_target = gymapi.Vec3(0, 0, 0)
        # self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # self.vel_loc = torch.zeros((self.num_envs, 3), device=self.device)
        # self.goal_pos = torch.zeros((self.num_envs, 3), device=self.device)

        self.roll_range = torch.arange(-10, 10, 1)
        # self.roll_range = [0]
        # self.pitch_range = torch.arange(-10, 10, 1)
        self.pitch_range = [0]
        # self.yaw_range = torch.arange(-10, 10, 1)
        self.yaw_range = [0]
        # self.wind_range = torch.arange(1, 30, 1)
        self.wind_range = [20.0]

        self.joy=Joystick()

        self.glider_params = {"wings": self.cfg["env"]["glider"]["wings"]["count"],
                              "station_pts": self.cfg["env"]["glider"]["station_pts"],
                              "cla": self.cfg["env"]["glider"]["cla"],
                              "rho": self.cfg["env"]["glider"]["rho"],
                              "eps": self.cfg["env"]["glider"]["eps"],
                              "offsets": self.cfg["env"]["glider"]["wings"]["offsets"],
                              "headings": self.cfg["env"]["glider"]["wings"]["headings"],
                              "chords": self.cfg["env"]["glider"]["wings"]["chords"],
                              "wing_pos": self.cfg["env"]["glider"]["wings"]["pos"],
                              "wind_speed":  torch.tensor(self.cfg["env"]["glider"]["wind_speed"]),
                              "wind_angle":  torch.tensor(self.cfg["env"]["glider"]["wind_angle"]),
                              "wind_randomize":  torch.tensor(self.cfg["env"]["glider"]["wind_randomize"]),
                              "wind_speed_min":  torch.tensor(self.cfg["env"]["glider"]["wind_speed_min"]),
                              "wind_speed_max":  torch.tensor(self.cfg["env"]["glider"]["wind_speed_max"]),
                              "wind_thickness":  torch.tensor(self.cfg["env"]["glider"]["wind_thickness"]),
                              "mass":  self.cfg["env"]["glider"]["mass"],
                              "ixx":  self.cfg["env"]["glider"]["ixx"],
                              "iyy":  self.cfg["env"]["glider"]["iyy"],
                              "izz":  self.cfg["env"]["glider"]["izz"],
                              "ixz":  self.cfg["env"]["glider"]["ixz"],
                              "Cd0":  self.cfg["env"]["glider"]["Cd0"],
                              "front_area":  self.cfg["env"]["glider"]["front_area"],
                              "batt_update":  self.cfg["env"]["glider"]["battery"]["update"],
                              "batt_cap":  self.cfg["env"]["glider"]["battery"]["capacity"],
                              "batt_init":  self.cfg["env"]["glider"]["battery"]["initial_charge"],
                              "thrust_update":  self.cfg["env"]["glider"]["thruster"]["update"],
                              "device":  (self.device),
                              "envs" : (self.num_envs),
                              "actions" : (self.num_actions),
                              "observations" : (self.num_observations),
                             }
        
        # base init state
        pos = self.cfg["env"]["baseInitState"]["pos"]
        rot = self.cfg["env"]["baseInitState"]["rot"]
        v_lin = self.cfg["env"]["baseInitState"]["vLinear"]
        v_ang = self.cfg["env"]["baseInitState"]["vAngular"]
        state = pos + rot + v_lin + v_ang

        self.base_init_state = state
        
        self.debug_flags = self.cfg["env"]["debug_flags"]
                                
        self.physics = LiftingLine(self.glider_params, self.dt, self.debug_flags)

        # self.get_state_tensors()

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
        self.sim = self.gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, self.sim_params)

        self._create_custom_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))
        self.get_state_tensors()

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # create viewer using the default camera properties
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        if self.viewer is None:
            raise ValueError('*** Failed to create viewer')
        
        # Look at the first env
        cam_pos = gymapi.Vec3(2, 1, 1)
        cam_target = gymapi.Vec3(0, 0, 0)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        self.vel_loc = torch.zeros((self.num_envs, 3), device=self.device)
        self.goal_pos = torch.zeros((self.num_envs, 3), device=self.device)

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

    def get_state_tensors(self):
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.initial_root_states = self.root_states.clone()
        self.num_actors=2
        self.glider_root_state = self.root_states.view(self.num_envs, self.num_actors, 13)[:, 0, :] #num_envs, num_actors, 13 (pos,ori,Lvel,Avel)
        self.pos = self.root_states.view(self.num_envs, self.num_actors, 13)[:, 0, 0:3] #num_envs, num_actors, 13 (pos,ori,Lvel,Avel)
        self.ori = self.root_states.view(self.num_envs, self.num_actors, 13)[:, 0, 3:7] #num_envs, num_actors, 13 (pos,ori,Lvel,Avel)
        self.linvel = self.root_states.view(self.num_envs, self.num_actors, 13)[:, 0, 7:10] #num_envs, num_actors, 13 (pos,ori,Lvel,Avel)
        self.angvel = self.root_states.view(self.num_envs, self.num_actors, 13)[:, 0, 10:13] #num_envs, num_actors, 13 (pos,ori,Lvel,Avel)
        self.gym.refresh_actor_root_state_tensor(self.sim)

    def simulation_loop(self):
        print('loop start')
        # self.gym.refresh_actor_root_state_tensor(self.sim)
        # self.gym.refresh_dof_state_tensor(self.sim)

        deg2rad = torch.pi/180

        for W in self.wind_range:
            for R in self.roll_range:
                for P in self.pitch_range:
                    for Y in self.yaw_range:
                        # print(P)
                        #Create a quaternion from the roll pitch and yaw
                        euler_angles = torch.tensor((Y*deg2rad,P*deg2rad,R*deg2rad))
                        RotMat = euler_angles_to_matrix(euler_angles, 'ZYX')
                        Q = matrix_to_quaternion(RotMat)
                        # print(Q)
                        #Record Force and Torque
                        #Plot
                        self.physics.wind_strength = torch.ones((self.num_envs,1), device=self.device)*W

                        self.glider_root_state[:, 0:2] = torch.zeros((2))
                        self.glider_root_state[:, 2] = torch.tensor(1)
                        self.glider_root_state[:,3:7] = torch.roll(Q, -1, 0)
                        self.glider_root_state[:, 7:10] = torch.zeros((3))
                        self.glider_root_state[:, 10:13] = torch.zeros((3))
                        
                        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
                        
                        actions = torch.zeros((self.num_envs, self.num_actions))


                        self.physics.update(self.glider_root_state, 
                                            actions,
                                            self.debug_flags)


                        # update the viewer
                        self.gym.step_graphics(self.sim)
                        self.gym.draw_viewer(self.viewer, self.sim, True)
                        
                    
                        # Wait for dt to elapse in real time.
                        # This synchronizes the physics simulation with the rendering rate.
                        # self.gym.sync_frame_time(self.sim)
                        # time.sleep(0.1)

        print('loop done')

        # self.gym.destroy_viewer(self.viewer)
        # self.gym.destroy_sim(self.sim)

sc = State_Check()
sc.create_sim()
print(sc.glider_params)

while(1):
    sc.simulation_loop()