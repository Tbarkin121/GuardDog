import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from isaacgym.torch_utils import *
# import pygame
from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_euler_angles, matrix_to_quaternion
from pytorch3d.transforms import  quaternion_to_matrix, quaternion_apply, quaternion_invert, quaternion_multiply
import matplotlib.pyplot as plt
from .csv_logger import CSVLogger

from .glider_components.battery_component import Battery
from .glider_components.battery_manager import Battery_Manager
from .glider_components.thruster_component import Thruster
from .glider_components.thruster_manager import Thruster_Manager

class LiftingLine:
    def __init__(self, glider_params, dt, debug_flags):
        # https://journals.sfu.ca/ts/index.php/ts/article/viewFile/42/38
        # https://www.researchgate.net/publication/318421320_Dynamic_Maneuver_Loads_Calculation_for_a_Sailplane_and_Comparison_with_Flight_Test/link/5d0ccb22a6fdcc2462982ede/download
        if(debug_flags['mode'] == 1):
            debug_flags['pos_xy_hold'] = True
            debug_flags['pos_z_hold'] = True
            debug_flags['quat_hold'] = True
            debug_flags['linvel_hold'] = True
            debug_flags['angvel_hold'] = True
            debug_flags['cam_follow'] = False
            debug_flags['joystick_quat'] = True
            debug_flags['control_en'] = True


        self.dt = dt
        self.glider_params = glider_params

        self.device = glider_params['device']
        self.mass = glider_params['mass']
        
        Ixx = glider_params['ixx']
        Ixz = glider_params['ixz']
        Iyy = glider_params['iyy']
        Izz = glider_params['izz']
        self.inertia = torch.tensor([[Ixx, 0.0, -Ixz],
                                     [0.0, Iyy, 0.0],
                                     [-Ixz, 0.0, Izz]], device=self.device) * self.mass


        self.eps = torch.tensor(self.glider_params["eps"], device=self.device)
        self.N = torch.tensor(self.glider_params["station_pts"], device=self.device)
        self.W = torch.tensor(self.glider_params["wings"], device=self.device)
        self.M = torch.tensor(self.glider_params["envs"], device=self.device)
        self.Cla = torch.tensor(self.glider_params["cla"], device=self.device)
        self.rho = torch.tensor(self.glider_params["rho"], device=self.device)

        self.Cd0 = glider_params['Cd0']
        self.frontal_area = glider_params['front_area']
        self.setup(self.glider_params)

        self.obs_raw = torch.zeros((self.glider_params["envs"], 6), device=self.device) 
        self.ground_speed_m1 = torch.zeros((self.glider_params["envs"], 1), device=self.device) 
        self.air_speed_m1 = torch.zeros((self.glider_params["envs"], 1), device=self.device) 

        self.root_state_dimensions = 16
        self.glider_states = torch.zeros((self.M, self.root_state_dimensions), device=self.device)
        self.kinematic_states = self.glider_states.view(self.M, self.root_state_dimensions)[:,0:13] # Pos(3), Ori(4), LinVel(3), AngVel(3) 
        self.battery_states = self.glider_states.view(self.M, self.root_state_dimensions)[:,13]     # Battery Energy(1)
        self.thruster_states = self.glider_states.view(self.M, self.root_state_dimensions)[:,14:16] # Rotor Ang Velocity(1), Rotor Pitch(1)
        
        batt_1 = Battery(0.05,                                      # Mass
                torch.tensor([0.0, 0.0, 0.0, 1.0], device='cuda'),  # Orientaiton
                torch.tensor([0.0, 0.0, 0.0], device='cuda'),       # Translation
                self.glider_params["batt_cap"],                     # Max Energy
                self.glider_params["batt_init"])                    # Initial Charge Percent
        
        batt_list = [batt_1]
        self.batt_man = Battery_Manager(self.device,
                                        self.M,
                                        self.dt,
                                        batt_list)
        
        self.batt_man.reset_battery_states(torch.arange(self.M, device=self.device))
        
        thuster_1 = Thruster(self.M,                                            # Num Envs
                             self.dt,
                             0.05,                                               # Mass
                             torch.tensor([-1.0, 0.0, 0.0], device='cuda'),  # Orientaiton
                             torch.tensor([0.0, 0.0, 0.0], device='cuda'))
        thrust_list = [thuster_1]
        self.thrust_man = Thruster_Manager('cuda',
                                    thrust_list)

        self.actions = torch.zeros((self.M,self.glider_params["actions"]), device=self.device)
        # self.get_dydt = lambda Y, A, O: torch.cat([self.physics_step(Y, A, O), 
        #                                             self.batt_man.physics_step(Y, A, O),
        #                                             self.thrust_man.physics_step(Y, A, O)], dim=1)
        self.get_dydt = lambda Y, A, O: (self.physics_step(Y, A, O)+
                                         self.batt_man.physics_step(Y, A, O)+
                                         self.thrust_man.physics_step(Y, A, O))

        # Log Files
        self.alpha_log = CSVLogger('alphas.csv', fields=['Env','A1','A2','A3','A4'], test_name=debug_flags['log_name'])
        self.state_global_log = CSVLogger('state_global.csv', fields=['Env','px', 'py', 'pz',
                                                                      'q1', 'q2', 'q3', 'q4',
                                                                      'vx', 'vy', 'vz', 
                                                                      'wx', 'wy', 'wz'], test_name=debug_flags['log_name'])
        self.euler_angles_log = CSVLogger('rk4_euler_angs.csv', fields=['Env','roll', 'pitch', 'yaw'], test_name=debug_flags['log_name'])
        self.lv_loc_log = CSVLogger('rk4_lv_loc.csv', fields=['Env','x', 'y', 'z'], test_name=debug_flags['log_name'])
        self.av_loc_log = CSVLogger('rk4_av_loc.csv', fields=['Env','x', 'y', 'z'], test_name=debug_flags['log_name'])
        self.as_log = CSVLogger('as.csv', fields=['Env','air_speed'], test_name=debug_flags['log_name'])
        self.gs_log = CSVLogger('gs.csv', fields=['Env','ground_speed_m1'], test_name=debug_flags['log_name'])
        self.force_log = CSVLogger('force.csv', fields=['Env','fx', 'fy', 'fz'], test_name=debug_flags['log_name'])
        self.torque_log = CSVLogger('torque.csv', fields=['Env','fx', 'fy', 'fz'], test_name=debug_flags['log_name'])

        prof_fields = ['Env']
        for i in range(self.glider_params['station_pts']*self.glider_params['wings']):
            prof_fields.append('s{}'.format(i))
        self.lift_prof = CSVLogger('lift_prof.csv',fields=prof_fields, test_name=debug_flags['log_name'])
        self.drag_prof = CSVLogger('drag_prof.csv',fields=prof_fields, test_name=debug_flags['log_name'])

        self.thruster_state_log = CSVLogger('t_state.csv',fields=['Env','Batt[J]', 'Pitch[Deg]', 'Omega[rad/sec]'], test_name=debug_flags['log_name'])
        self.thruster_power_log = CSVLogger('t_power.csv', fields=['Env','Motor[W]','Elec[W]', 'Prop[W]', 'Damp[W]'], test_name=debug_flags['log_name'])
        self.thruster_torque_log = CSVLogger('t_torque.csv', fields=['Env','Motor[Nm]', 'Prop[Nm]', 'Damp[Nm]'], test_name=debug_flags['log_name'])
        self.thruster_thrust_log = CSVLogger('t_thrust.csv', fields=['Env','Thrust[N]'], test_name=debug_flags['log_name'])

        self.goal_log = CSVLogger('goal.csv', fields=['Env','X[m]', 'Y[m]', 'Z[m]'], test_name=debug_flags['log_name'])

        self.wind_angle = torch.ones((self.M,1), device=self.device)*self.glider_params['wind_angle']
        self.wind_strength = torch.ones((self.M,1), device=self.device)*self.glider_params['wind_speed']
        if(self.glider_params['wind_randomize']):
            self.wind_randomize(torch.arange(self.M, device=self.device))

    def make_T_wing(self):
        T_list = []
        # root_offset_name = "TW"
        # for w in range(1,self.W+1):
        #     full_offset_name = root_offset_name + str(w)
        #     t = torch.tensor((self.glider_params[full_offset_name][0],self.glider_params[full_offset_name][1],self.glider_params[full_offset_name][2]), device=self.device)
        #     T_list.append(t)
        for offset in self.glider_params['offsets']:
            t = torch.tensor([offset[0], offset[1], offset[2]], device=self.device)
            T_list.append(t)
        
        T_wing = torch.cat(T_list)
        T_wing = torch.reshape(T_wing, (self.W,1,3))
        return T_wing


    def make_H_wing(self):
        H_list = []
        # root_heading_name = "HW"
        # for w in range(1,self.W+1):
        #     full_heading_name = root_heading_name + str(w)
        #     theta_x = self.glider_params[full_heading_name][0]*torch.pi/180.0
        #     theta_y = self.glider_params[full_heading_name][1]*torch.pi/180.0

        #     h_wing_x = torch.tensor( [[1.0, 0.0, 0.0],
        #                         [0.0, torch.cos(theta_x), torch.sin(theta_x)],
        #                         [0.0, -torch.sin(theta_x), torch.cos(theta_x)]], device=self.device)

        #     # Check Sines on sines later
        #     h_wing_y = torch.tensor( [[torch.cos(theta_y), 0.0, -torch.sin(theta_y)],
        #                             [0.0, 1.0, 0.0],
        #                             [torch.sin(theta_y), 0.0, torch.cos(theta_y)]], device=self.device)

        #     h = torch.matmul(h_wing_y, h_wing_x)
        #     h = torch.unsqueeze(h, dim=0)
        #     H_list.append(h)
        for heading in self.glider_params['headings']:
            theta_x = torch.tensor(heading[0]*torch.pi/180.0)
            theta_y = torch.tensor(heading[1]*torch.pi/180.0)
            
            h_wing_x = torch.tensor( [[1.0, 0.0, 0.0],
                                [0.0, torch.cos(theta_x), torch.sin(theta_x)],
                                [0.0, -torch.sin(theta_x), torch.cos(theta_x)]], device=self.device)

            # Check signs on sines later
            h_wing_y = torch.tensor( [[torch.cos(theta_y), 0.0, -torch.sin(theta_y)],
                                    [0.0, 1.0, 0.0],
                                    [torch.sin(theta_y), 0.0, torch.cos(theta_y)]], device=self.device)

            h = torch.matmul(h_wing_y, h_wing_x)
            h = torch.unsqueeze(h, dim=0)
            H_list.append(h)

        
        H_wing = torch.cat(H_list, dim=0)
        H_wing = torch.reshape(H_wing, (1,self.W*3,3))         
        
        return H_wing

    def make_trap_profile(self):
        theta_list = []
        Y_list = []
        C_list = []
        S_list = []
        
        for chord, pos in zip(self.glider_params['chords'], self.glider_params['wing_pos']):
            
            p1 = pos[-1] #Right Side (Positive)
            p2 = pos[0]  #Left Side (Negitive)
            s = torch.tensor((p1-p2)/2, device=self.device)

            theta = torch.reshape((torch.linspace(torch.pi-self.eps, self.eps, self.N, device=self.device)),[self.N,1]); # "station" locs
            y = s*torch.cos(theta)

            # c0 = chord[0]
            # c1 = chord[1]
            # # c = (y+s)*((c1-c0)/2.0/s)+c0
            # c = (c1-c0)/2*torch.cos(theta) + (c0+c1)/2
            dist = (torch.tensor(pos, device=self.device) - y) > 0 
            idx = torch.reshape(torch.arange(dist.shape[0], 0, -1, device=self.device), (-1,1))
            dist_num = dist*idx
            indices = torch.argmax(dist_num, 1, keepdim=True)
            p1 = torch.tensor(pos, device=self.device)[indices-1]
            p2 = torch.tensor(pos, device=self.device)[indices]
            c1 = torch.tensor(chord, device=self.device)[indices-1]
            c2 = torch.tensor(chord, device=self.device)[indices]

            lerp_percent = (y-p1) / (p2-p1)
            lerped_chord = torch.lerp(c1, c2, lerp_percent)

            theta_list.append(torch.unsqueeze(theta, 0))
            Y_list.append(torch.unsqueeze(y, 0))
            C_list.append(torch.unsqueeze(lerped_chord, 0))
            S_list.append(torch.unsqueeze(s, 0))

            
        theta = torch.cat(theta_list, dim=0)
        Y = torch.cat(Y_list, dim=0)
        C = torch.cat(C_list, dim=0)
        S = torch.cat(S_list, dim=0)
        S = torch.reshape(S, [self.W,1,1])
        
        return theta, Y, C, S

    def make_r_plane(self):
        H_ = self.H_wing[:, 1::3, :]
        H_ = torch.reshape(H_, (self.W,1,3))
        Y_ = torch.unsqueeze(self.Y_wn1[...,0], dim=-1)
        r_plane = H_*Y_ + self.T_wing

        return r_plane

    def setup(self, params):
        # self.old = time.perf_counter()
        self.T_wing = self.make_T_wing() # [W, 1, 3]
        self.H_wing = self.make_H_wing() # [1, W*3, 3]
        self.theta_wn1, self.Y_wn1, self.C_wn1, self.s_wn1 = self.make_trap_profile()
        self.r_plane = self.make_r_plane() #[W,N,3]

        self.vec1 = torch.sin(self.theta_wn1)*self.C_wn1*self.Cla/8.0/self.s_wn1 #[W,N,1]

        self.n = torch.reshape(torch.linspace(1,self.N,self.N, dtype=torch.float32, device=self.device),[1, 1,self.N]) #[1,1,W]
        self.mat1 = (self.n*self.C_wn1*self.Cla/8.0/self.s_wn1 +  torch.sin(self.theta_wn1))*torch.sin(self.n*self.theta_wn1)  #[W,N,N]

        self.mat1_LU, self.mat1_pivots = torch.linalg.lu_factor(self.mat1)
        self.mat2 = 4.0*self.s_wn1*torch.sin(self.n*self.theta_wn1)  #[W,N,N]
        self.mat3 = self.n/torch.sin(self.theta_wn1) * torch.sin(self.n*self.theta_wn1)   #[W,N,N]
        
    def wind_function(self, height, env_ids):
        k999 = 2*6.907
        k99 = 2*4.595
        
        
        
        thickness = self.glider_params['wind_thickness']
        center = thickness/2
        c = k999/thickness
        
        speed = torch.squeeze(self.wind_strength[env_ids,:])
        w_speed = speed /(1+torch.exp(-c*(height - center)))
        w_speed = torch.reshape(w_speed,[len(height),1,1])
        # wind = torch.cat((w_speed, torch.zeros_like(w_speed, device=self.device), torch.zeros_like(w_speed, device=self.device)), dim=1)
        #[M,3,1]
        w_speed_x = w_speed * torch.cos(torch.reshape(self.wind_angle[env_ids,:],[len(height),1,1]))
        w_speed_y = w_speed * torch.sin(torch.reshape(self.wind_angle[env_ids,:],[len(height),1,1]))
        wind = torch.cat((w_speed_x, w_speed_y, torch.zeros_like(w_speed, device=self.device)), dim=1)
        return wind
    
    def wind_randomize(self, env_ids):
        deg_2_rad = torch.pi/180.0
        new_wind_angles = torch_rand_float(-torch.pi, torch.pi, (len(env_ids), 1), device=self.device)
        # print(self.wind_angle[env_ids,:].shape)
        # print(new_wind_angles.shape)
        self.wind_angle[env_ids,:] = new_wind_angles

        new_wind_strengths = torch_rand_float(self.glider_params['wind_speed_min'], self.glider_params['wind_speed_max'], (len(env_ids), 1), device=self.device)
        self.wind_strength[env_ids,:] = new_wind_strengths


    def compute_force_moment(self, Y, actions):

        # self.now = time.perf_counter()
        # dtime = self.now - self.old
        # self.old = self.now;

        # print(1/dtime)

        glider_2_world_quat = Y[:,3:7]
        Quat = torch.unsqueeze(glider_2_world_quat, dim=1)
        Quat = torch.roll(Quat, 1, 2)


        Wind_global = self.wind_function(Y[:,2], torch.arange(self.M, device=self.device)) #[M,3,1]
        self.W_global = Wind_global #For Plotting Line Graphs

        # V_global = torch.cat([0.0*torch.ones([self.M,1,1]), 0.0*torch.ones([self.M,1,1]),  0.0*torch.ones([self.M,1,1])],1)
        V_com_global = torch.unsqueeze(Y[:,7:10], dim=2) #[M,3,1]
        #~~~~~~~~~~~~~~~~~
        r_plane_ = torch.reshape(self.r_plane, (1,self.N*self.W,3))
        r_global_ = quaternion_apply(Quat, r_plane_)
        self.r_global_mwn3 = torch.reshape(r_global_, (self.M,self.W,self.N,3))

        world_ang_vel = Y[:,10:13]
        world_ang_vel_repeated = torch.ones((1,self.W, self.N,1), device=self.device)*torch.reshape(world_ang_vel,[self.M,1,1,3])
        world_ang_vel_mwn3_cross_term = torch.reshape(world_ang_vel_repeated, [self.M*self.W*self.N, 3])

        r_global_mwn3_cross_term = torch.reshape(self.r_global_mwn3, [self.M*self.W*self.N, 3])
        station_pt_vel_reshaped = torch.cross(r_global_mwn3_cross_term, -world_ang_vel_mwn3_cross_term)
        station_pt_vel = torch.reshape(station_pt_vel_reshaped, [self.M, self.W, self.N, 3])
        
        Wind_apparent_global_body_m113 = torch.reshape(Wind_global, [self.M, 1, 1, 3]) - torch.reshape(V_com_global, [self.M, 1, 1, 3])
        Wind_apparent_global_body_m3 = torch.squeeze(Wind_apparent_global_body_m113)
        Wind_apparent_global_mwn3 = Wind_apparent_global_body_m113 - station_pt_vel #[M,W,N,3]

        self.WAG=Wind_apparent_global_body_m3

        self.air_speed_m1 = torch.reshape(torch.sqrt(torch.sum(Wind_apparent_global_body_m113**2,3)),[self.M,1]) #[M,1]
        Vinf_mwn1 = torch.reshape(torch.sqrt(torch.sum(Wind_apparent_global_mwn3**2,3)),[self.M,self.W,self.N,1]) #[M,W,N,1]

        Wind_apparent_global_normalized_mwn3 = Wind_apparent_global_mwn3/Vinf_mwn1 #[M,W,N,3]

        WAGN_matmul_term = torch.permute(Wind_apparent_global_normalized_mwn3, (0,1,3,2)) #[M,W,3,N]
        WingInWorldFrame = torch.reshape(quaternion_apply(Quat, self.H_wing),[self.M, self.W, 3, 3]) #[M,W,3,3]

        Wind_apparent_wing_normalized = torch.matmul(WingInWorldFrame, -WAGN_matmul_term) #[M,W,3,N]
        Wind_apparent_wing_normalized_mwn3 = torch.permute(Wind_apparent_wing_normalized, [0,1,3,2])
        

        alpha0_rad_mwn = torch.atan(Wind_apparent_wing_normalized_mwn3[..., 2]/ -Wind_apparent_wing_normalized_mwn3[..., 0])
        #~~~~~~~~~~
        alpha_wnm = torch.permute(alpha0_rad_mwn,(1,2,0))
        self.ground_speed_m1[:] = torch.norm(Y[:, 7:10], dim=-1).reshape(self.M,1)
        self.V_inf_mwn1 = Vinf_mwn1
        self.alpha = alpha_wnm

        c1_start = 0
        c1_end = int(self.N/4)
        c2_start = int(self.N*3/4)
        c2_end = int(self.N)

        # print('Needs Work!')
        # This part needs some thinking abou
        alpha_mod = torch.zeros([self.W,self.N,self.M], device=self.device)
        # alpha_mod[0,c1_start:c1_end,:] = (-actions[:,0])
        # alpha_mod[1,c2_start:c2_end,:] = (actions[:,0])
        # alpha_mod[2,:,:] = actions[:,1]
        # alpha_mod[3,:,:] = actions[:,1]
        alpha_mod[0,c1_start:c1_end,:] = -actions[:,0]
        alpha_mod[0,c2_start:c2_end,:] = actions[:,0]
        
        alpha_mod[1,:,:] = actions[:,1]
        alpha_mod[2,:,:] = actions[:,1]
        
        
        self.vec2 = alpha_wnm + alpha_mod
        # print(self.vec2[:,:,0])

        RHS = self.vec1*self.vec2
        self.RHS = RHS

        A = torch.linalg.lu_solve(self.mat1_LU, self.mat1_pivots, RHS)
        # A = torch.linalg.solve(self.mat1,RHS)
        
        # A = torch.matmul(mat1inv,RHS)
        # each col in above will have the "A" coeffs for the mth wing 

        Vinf_wnm = torch.squeeze(torch.permute(Vinf_mwn1, [1,2,0,3]))
        Gamma_wmn = torch.matmul(self.mat2,A)*Vinf_wnm
        # exec_time = time.perf_counter() - now; 
        Alpha_i = torch.matmul(self.mat3, A) 

        term = Gamma_wmn*Vinf_wnm*self.rho

        LiftDist_wnm1 = term*torch.cos(Alpha_i) #Needs to be reshaped
        DragDist_wnm1 = torch.abs(term*torch.sin(Alpha_i)) #Needs to be reshaped
        LiftDist_wnm1 = torch.reshape(LiftDist_wnm1, [self.W, self.N, self.M, 1])
        DragDist_wnm1 = torch.reshape(DragDist_wnm1, [self.W, self.N, self.M, 1])

        self.LD = LiftDist_wnm1.clone()
        self.DD = DragDist_wnm1.clone()

        # WingYWorld = WingInWorldFrame[:, :, 1]
        # WingYWorld = torch.reshape(WingYWorld, (self.M, self.W, 3))
        
        WingInWorldFrame = torch.reshape(WingInWorldFrame, [self.M, self.W, 3, 3]) #[envs, wings, 3(xl,yl,zl), 3(xg, yg, gz)] 
        WingYWorld = WingInWorldFrame[:, :, 1, :] #[envs, wings, 1Y, 3components] 
        WingYWorld = torch.unsqueeze(WingYWorld, dim=2)
        WingYWorld = WingYWorld.repeat(1,1,self.N,1) #Create 20 station points
        WingYWorld = torch.reshape(WingYWorld, [self.M*self.W*self.N, 3]) #Interleave

        Wind_apparent_global_normalized_mwn3
        WAGN_cross_term = torch.reshape(Wind_apparent_global_normalized_mwn3, [self.M*self.W*self.N, 3])
 
        Direction_Of_Lift_cross = torch.cross(WingYWorld, WAGN_cross_term, dim=1)
        Direction_Of_Lift_mwn3 = torch.reshape(Direction_Of_Lift_cross, [self.M, self.W, self.N, 3])

        #Normalizing Vector
        Direction_Of_Lift_norm_mwn1 = torch.unsqueeze(torch.norm( Direction_Of_Lift_mwn3, dim=3), dim=3)
        Direction_Of_Lift_mwn3 = Direction_Of_Lift_mwn3/Direction_Of_Lift_norm_mwn1 


        ZeroLiftDrag_m1 = torch.reshape(0.5 * self.rho * self.air_speed_m1**2 * self.Cd0 * self.frontal_area, (self.M,1))

        Wind_apparent_global_normalized_wnm3 = torch.permute(Wind_apparent_global_normalized_mwn3, [1,2,0,3])

        Direction_Of_Lift_wnm3 = torch.permute(Direction_Of_Lift_mwn3, [1,2,0,3])

        self.Lift_global_wnm3 = LiftDist_wnm1 * Direction_Of_Lift_wnm3
        self.Drag_global_wnm3 = DragDist_wnm1 * Wind_apparent_global_normalized_wnm3
        self.ZLD_global_m3 = ZeroLiftDrag_m1 * Wind_apparent_global_body_m3

        self.F_global_wnm3 = self.Lift_global_wnm3 + self.Drag_global_wnm3

        F_sum_wm3 = torch.trapz(self.F_global_wnm3, torch.unsqueeze(self.Y_wn1,dim=3), dim=1)
        self.CheckThisForce = F_sum_wm3
        F_sum_m3 = torch.sum(F_sum_wm3, dim=0)
        self.Force_sum_m3 = F_sum_m3 + self.ZLD_global_m3

        # print(actions[0,:])
        thrust_plane, torque_plane = self.thrust_man.thruster_list[0].update(self.air_speed_m1, 
                                                                             torch.unsqueeze(actions[:,2], dim=1), 
                                                                             torch.unsqueeze(actions[:,3],dim=1),
                                                                             debug_msg=False)
        
        if(self.glider_params["batt_update"]):
            self.batt_man.update_states(self.thrust_man.thruster_list[0].motor.electrical_power, debug_msg=False)
    
        thrust_global = quat_rotate(glider_2_world_quat, thrust_plane)
        # print(thrust_global[0,:])
        # print(self.Force_sum_m3[0,:])
        if(self.glider_params["thrust_update"]):
            self.Force_sum_m3 -= thrust_global

        self.r_global_wnm3 = torch.permute(self.r_global_mwn3, [1,2,0,3])
        # Appending onto the terms below 
        r_global_wnm3_cross_term = torch.reshape(self.r_global_wnm3, [self.W*self.N*self.M, 3])
        F_global_wnm3_cross_term = torch.reshape(self.F_global_wnm3, [self.W*self.N*self.M, 3])
        Torque_cross_term = torch.cross(r_global_wnm3_cross_term, F_global_wnm3_cross_term)
        
        # Split Off Thruster Terms
        Torque_wnm3 = torch.reshape(Torque_cross_term, [self.W, self.N, self.M, 3])
        T_sum_wm3 = torch.trapz(Torque_wnm3, torch.unsqueeze(self.Y_wn1,dim=3), dim=1)
        self.Torque_sum_m3 = torch.sum(T_sum_wm3, dim=0)

        self.CheckThisTorque = T_sum_wm3
        


        return self.Force_sum_m3, self.Torque_sum_m3

    def update(self, isaac_root_states, actions, debug_flags):
        self.kinematic_states[...] = isaac_root_states
        self.actions[:, :] = actions
        t = 0.0 #We aren't really using t for anything right now anyways
        # RK4_update = self.rk4(t, root_states, actions)
        self.Collect_Raw_Observation()
        self.ode_step(self.get_dydt, t)
        
        # print('-----------------------')
        # print(self.actions[0,:])
        # print(self.battery_states[0,...])
        # print(self.thruster_states[0,...])

        self.debug_modifications(debug_flags)
        
            
        return self.kinematic_states, self.alpha, self.obs_raw

    def ode_step(self, fun, t):
        self.glider_states[...] += fun(self.glider_states, self.actions, self.obs_raw)*self.dt
        self.kinematic_states[:, 3:7] = quat_unit(self.kinematic_states[:, 3:7])

    def physics_step(self, glider_states, actions, obs_raw):
        forces, moments = self.compute_force_moment(self.kinematic_states, actions)

        world_lin_vel = self.kinematic_states[:,7:10]
        world_ang_vel = self.kinematic_states[:,10:13]
        # world_ang_vel[:,2] = 1.0
        world_ang_vel = torch.unsqueeze(world_ang_vel, dim=-1)


        accel = forces / self.mass
        accel[:,2] += -9.81
        
        Quat = torch.unsqueeze(self.kinematic_states[:, 3:7], dim=1)
        Quat = torch.roll(Quat, 1, 2)
        rotated_inertia = quaternion_apply(Quat, torch.unsqueeze(self.inertia,dim=0))
        rotated_inertia = torch.permute(rotated_inertia,(0,2,1))
        rotated_inertia = quaternion_apply(Quat, rotated_inertia)
        rotated_inertia = torch.permute(rotated_inertia,(0,2,1))



        tmp_var = torch.matmul(rotated_inertia,world_ang_vel)
        tmp_var2 = torch.squeeze(torch.cross(world_ang_vel, tmp_var))
        tmp_var3 = moments - tmp_var2
        alpha = torch.linalg.solve(rotated_inertia, tmp_var3) # Rotate moment of inertia tensor    

        omega_mat = torch.zeros((self.M, 4, 4), device=self.device)
        omega_mat[:, 0, 1:4] = -world_ang_vel[:,:,0]
        omega_mat[:, 1:4, 0] = world_ang_vel[:,:,0]
        omega_mat[:, 1, 2:4] = torch.cat( (-world_ang_vel[:,2,:], 
                                        world_ang_vel[:,1,:]), dim=-1)

        omega_mat[:, 2, 1:4] = torch.cat( (world_ang_vel[:,2,:], 
                                        torch.zeros_like(world_ang_vel[:,2,:]), 
                                        -world_ang_vel[:,0,:]), dim=-1)

        omega_mat[:, 3, 1:3] = torch.cat( (-world_ang_vel[:,1,:], 
                                        world_ang_vel[:,0,:]), dim=-1)

        
        Quat = torch.unsqueeze(self.kinematic_states[:, 3:7], dim=-1)
        Quat = torch.roll(Quat, 1, 1)
        dQuat = torch.squeeze(0.5 * torch.matmul(omega_mat, Quat))
        dQuat = torch.roll(dQuat, -1, 1)

        dYdt_additions = torch.zeros( (self.M, 3), device=self.device) #This is the dy/dt for battery_state, rotor_speed, and rotor angle
        dYdt = torch.cat( (world_lin_vel, dQuat, accel, alpha, dYdt_additions), dim=1)

        return dYdt

    def Collect_Raw_Observation(self):
        base_quat = self.kinematic_states[:, 3:7]
        rot_mat = quaternion_to_matrix(torch.roll(base_quat, 1, 1))
        angles = matrix_to_euler_angles(rot_mat, 'ZYX')
        roll = torch.unsqueeze(angles[:,2], dim=-1)
        pitch = torch.unsqueeze(angles[:,1], dim=-1)
        yaw = torch.unsqueeze(angles[:,0], dim=-1)

        height = self.kinematic_states[:,2].reshape(self.M, 1)
 
        self.obs_raw[:] = torch.cat((height, 
                                     self.ground_speed_m1,
                                     self.air_speed_m1,
                                     roll,
                                     pitch,
                                     yaw
                                     ), dim=1)


    def log_writer(self, env_num):
        base_quat = self.glider_states[:, 3:7]
        en = torch.tensor((env_num,), device=self.device)
        glider_2_world_quat = torch.roll(base_quat[en,...], 1, 1)
        world_lin_vel = torch.unsqueeze(self.glider_states[env_num,7:10], 0)
        world_ang_vel = torch.unsqueeze(self.glider_states[env_num,10:13], 0)

        
        # Alpha
        self.alpha_log.write([torch.cat((en, self.alpha[:,0, env_num])).detach().cpu().numpy()])
        # Root State
        self.state_global_log.write([torch.cat((en, self.kinematic_states[env_num,...])).detach().cpu().numpy()])
        #Euler Angles
        
        
        rot_mat = quaternion_to_matrix(torch.roll(base_quat, 1, 1))
        angles = self.rotationMatrixToEulerAngles(rot_mat)
        ang_scale = 1/(torch.pi)
        roll = torch.unsqueeze(angles[env_num,0], dim=-1) * ang_scale
        pitch = torch.unsqueeze(angles[env_num,1], dim=-1) * ang_scale
        yaw = torch.unsqueeze(angles[env_num,2], dim=-1) * ang_scale
        self.euler_angles_log.write([torch.cat((en, roll, pitch, yaw)).detach().cpu().numpy()])
        #Linear Vel Local Frame
        
        plane_lin_vel = quat_rotate_inverse(base_quat[en,...], world_lin_vel.to(self.device))
        plane_ang_vel = quat_rotate_inverse(base_quat[en,...], world_ang_vel.to(self.device))
        
        self.lv_loc_log.write([torch.cat((en, plane_lin_vel[env_num,...])).detach().cpu().numpy()])
        #Angular Vel Local Frame
        self.av_loc_log.write([torch.cat((en, plane_ang_vel[env_num,...])).detach().cpu().numpy()])
        #Air Speed
        self.as_log.write([torch.cat((en, self.air_speed_m1[env_num,...])).detach().cpu().numpy()])
        #Ground Speed
        self.gs_log.write([torch.cat((en, self.ground_speed_m1[env_num,...])).detach().cpu().numpy()])

        # print(quaternion_invert(base_quat[en,...]))
        # print(quaternion_invert(base_quat[en,...]).shape)
        # print(self.Force_sum[en, ...])
        # print(self.Force_sum[en, ...].shape)

        
        F_local = torch.squeeze(quaternion_apply(quaternion_invert(glider_2_world_quat), self.Force_sum_m3[en, ...] ))
        # F_local = torch.squeeze(self.Force_sum[en, ...])
        self.force_log.write([torch.cat((en, F_local)).detach().cpu().numpy()])
        T_local = torch.squeeze(quaternion_apply(quaternion_invert(glider_2_world_quat), self.Torque_sum_m3[en, ...] ))
        self.torque_log.write([torch.cat((en, T_local)).detach().cpu().numpy()])

        self.lift_prof.write([torch.cat((en, 
                                         torch.squeeze(self.LD[0,:,en]), 
                                         torch.squeeze(self.LD[1,:,en]), 
                                         torch.squeeze(self.LD[2,:,en]), 
                                         torch.squeeze(self.LD[3,:,en]))).detach().cpu().numpy()])
        self.drag_prof.write([torch.cat((en, 
                                         torch.squeeze(self.DD[0,:,en]), 
                                         torch.squeeze(self.DD[1,:,en]), 
                                         torch.squeeze(self.DD[2,:,en]), 
                                         torch.squeeze(self.DD[3,:,en]))).detach().cpu().numpy()])
        
        # self.thruster_state_log = CSVLogger('t_batt.csv',fields=["Batt[J]", "Pitch[Deg]", "Omega[rad/sec]"], test_name=debug_flags['log_name'])
        self.thruster_state_log.write([torch.cat((en, 
                                                 self.batt_man.battery_state[env_num,:], 
                                                 torch.unsqueeze(self.thrust_man.thruster_list[0].prop.pitch,dim=1)[env_num,:], 
                                                 self.thrust_man.thruster_list[0].omega[env_num,:])).detach().cpu().numpy()])
        # self.thruster_power_log = CSVLogger('t_power.csv', fields=['Motor[W]','Elec[W]', 'Prop[W]', 'Damp[W]'], test_name=debug_flags['log_name'])
        self.thruster_power_log.write([torch.cat((en, 
                                                  self.thrust_man.thruster_list[0].motor.mechanical_power[env_num,:], 
                                                  self.thrust_man.thruster_list[0].motor.electrical_power[env_num,:], 
                                                  self.thrust_man.thruster_list[0].prop.mechanical_power[env_num,:],
                                                  self.thrust_man.thruster_list[0].damping_power[env_num,:])).detach().cpu().numpy()])
        # self.thruster_torque_log = CSVLogger('t_thrust.csv', fields=['Motor[Nm]', 'Prop[Nm]', 'Damp[Nm]'], test_name=debug_flags['log_name'])
        self.thruster_torque_log.write([torch.cat((en, 
                                                   self.thrust_man.thruster_list[0].motor_torque[env_num,:], 
                                                   self.thrust_man.thruster_list[0].prop_torque[env_num,:],
                                                   self.thrust_man.thruster_list[0].damping_torque[env_num,:])).detach().cpu().numpy()])
        # self.thruster_thrust_log = CSVLogger('t_torque.csv', fields=['Thrust[N]'], test_name=debug_flags['log_name'])
        self.thruster_thrust_log.write([torch.cat((en,
                                                  self.thrust_man.thruster_list[0].prop_thrust[env_num,:])).detach().cpu().numpy()])
        
        self.goal_log.write([torch.cat((en, 
                                       self.goal_states[env_num,0:3])).detach().cpu().numpy()])



        
        
    def print_all_tensors(self):
        name = ''
        value = 0
        num_tensors = 0
        for name, value in zip(vars(self).keys(), vars(self).values()):
            if(torch.is_tensor(value)):
                print('Tensor Name  : {}'.format(name))
                print('Tensor Shape : {}'.format(value.shape))
                num_tensors += 1
        print('Total Tensors : {}'.format(num_tensors))

    def debug_modifications(self, debug_flags):
        if(debug_flags['pos_xy_hold']):
            self.kinematic_states[:, 0:2] = 0.0
        if(debug_flags['pos_z_hold']):
            self.kinematic_states[:, 2] = 10.0

        
        if(debug_flags['quat_hold']):
            self.kinematic_states[:, 3:6] = 0.0                                            
            self.kinematic_states[:, 6] = 1.0

        if(debug_flags['linvel_hold']):
            self.kinematic_states[:, 7:10] = 0.0                                          
        if(debug_flags['angvel_hold']):
            self.kinematic_states[:, 10:13] = 0.0                                        

        # if(debug_flags['joystick_quat']):
        #     euler_angles = torch.zeros([1,3], device=self.device)
        #     threshold = 0.1
        #     if(torch.abs(actions[0,3]) > threshold):
        #         euler_angles[0,0] = (actions[0,3] - torch.sign(actions[0,3])*threshold)/10.0
        #     if(torch.abs(actions[0,1]) > threshold):
        #         euler_angles[0,1] = (actions[0,1] - torch.sign(actions[0,1])*threshold)/10.0
        #     if(torch.abs(actions[0,0]) > threshold):
        #         euler_angles[0,2] = (actions[0,0] - torch.sign(actions[0,0])*threshold)/10.0
        #     RQ = matrix_to_quaternion(euler_angles_to_matrix(euler_angles, 'ZYX'))
        #     root_states[:,3:7] = quaternion_multiply(RQ, root_states[:,3:7])

        # self.print_all_tensors()

    def glider_log(self, debug_flags, goal_states):
        self.goal_states = goal_states
        if(debug_flags['logging']):
            for m in debug_flags['log_envs']:
                self.log_writer(m)

