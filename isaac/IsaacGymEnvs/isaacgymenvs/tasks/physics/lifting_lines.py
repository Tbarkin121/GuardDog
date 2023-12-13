import torch

from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_euler_angles, matrix_to_quaternion
from pytorch3d.transforms import quaternion_to_matrix, quaternion_apply, quaternion_invert, quaternion_multiply


class LiftingLines:
    def __init__(self, cfg, wind):
        print('Init Lifting Lines')
        self.wind = wind
        self.plot_dict = {}
        self.log_dict = {}
        self.extra_dict = {}

        self.init_inertial_parameters(cfg)
        self.init_aero_parameters(cfg)
        self.init_liftinglines_params(cfg)

    def init_inertial_parameters(self, cfg):
        self.mass = cfg['env']['glider']['mass']
        Ixx = cfg["env"]["glider"]["ixx"]
        Ixz = cfg["env"]["glider"]["ixz"]
        Iyy = cfg["env"]["glider"]["iyy"]
        Izz = cfg["env"]["glider"]["izz"]
        self.inertia = torch.tensor([[Ixx, 0.0, -Ixz],
                                     [0.0, Iyy, 0.0],
                                     [-Ixz, 0.0, Izz]]) * self.mass
        
    def init_aero_parameters(self, cfg):
        self.Cla = torch.tensor(cfg["env"]["glider"]["cla"])
        self.rho = torch.tensor(cfg["env"]["glider"]["rho"])
        self.Cd0 = cfg["env"]["glider"]["Cd0"]
        self.front_area = cfg["env"]["glider"]["front_area"]
        self.eps = cfg["env"]["glider"]["eps"]

    def init_liftinglines_params(self, cfg):
        self.N = torch.tensor(cfg["env"]["glider"]["station_pts"])
        self.W = torch.tensor(cfg["env"]["glider"]["wings"]["count"])
        self.M = torch.tensor(cfg["env"]["num_envs"])

        self.T_wing = self.make_T_wing(cfg) # [W, 1, 3]
        self.H_wing = self.make_H_wing(cfg) # [1, W*3, 3]
        self.theta_wn1, self.Y_wn1, self.C_wn1, self.s_wn1 = self.make_trap_profile(cfg)
        self.r_plane = self.make_r_plane() #[W,N,3]

        self.vec1 = torch.sin(self.theta_wn1)*self.C_wn1*self.Cla/8.0/self.s_wn1 #[W,N,1]
        self.n = torch.reshape(torch.linspace(1,self.N,self.N, dtype=torch.float32),[1, 1,self.N]) #[1,1,W]
        self.mat1 = (self.n*self.C_wn1*self.Cla/8.0/self.s_wn1 +  torch.sin(self.theta_wn1))*torch.sin(self.n*self.theta_wn1)  #[W,N,N]
        self.mat1_LU, self.mat1_pivots = torch.linalg.lu_factor(self.mat1)
        self.mat2 = 4.0*self.s_wn1*torch.sin(self.n*self.theta_wn1)  #[W,N,N]
        self.mat3 = self.n/torch.sin(self.theta_wn1) * torch.sin(self.n*self.theta_wn1)   #[W,N,N]


    def make_T_wing(self, cfg):
        T_list = []

        for offset in cfg["env"]["glider"]["wings"]["offsets"]:
            t = torch.tensor([offset[0], offset[1], offset[2]])
            T_list.append(t)
        
        T_wing = torch.cat(T_list)
        T_wing = torch.reshape(T_wing, (self.W,1,3))
        return T_wing
    
    def make_H_wing(self, cfg):
        H_list = []

        for heading in cfg["env"]["glider"]["wings"]["headings"]:
            theta_x = torch.tensor(heading[0]*torch.pi/180.0)
            theta_y = torch.tensor(heading[1]*torch.pi/180.0)
            
            h_wing_x = torch.tensor( [[1.0, 0.0, 0.0],
                                [0.0, torch.cos(theta_x), torch.sin(theta_x)],
                                [0.0, -torch.sin(theta_x), torch.cos(theta_x)]])

            # Check signs on sines later
            h_wing_y = torch.tensor( [[torch.cos(theta_y), 0.0, -torch.sin(theta_y)],
                                    [0.0, 1.0, 0.0],
                                    [torch.sin(theta_y), 0.0, torch.cos(theta_y)]])

            h = torch.matmul(h_wing_y, h_wing_x)
            h = torch.unsqueeze(h, dim=0)
            H_list.append(h)

        
        H_wing = torch.cat(H_list, dim=0)
        H_wing = torch.reshape(H_wing, (1,self.W*3,3))         
        
        return H_wing
    

    def make_trap_profile(self, cfg):
        theta_list = []
        Y_list = []
        C_list = []
        S_list = []
        
        for chord, pos in zip(cfg["env"]["glider"]["wings"]["chords"], cfg["env"]["glider"]["wings"]["pos"]):
            
            p1 = pos[-1] #Right Side (Positive)
            p2 = pos[0]  #Left Side (Negitive)
            s = torch.tensor((p1-p2)/2)

            theta = torch.reshape((torch.linspace(torch.pi-self.eps, self.eps, self.N)),[self.N,1]); # "station" locs
            y = s*torch.cos(theta)

            dist = (torch.tensor(pos) - y) > 0 
            idx = torch.reshape(torch.arange(dist.shape[0], 0, -1), (-1,1))
            dist_num = dist*idx
            indices = torch.argmax(dist_num, 1, keepdim=True)
            p1 = torch.tensor(pos)[indices-1]
            p2 = torch.tensor(pos)[indices]
            c1 = torch.tensor(chord)[indices-1]
            c2 = torch.tensor(chord)[indices]

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


    def compute_force_moment(self, states, actions):
        glider_2_world_quat = states[:,3:7].view(self.M, 1, 4)
        Quat = torch.roll(glider_2_world_quat, 1, 2)

        height = states[:,2]
        Wind_global = self.wind.get_wind_vector(height, torch.arange(self.M)) #[M,3,1]
        self.plot_dict['Wind Global'] = Wind_global

        # V_global = torch.cat([0.0*torch.ones([self.M,1,1]), 0.0*torch.ones([self.M,1,1]),  0.0*torch.ones([self.M,1,1])],1)
        V_com_global = states[:,7:10].view(self.M, 3, 1)
        #~~~~~~~~~~~~~~~~~
        r_plane_ = torch.reshape(self.r_plane, (1,self.N*self.W,3))
        r_global_ = quaternion_apply(Quat, r_plane_)
        r_global_mwn3 = torch.reshape(r_global_, (self.M,self.W,self.N,3))

        world_ang_vel = states[:,10:13]
        world_ang_vel_repeated = torch.ones((1,self.W, self.N,1))*torch.reshape(world_ang_vel,[self.M,1,1,3])
        world_ang_vel_mwn3_cross_term = torch.reshape(world_ang_vel_repeated, [self.M*self.W*self.N, 3])

        r_global_mwn3_cross_term = torch.reshape(r_global_mwn3, [self.M*self.W*self.N, 3])
        station_pt_vel_reshaped = torch.cross(r_global_mwn3_cross_term, -world_ang_vel_mwn3_cross_term)
        station_pt_vel = torch.reshape(station_pt_vel_reshaped, [self.M, self.W, self.N, 3])
        
        Wind_apparent_global_body_m113 = torch.reshape(Wind_global, [self.M, 1, 1, 3]) - torch.reshape(V_com_global, [self.M, 1, 1, 3])
        Wind_apparent_global_body_m3 = torch.squeeze(Wind_apparent_global_body_m113)
        Wind_apparent_global_mwn3 = Wind_apparent_global_body_m113 - station_pt_vel #[M,W,N,3]

        self.plot_dict['Wind Apparent Global'] = Wind_apparent_global_body_m3

        air_speed_m1 = torch.reshape(torch.sqrt(torch.sum(Wind_apparent_global_body_m113**2,3)),[self.M,1]) #[M,1]
        self.log_dict['Air Speed'] = air_speed_m1

        Vinf_mwn1 = torch.reshape(torch.sqrt(torch.sum(Wind_apparent_global_mwn3**2,3)),[self.M,self.W,self.N,1]) #[M,W,N,1]

        Wind_apparent_global_normalized_mwn3 = Wind_apparent_global_mwn3/Vinf_mwn1 #[M,W,N,3]

        WAGN_matmul_term = torch.permute(Wind_apparent_global_normalized_mwn3, (0,1,3,2)) #[M,W,3,N]
        WingInWorldFrame = torch.reshape(quaternion_apply(Quat, self.H_wing),[self.M, self.W, 3, 3]) #[M,W,3,3]

        Wind_apparent_wing_normalized = torch.matmul(WingInWorldFrame, -WAGN_matmul_term) #[M,W,3,N]
        Wind_apparent_wing_normalized_mwn3 = torch.permute(Wind_apparent_wing_normalized, [0,1,3,2])
        

        alpha0_rad_mwn = torch.atan(Wind_apparent_wing_normalized_mwn3[..., 2]/ -Wind_apparent_wing_normalized_mwn3[..., 0])
        #~~~~~~~~~~
        alpha_wnm = torch.permute(alpha0_rad_mwn,(1,2,0))
        ground_speed_m1 = torch.norm(states[:,7:10], dim=-1).reshape(self.M,1)
        self.log_dict['Ground Speed'] = ground_speed_m1

        self.V_inf_mwn1 = Vinf_mwn1
        self.log_dict['Alpha'] = alpha_wnm

        alpha_mod = torch.zeros([self.W,self.N,self.M])

        # This part needs some thinking about
        c1_start = 0
        c1_end = int(self.N/4)
        c2_start = int(self.N*3/4)
        c2_end = int(self.N)

        alpha_mod[0,c1_start:c1_end,:] = (-actions[:,0])
        alpha_mod[1,c2_start:c2_end,:] = (actions[:,0])
        alpha_mod[2,:,:] = actions[:,1]
        alpha_mod[3,:,:] = actions[:,1]
        # alpha_mod[0,c1_start:c1_end,:] = -actions[:,0]
        # alpha_mod[0,c2_start:c2_end,:] = actions[:,0]
        # alpha_mod[1,:,:] = actions[:,1]
        # alpha_mod[2,:,:] = actions[:,1]
        
        
        self.vec2 = alpha_wnm + alpha_mod
        # print(self.vec2[:,:,0])

        RHS = self.vec1*self.vec2
        self.RHS = RHS

        # each col will have the "A" coeffs for the mth wing
        A = torch.linalg.lu_solve(self.mat1_LU, self.mat1_pivots, RHS)
        # A = torch.linalg.solve(self.mat1,RHS)
        # A = torch.matmul(mat1inv,RHS)
 

        Vinf_wnm = torch.squeeze(torch.permute(Vinf_mwn1, [1,2,0,3]))
        Gamma_wmn = torch.matmul(self.mat2,A)*Vinf_wnm
        Alpha_i = torch.matmul(self.mat3, A) 

        term = Gamma_wmn*Vinf_wnm*self.rho

        LiftDist_wnm1 = term*torch.cos(Alpha_i) #Needs to be reshaped
        DragDist_wnm1 = torch.abs(term*torch.sin(Alpha_i)) #Needs to be reshaped
        LiftDist_wnm1 = torch.reshape(LiftDist_wnm1, [self.W, self.N, self.M, 1])
        DragDist_wnm1 = torch.reshape(DragDist_wnm1, [self.W, self.N, self.M, 1])

        self.log_dict['Lift Dist'] = LiftDist_wnm1
        self.log_dict['Drag Dist'] = DragDist_wnm1

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


        ZeroLiftDrag_m1 = torch.reshape(0.5 * self.rho * air_speed_m1**2 * self.Cd0 * self.front_area, (self.M,1))

        Wind_apparent_global_normalized_wnm3 = torch.permute(Wind_apparent_global_normalized_mwn3, [1,2,0,3])

        Direction_Of_Lift_wnm3 = torch.permute(Direction_Of_Lift_mwn3, [1,2,0,3])

        Lift_global_wnm3 = LiftDist_wnm1 * Direction_Of_Lift_wnm3
        Drag_global_wnm3 = DragDist_wnm1 * Wind_apparent_global_normalized_wnm3
        ZLD_global_m3 = ZeroLiftDrag_m1 * Wind_apparent_global_body_m3
        F_global_wnm3 = Lift_global_wnm3 + Drag_global_wnm3

        self.plot_dict['Lift Global'] = Lift_global_wnm3
        self.plot_dict['Drag Global'] = Drag_global_wnm3
        self.plot_dict['Zero Lift Drag Global'] = ZLD_global_m3
        self.plot_dict['F Global'] = ZLD_global_m3

        F_sum_wm3 = torch.trapz(F_global_wnm3, torch.unsqueeze(self.Y_wn1,dim=3), dim=1)
        # self.CheckThisForce = F_sum_wm3
        F_sum_m3 = torch.sum(F_sum_wm3, dim=0)
        Force_sum_m3 = F_sum_m3 + ZLD_global_m3

        r_global_wnm3 = torch.permute(r_global_mwn3, [1,2,0,3])
        self.plot_dict['R Global'] = r_global_wnm3
        # Appending onto the terms below 
        r_global_wnm3_cross_term = torch.reshape(r_global_wnm3, [self.W*self.N*self.M, 3])
        F_global_wnm3_cross_term = torch.reshape(F_global_wnm3, [self.W*self.N*self.M, 3])
        Torque_cross_term = torch.cross(r_global_wnm3_cross_term, F_global_wnm3_cross_term)
        
        # Split Off Thruster Terms
        Torque_wnm3 = torch.reshape(Torque_cross_term, [self.W, self.N, self.M, 3])
        T_sum_wm3 = torch.trapz(Torque_wnm3, torch.unsqueeze(self.Y_wn1,dim=3), dim=1)
        Torque_sum_m3 = torch.sum(T_sum_wm3, dim=0)

        # self.CheckThisTorque = T_sum_wm3
        
        return Force_sum_m3, Torque_sum_m3
    


    def physics_step(self, states, actions):
        forces, moments = self.compute_force_moment(states, actions)

        world_lin_vel = states[:,7:10]
        world_ang_vel = states[:,10:13]
        # world_ang_vel[:,2] = 1.0
        world_ang_vel = torch.unsqueeze(world_ang_vel, dim=-1)


        accel = forces / self.mass
        accel[:,2] += -9.81
        

        glider_2_world_quat = states[:,3:7].view(self.M, 1, 4)
        Quat = torch.roll(glider_2_world_quat, 1, 2)

        rotated_inertia = quaternion_apply(Quat, torch.unsqueeze(self.inertia,dim=0))
        rotated_inertia = torch.permute(rotated_inertia,(0,2,1))
        rotated_inertia = quaternion_apply(Quat, rotated_inertia)
        rotated_inertia = torch.permute(rotated_inertia,(0,2,1))




        tmp_var = torch.matmul(rotated_inertia,world_ang_vel)
        tmp_var2 = torch.squeeze(torch.cross(world_ang_vel, tmp_var))
        tmp_var3 = moments - tmp_var2
        alpha = torch.linalg.solve(rotated_inertia, tmp_var3) # Rotate moment of inertia tensor    

        omega_mat = torch.zeros((self.M, 4, 4))
        omega_mat[:, 0, 1:4] = -world_ang_vel[:,:,0]
        omega_mat[:, 1:4, 0] = world_ang_vel[:,:,0]
        omega_mat[:, 1, 2:4] = torch.cat( (-world_ang_vel[:,2,:], 
                                        world_ang_vel[:,1,:]), dim=-1)

        omega_mat[:, 2, 1:4] = torch.cat( (world_ang_vel[:,2,:], 
                                        torch.zeros_like(world_ang_vel[:,2,:]), 
                                        -world_ang_vel[:,0,:]), dim=-1)

        omega_mat[:, 3, 1:3] = torch.cat( (-world_ang_vel[:,1,:], 
                                        world_ang_vel[:,0,:]), dim=-1)

        
        Quat = torch.unsqueeze(states[:, 3:7], dim=-1)
        Quat = torch.roll(Quat, 1, 1)
        dQuat = torch.squeeze(0.5 * torch.matmul(omega_mat, Quat))
        dQuat = torch.roll(dQuat, -1, 1)

        dYdt_additions = torch.zeros( (self.M, 3)) #This is the dy/dt for battery_state, rotor_speed, and rotor angle
        dYdt = torch.cat( (world_lin_vel, dQuat, accel, alpha, dYdt_additions), dim=1)

        return dYdt