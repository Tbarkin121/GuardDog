import torch
from .physics.torch_utils import * 

from .physics.lifting_lines import LiftingLines
from .physics.wind import Wind

from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_euler_angles, matrix_to_quaternion
from pytorch3d.transforms import quaternion_to_matrix, quaternion_apply, quaternion_invert, quaternion_multiply

class DynasoarEnv:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dt = cfg['sim']['dt']
        self.device = 'cuda:0'
        torch.set_default_device(self.device)

        self.debug_flags=None
        
        # Does this no grad work for lower level classes?
        with torch.no_grad():
            self.init_data_buffers(cfg)
            self.init_starting_state_vars(cfg)
            self.init_extras(cfg)
            self.wind = Wind(cfg, self.wind_strength_buf, self.wind_angle_buf)
            self.physics = LiftingLines(cfg, self.wind)

        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        
            

    def step(self, actions):
        self.scale_and_store_actions(actions)

        dYdt = self.physics.physics_step(self.kinematic_states, self.actions_buf)
        self.states_buf += dYdt*self.dt
        self.kinematic_states[:, 3:7] = quat_unit(self.kinematic_states[:, 3:7])
        

        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        env_ids = self.goal_reached.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_goals(env_ids)

        self.get_observations()
        self.get_rewards()



        return self.kinematic_states, self.obs_buf, self.actions_buf, self.rew_buf, self.goal_states
    
    def init_data_buffers(self, cfg):
        self.num_envs     = cfg['env']['num_envs']
        self.num_acts     = 4           # Roll(1), Pitch(1), MotorTorque(1), Prop Pitch(1)
        self.num_obs      = 13           # Height(1), Ground Speed(1), Air Speed(1), Ori(3), Battery(1), Vec2Target(2), Dist2Target(1), PropSpeed(1), MotorPower(1), Prop Pitch(1)
        self.num_states   = 16        # Pos(3), Quat(4), Vel(3), AngVel(3), BatteryEnergy(1), Rotor AngVel(1), RotorPitch(1)
        # Data Storage (RL)
        self.obs_buf      = torch.zeros( (self.num_envs, self.num_obs), dtype=torch.float)
        self.states_buf   = torch.zeros( (self.num_envs, self.num_states), dtype=torch.float)
        self.goal_states  = torch.zeros( (self.num_envs, 3), dtype=torch.float )
        self.actions_buf  = torch.zeros( (self.num_envs, self.num_acts), dtype=torch.float)
        self.rew_buf      = torch.zeros(self.num_envs, dtype=torch.float)
        self.reset_buf    = torch.ones(self.num_envs, dtype=torch.long)
        self.timeout_buf  = torch.zeros(self.num_envs, dtype=torch.long)
        self.progress_buf = torch.zeros(self.num_envs, dtype=torch.long)
        # Memory Slices
        self.kinematic_states  = self.states_buf.view(self.num_envs, self.num_states)[:,0:13]   # Pos(3), Ori(4), LinVel(3), AngVel(3) 
        self.kinematic_states[:, 6] = 1.0                                                       # Fix the Quaternion
        self.kinematic_states_prev = self.kinematic_states.clone()
        self.battery_states    = self.states_buf.view(self.num_envs, self.num_states)[:,13]     # Battery Energy(1)
        self.thruster_states   = self.states_buf.view(self.num_envs, self.num_states)[:,14:16]  # Rotor Ang Velocity(1), Rotor Pitch(1)
        # Data Storage (Env)
        self.wind_angle_buf    = torch.ones((self.num_envs,1))*cfg['env']['glider']['wind_angle']
        self.wind_strength_buf = torch.ones((self.num_envs,1))*cfg['env']['glider']['wind_speed']
        # Data Storage (Goal)
        self.target_pos = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float)
        self.goal_reached = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.use_goal_list = self.cfg["env"]["goal"]["use_goal_list"]
        self.goal_list = torch.tensor(self.cfg["env"]["goal"]["goal_list"], device=self.device)
        self.goal_idx = torch.zeros((self.num_envs,), dtype=torch.long)
        self.num_goals = self.goal_list.shape[0]

    def init_starting_state_vars(self, cfg):
        # Starting State Ranges 
        self.roll_range = cfg["env"]["randomRanges"]["roll"]
        self.pitch_range = cfg["env"]["randomRanges"]["pitch"]
        self.yaw_range = cfg["env"]["randomRanges"]["yaw"]
        self.vinf_range = cfg["env"]["randomRanges"]["vinf"]
        self.height_range = cfg["env"]["randomRanges"]["height"]
        self.target_range = cfg["env"]["goal"]["target_range"]
        self.target_radius = cfg["env"]["goal"]["target_radius"]

        # base init state
        pos = self.cfg["env"]["baseInitState"]["pos"]
        rot = self.cfg["env"]["baseInitState"]["rot"]
        v_lin = self.cfg["env"]["baseInitState"]["vLinear"]
        v_ang = self.cfg["env"]["baseInitState"]["vAngular"]
        state = pos + rot + v_lin + v_ang
        self.base_init_state = state
        self.initial_glider_states = self.kinematic_states.clone()
        self.initial_glider_states[:] = to_torch(self.base_init_state, device=self.device, requires_grad=False)

    def init_extras(self, cfg):
        self.rew_scales = {}
        self.rew_scales["heading"] = self.cfg["env"]["learn"]["headingRewardScale"]
        self.rew_scales["energy"] = self.cfg["env"]["learn"]["energyRewardScale"]

        for key in self.rew_scales.keys():
            self.rew_scales[key] *= self.dt

        self.obs_scales = self.cfg["env"]["obs_scales"]

        self.action_scale_main = self.cfg["env"]["control"]["actionScale_main"]
        self.action_scale_tail = self.cfg["env"]["control"]["actionScale_tail"]

        self.max_episode_length_s = self.cfg["env"]["learn"]["episodeLength_s"]
        self.max_episode_length = int(self.max_episode_length_s / self.dt + 0.5)

    def get_rewards(self):
        self.rew_buf[:], self.reset_buf[:], self.goal_reached[:] = calc_rewards_jit(
            # tensors
            self.kinematic_states,
            self.kinematic_states_prev,
            self.actions_buf,
            self.progress_buf,
            self.physics.log_dict['Alpha'],
            self.physics.log_dict['Air Speed'], 
            self.obs_buf,
            self.target_pos,
            # Dict
            self.rew_scales,
            # other
            self.max_episode_length,
            self.num_envs,
            self.target_radius,
        )
        self.kinematic_states_prev = self.kinematic_states.clone()
        # print(self.rew_buf)

    def get_observations(self):
        # State Refresh happens in 
        base_quat = self.kinematic_states[:, 3:7]
        rot_mat = quaternion_to_matrix(torch.roll(base_quat, 1, 1))
        angles = rotationMatrixToEulerAngles(rot_mat) # This is an ZYX format but returns X,Y,Z... 
        # angles = matrix_to_euler_angles(rot_mat, 'XYZ')
        ang_scale = 1/(torch.pi)

        roll = torch.unsqueeze(angles[:,0], dim=-1) * ang_scale
        pitch = torch.unsqueeze(angles[:,1], dim=-1) * ang_scale
        yaw = torch.unsqueeze(angles[:,2], dim=-1) * ang_scale

        pos = self.kinematic_states[:, 0:3] * torch.tensor([self.obs_scales['height']])
        # height = torch.unsqueeze(height, dim=-1)

        air_speed = self.physics.log_dict['Air Speed'].reshape(self.num_envs,1)  * torch.tensor([self.obs_scales['air_speed']])

        ground_speed = self.physics.log_dict['Ground Speed'].reshape(self.num_envs,1)  * torch.tensor([self.obs_scales['ground_speed']])
        
        v_to_target_g = self.target_pos - self.kinematic_states[:, 0:2]
        v_to_target_norm = torch.unsqueeze(torch.norm(v_to_target_g, dim=1),dim=1)
        v_to_target_g = torch.unsqueeze(v_to_target_g/v_to_target_norm, dim=-1)
        v_to_target_g = torch.cat((v_to_target_g, torch.ones((self.num_envs,1,1),device=self.device)), dim=1)

        euler_angles = torch.cat((-yaw/ang_scale,
                                 torch.zeros_like(yaw),
                                 torch.zeros_like(yaw)), dim=-1)
                                 
        R = euler_angles_to_matrix(euler_angles, 'ZYX')
        vec_2_target_p = torch.matmul(R,v_to_target_g)

        # battery_percent = self.physics.batt_man.battery_state/self.physics.batt_man.max_energy
        # battery_percent = battery_percent.reshape(self.num_envs,1)
        battery_percent = torch.zeros((self.num_envs,1))
        # scaled_propspeed = self.physics.thrust_man.thruster_list[0].omega/3000.0,
        scaled_propspeed = torch.zeros((self.num_envs,1))
        # scaled_elec_pow = self.physics.thrust_man.thruster_list[0].motor.electrical_power/100.0
        scaled_elec_pow = torch.zeros((self.num_envs,1))
        # scaled_prop_pitch = torch.unsqueeze(self.physics.thrust_man.thruster_list[0].prop.pitch, dim=1)/30.0
        scaled_prop_pitch = torch.zeros((self.num_envs,1))
        # print(battery_percent)
        self.obs_buf[:] = torch.cat((
                                     torch.unsqueeze(pos[:,2],dim=1), # Height
                                     ground_speed,
                                     air_speed,
                                     roll,
                                     pitch,
                                     yaw,
                                     battery_percent,
                                     vec_2_target_p[:,0:2, 0], #X,Y component of target heading vector in plane frame
                                     v_to_target_norm/1000,
                                     scaled_propspeed,
                                     scaled_elec_pow,
                                     scaled_prop_pitch
                                     ), dim=-1)
    
    def scale_and_store_actions(self, actions):
        # Action Scaling
        # actions[:,0:2] = actions[:,0:2] * self.action_scale_tail #0,1
        # actions[:,3:5] = actions[:,3:5] * self.action_scale_main #3,4

        actions[:,0] = actions[:,0] * self.action_scale_main #3,4
        actions[:,1] = actions[:,1] * self.action_scale_tail #0,1
        # actions[:,0] = actions[:,0] * self.action_scale_tail 
        # actions[:,1] = actions[:,1] * self.action_scale_tail #Scale Both The Same For AWing

        # actions[:,2] = (actions[:,2]+1)/2.0 #Left Trigger [0,1] (Only Positive Inputs?)
        # actions[:,3] = (actions[:,3]+1)/2.0 #Right Trigger [0,1] (Only Positive Inputs?)
        action_len = len(actions)
        self.actions_buf[0:action_len, ...] = actions


    def reset_idx(self, env_ids):
        # State Update moved to Post Physics Step
        # Randomization can happen only at reset time, since it can reset actor positions on GPU

        deg_2_rad = torch.pi/180.0
        roll_init = torch_rand_float(self.roll_range[0], self.roll_range[1], (len(env_ids), 1), device=self.device).squeeze()
        pitch_init = torch_rand_float(self.pitch_range[0], self.pitch_range[1], (len(env_ids), 1), device=self.device).squeeze()
        yaw_init = torch_rand_float(self.yaw_range[0], self.yaw_range[1], (len(env_ids), 1), device=self.device).squeeze()
        height_init = torch_rand_float(self.height_range[0], self.height_range[1], (len(env_ids), 1), device=self.device).squeeze()
        vinf_init = torch_rand_float(self.vinf_range[0], self.vinf_range[1], (len(env_ids), 1), device=self.device).squeeze()
        
        if(roll_init.dim() == 0):
            roll_init = torch.unsqueeze(roll_init, dim=-1)
            pitch_init = torch.unsqueeze(pitch_init, dim=-1)
            yaw_init = torch.unsqueeze(yaw_init, dim=-1)
            height_init = torch.unsqueeze(height_init, dim=-1)
            vinf_init = torch.unsqueeze(vinf_init, dim=-1)

        if(self.cfg["env"]["glider"]["wind_randomize"]):
            self.wind.randomize_settings(env_ids)

        wind_vec = self.wind.get_wind_vector(height_init, env_ids)
            
        new_quat = quat_from_euler_xyz(roll_init*deg_2_rad, pitch_init*deg_2_rad, yaw_init*deg_2_rad)
        

        euler_angles = torch.cat((torch.reshape(yaw_init,(len(env_ids),1))*deg_2_rad,
                                 torch.reshape(pitch_init,(len(env_ids),1))*deg_2_rad,
                                 torch.reshape(roll_init,(len(env_ids),1))*deg_2_rad), dim=-1)
                                 
        new_r_mat = euler_angles_to_matrix(euler_angles, 'ZYX')
        v_apparent = -new_r_mat[:,:,0]*torch.reshape(vinf_init, (len(env_ids),1))
        v_global = v_apparent + torch.squeeze(wind_vec)

        modified_initial_state = self.initial_glider_states.clone()
        modified_initial_state[env_ids, 2] = height_init
        modified_initial_state[env_ids, 3:7] = new_quat
        modified_initial_state[env_ids, 7:10] = v_global
        
        self.kinematic_states[env_ids, ...] = modified_initial_state[env_ids, ...]
    
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.goal_idx[env_ids] = 100
        self.reset_goals(env_ids)
        # self.physics.batt_man.reset_battery_states(env_ids)


    def reset_goals(self, env_ids):
        if(self.use_goal_list):
            # Update the goal index for the resetting environments. Roll back to zero if the index exceeds the number of goals
            self.goal_idx[env_ids] += 1
            self.goal_idx[env_ids] = torch.where(self.goal_idx[env_ids] >= self.num_goals, torch.zeros_like(self.goal_idx[env_ids]), self.goal_idx[env_ids] )
            # Create the target tensor
            self.target_pos[env_ids, :] = self.goal_list[self.goal_idx[env_ids]]
        else:
            self.target_pos[env_ids, :] = torch_rand_float(self.target_range[0], self.target_range[1], (len(env_ids), 2), device=self.device).squeeze()
        
        # Update the goal state tensors
        self.goal_states[env_ids, 0:2] = self.target_pos[env_ids, 0:2]
        self.goal_states[env_ids, 2] = -20.0
        self.goal_reached[env_ids] = 0



#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def calc_rewards_jit(
    # tensors
    glider_states,
    glider_states_prev,
    actions,
    episode_lengths,
    alphas,
    V_inf,
    obs,
    target_pos,
    # Dict
    rew_scales,
    # other
    max_episode_length,
    num_envs,
    target_radius
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Dict[str, float], int, int, float) -> Tuple[Tensor, Tensor, Tensor]

    # prepare quantities (TODO: return from obs ?)
    base_pos = glider_states[:,0:3]
    base_quat = glider_states[:, 3:7]
    base_lin_vel = quat_rotate_inverse(base_quat, glider_states[:, 7:10])
    base_ang_vel = quat_rotate_inverse(base_quat, glider_states[:, 10:13])

    # Energy Rewards
    mass = 1.0
    gravity = 9.81
    # height = obs[:,2]
    # ground_speed = obs[:,5]
    # air_speed = obs[:,6]
    # roll = obs[:,7]
    height = obs[:,0]
    ground_speed = obs[:,1]
    air_speed = obs[:,2]
    roll = obs[:,3]

    PE = mass*gravity*height 
    KE = 0.5*mass*(torch.abs(ground_speed)**2)  
    # reward_vinf = torch.squeeze(V_inf)/100.0
    # reward = reward_vinf
    # speed_reward = (ground_speed)**2 + (air_speed)**2 + (ground_speed-air_speed)**2
    speed_reward = (ground_speed)**2 + (air_speed)**2
    # thrust_reward = actions[:,3] - actions[:, 2] #Generate - Thrust
    # thrust_reward = -actions[:,2]
    vec_2_target = target_pos - base_pos[:,0:2]
    dist_2_target = torch.norm(vec_2_target, dim=1)
    unit_vec_2_target = vec_2_target/torch.unsqueeze(dist_2_target + 0.00001, dim=1).repeat(1,2)
    dist_reward = target_radius/(dist_2_target + 0.00001)
    
    # Don't go above 1 dist reward...
    reward_limit_idx = dist_reward>1.0
    dist_reward[reward_limit_idx] = 1.0
    

    punishment = torch.zeros((num_envs), device='cuda', dtype=torch.float)
    bank_punishment = roll**10

    diff_pos = glider_states[:,0:2] - glider_states_prev[:,0:2]
    target_reward  = torch.sum(diff_pos*unit_vec_2_target, dim=1)
    target_reward = torch.where(dist_2_target>target_radius, target_reward, 1.0)
    # reward = target_reward - bank_punishment

    glider_vel_uv = glider_states[:, 7:9]
    glider_vel = torch.norm(glider_vel_uv, dim=1)

    unit_heading_xy= glider_vel_uv/torch.unsqueeze(glider_vel + 0.00001, dim=1).repeat(1,2)

    vel_max = 30.0
    max_height = 10.0
    clipped_vel = torch.min(glider_vel,vel_max*torch.ones((num_envs), device='cuda', dtype=torch.float))
    clipped_height = torch.min(glider_states[:, 2],max_height*torch.ones((num_envs), device='cuda', dtype=torch.float))

    max_energy = (0.5*torch.pow(vel_max,2) + 9.8*max_height)
    mechanical_energy_scaled =  (0.5*torch.pow(clipped_vel,2) + 9.8*clipped_height)/max_energy
    # energy_rew = (mechanical_energy_scaled + obs[:,6])/2
    energy_rew = mechanical_energy_scaled

    hdg_rew_1 = 0.5*(sum(unit_heading_xy*unit_vec_2_target,dim=1)+torch.ones((num_envs), device='cuda', dtype=torch.float))     # 0     -> 1
    hdg_rew_2 = 0.5*(sum(unit_heading_xy*unit_vec_2_target,dim=1)-torch.ones((num_envs), device='cuda', dtype=torch.float))     # -1    -> 0
    hdg_rew_3 = 0.5*(sum(unit_heading_xy*unit_vec_2_target,dim=1))                                                              # -0.5  -> 0.5
    hdg_rew_4 = (sum(unit_heading_xy*unit_vec_2_target,dim=1))                                                                  # -1  -> 1
    hdg_rew_5 = 0.1*(sum(unit_heading_xy*unit_vec_2_target,dim=1))                                                              # -1  -> 1
    hdg_rew = hdg_rew_5
    # reward = hdg_rew + energy_rew
    reward = energy_rew*(torch.ones((num_envs), device='cuda', dtype=torch.float) + hdg_rew - energy_rew)
    # reward = hdg_rew_5*mechanical_energy_scaled

    # battery_reward = ((obs[:,6])**2)
    # reward *= obs[:,6]**2
    # reward = reward + battery_reward
    # reward += bank_punishment
    # reward = sum(unit_heading_xy*unit_vec_2_target,dim=1) + (0.5*torch.pow(clipped_vel,2) + 9.8*clipped_height)*0.01

    # reward = reward * 0.1
    # reward = dist_reward + (0.5*torch.pow(glider_vel,2) + 9.8*glider_states[:, 2])*0.00001



    # reward = PE + KE
    # reset agents if... 
    crash = glider_states[:,2] < 0.
    dead_batt = obs[:,6] < 0.
    alpha_fail = torch.abs(alphas) > 0.3
    alpha_reset = torch.where(alpha_fail, torch.ones_like(alpha_fail), torch.zeros_like(alpha_fail))
    alpha_reset = torch.squeeze(torch.sum(torch.sum(alpha_reset, dim=0), dim=0))>0

    
    # alpha_out_of_range = alpha > alpha_range
    # reset = reset | torch.any(torch.norm(contact_forces[:, knee_indices, :], dim=2) > 1., dim=1)
    time_out = episode_lengths >= max_episode_length - 1  # no terminal reward for time-outs
    
    # reward = torch.where(crash | alpha_reset, punishment, reward)
    reward = torch.where(crash, punishment, reward)
    # reset = crash | time_out | alpha_reset
    reset = crash | time_out | dead_batt | alpha_reset


    goal_reached=torch.zeros_like(reset)
    goal_reached = torch.where(dist_2_target>target_radius, goal_reached, 1)

    return reward, reset, goal_reached


@torch.jit.script #Add Alpha Observation
def calc_observations_jit(glider_states,
                                gravity_vec,
                                actions,
                                alphas,
                                height,
                                ground_speed,
                                air_speed,
                                roll,
                                pitch,
                                yaw
                                ):

    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor


    obs = torch.cat((height,
                     ground_speed,
                     air_speed,
                     roll,
                     pitch,
                     yaw
                     ), dim=-1)


    return obs
