import torch
from .torch_utils import * 

class Wind:
    def __init__(self, cfg, wind_strength_buf, wind_angle_buf):
        self.wind_strength = wind_strength_buf
        self.wind_angle = wind_angle_buf
        k999 = 2*6.907
        k99 = 2*4.595
        
        self.thickness = cfg['env']['glider']['wind_thickness']
        self.center = self.thickness/2
        self.c = k999/self.thickness

        self.wind_speed_min = cfg['env']['glider']['wind_speed_min']
        self.wind_speed_max = cfg['env']['glider']['wind_speed_max']
           

        
    def get_wind_vector(self, height, env_ids):
        speed = torch.squeeze(self.wind_strength[env_ids,:])
        w_speed = speed /(1+torch.exp(-self.c*(height - self.center)))
        w_speed = torch.reshape(w_speed,[len(height),1,1])
        # wind = torch.cat((w_speed, torch.zeros_like(w_speed), torch.zeros_like(w_speed)), dim=1)
        #[M,3,1]
        w_speed_x = w_speed * torch.cos(torch.reshape(self.wind_angle[env_ids,:],[len(height),1,1]))
        w_speed_y = w_speed * torch.sin(torch.reshape(self.wind_angle[env_ids,:],[len(height),1,1]))
        wind = torch.cat((w_speed_x, w_speed_y, torch.zeros_like(w_speed)), dim=1)
        return wind
    
    def randomize_settings(self, env_ids):
        deg_2_rad = torch.pi/180.0
        new_wind_angles = torch_rand_float(-torch.pi, torch.pi, (len(env_ids), 1), device='cuda:0')
        # print(self.wind_angle[env_ids,:].shape)
        # print(new_wind_angles.shape)
        self.wind_angle[env_ids,:] = new_wind_angles

        new_wind_strengths = torch_rand_float(self.wind_speed_min, self.wind_speed_max, (len(env_ids), 1), device='cuda:0')
        self.wind_strength[env_ids,:] = new_wind_strengths