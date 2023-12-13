import os 
import __main__
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from .base_lun import LUN

# Define model
# class QBlade_nn(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.linear_relu_stack = nn.Sequential(
#             nn.Linear(3, 1024),
#             nn.LeakyReLU(),
#             nn.Linear(1024, 1024),
#             nn.LeakyReLU(),
#             nn.Linear(1024, 1024),
#             nn.LeakyReLU(),
#             nn.Linear(1024, 2)
#         )

#     def forward(self, x):
#         logits = self.linear_relu_stack(x)
#         return logits

class QBlade_nn(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(3, 1024),
            nn.LeakyReLU()
            )
        self.layer2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.LeakyReLU()
            )
        self.layer3 = nn.Sequential(
            nn.Linear(1027, 1024),
            nn.LeakyReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Linear(1024, 2)
        )

    def forward(self, x):
        residual = x
        x = self.layer1(x)
        x = self.layer2(x)
        x = torch.cat((x,residual), dim=1)
        x = self.layer3(x)
        logits = self.layer4(x)
        return logits

setattr(__main__, "QBlade_nn", QBlade_nn)

class QBlade_LUN(LUN):
    def __init__(self, device):
        self.device = device
        super().__init__('tasks/glider_components/lookup_nets/QBlade/Resnet_TSR10', self.device)
        net_arch = QBlade_nn().to(self.device)
        self.load_net(net_arch)
        self.read_scales()

    def read_scales(self):
        with open('{}.txt'.format(self.net_name), 'rb') as f:  # Python 3: open(..., 'rb')
            scale_dict = pickle.load(f)
            self.wind_modifier = torch.tensor(scale_dict['wind_modifier'], device=self.device)
            self.rpm_modifier = torch.tensor(scale_dict['rpm_modifier'], device=self.device)
            self.pitch_modifier = torch.tensor(scale_dict['pitch_modifier'], device=self.device)
            self.thrust_modifier = torch.tensor(scale_dict['thrust_modifier'], device=self.device)
            self.torque_modifier = torch.tensor(scale_dict['torque_modifier'], device=self.device)

    def get_thrust_and_torque(self, i):
        i_scaled = i * torch.tensor((self.wind_modifier, self.rpm_modifier, self.pitch_modifier), device=self.device )
        
        i_scaled[:,0] = torch.clip(i_scaled[:,0], 0.01, 1.)
        i_scaled[:,1] = torch.clip(i_scaled[:,1], 0., 1.)
        i_scaled[:,2] = torch.clip(i_scaled[:,2], -1., 1.)

        o = self.network(i_scaled)
        thrust = o[:,0] / self.thrust_modifier
        torque = o[:,1] / self.torque_modifier
                
        return torch.unsqueeze(thrust, dim=1), torch.unsqueeze(torque, dim=1)


