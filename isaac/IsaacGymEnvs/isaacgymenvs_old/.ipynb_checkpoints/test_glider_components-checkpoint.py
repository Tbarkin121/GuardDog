import isaacgym
import torch
import math
from tasks.glider_components.thruster_component import Thruster

import numpy as np
import matplotlib.pyplot as plt

#%%

num_envs = 10
dt = 0.02
t_mass = torch.tensor([1])
t_ori = torch.tensor([-1,0,0])
t_pos = torch.tensor([0,0,0])
t = Thruster(num_envs, dt, t_mass, t_ori, t_pos)
# print(t.prop.LUN.wind_modifier)
# print(t.prop.LUN.network.eval())
# print(t.prop.LUN.wind_modifier)
# print(t.prop.LUN.rpm_modifier)
# print(t.prop.LUN.pitch_modifier)
# print(t.prop.LUN.thrust_modifier)
# print(t.prop.LUN.torque_modifier)


# Creating a state observation mesh

#%%
wind_vec = torch.arange(0.5, 30, 1, dtype=torch.float, device="cuda", requires_grad = False)
rpm_vec = torch.arange(-40000, 80000, 1000, dtype=torch.float, device="cuda", requires_grad = False)
pitch_vec = torch.arange(-50, 50, 5, dtype=torch.float, device="cuda", requires_grad = False)

wind_3d, rpm_3d, pitch_3d = torch.meshgrid(wind_vec, rpm_vec, pitch_vec)

data_pts_cnt = wind_3d.shape[0]*wind_3d.shape[1]*wind_3d.shape[2]
wind_in = torch.reshape(wind_3d, (data_pts_cnt,1))
rpm_in = torch.reshape(rpm_3d, (data_pts_cnt,1))
pitch_in = torch.reshape(pitch_3d, (data_pts_cnt,1))
net_input = torch.cat((wind_in, rpm_in, pitch_in), dim=1)


thrust, torque = t.prop.LUN.get_thrust_and_torque(net_input)
thrust_3d = torch.reshape(thrust, (wind_3d.shape[0], wind_3d.shape[1], wind_3d.shape[2]))
torque_3d = torch.reshape(torque, (wind_3d.shape[0], wind_3d.shape[1], wind_3d.shape[2]))

plt.close('all')
i = 15 #Setting Wind Speed Constant
fig1, ax1 = plt.subplots(subplot_kw={"projection": "3d"})
fig2, ax2 = plt.subplots(subplot_kw={"projection": "3d"})
ax1.plot_surface(rpm_3d[i,:,:].detach().cpu(), pitch_3d[i,:,:].detach().cpu(), thrust_3d[i,:,:].detach().cpu())
ax2.plot_surface(rpm_3d[i,:,:].detach().cpu(), pitch_3d[i,:,:].detach().cpu(), torque_3d[i,:,:].detach().cpu())
# for i in range(30):
    # ax1.plot_surface(rpm_3d[i,:,:].detach().cpu(), pitch_3d[i,:,:].detach().cpu(), thrust_3d[i,:,:].detach().cpu())
    # ax2.plot_surface(rpm_3d[i,:,:].detach().cpu(), pitch_3d[i,:,:].detach().cpu(), torque_3d[i,:,:].detach().cpu())


# print(thrust.shape)