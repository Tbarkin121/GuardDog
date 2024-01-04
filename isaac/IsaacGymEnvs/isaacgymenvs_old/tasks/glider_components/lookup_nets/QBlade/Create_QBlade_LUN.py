#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 17:30:08 2023

@author: tyler
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 17:00:47 2023

@author: Plutonium
"""
import pandas as pd
import torch
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator


from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import time
import pickle

dtype = torch.float
dtype_long = torch.long
device='cuda'
train = False
epochs = 200
# device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
#%%

# Coefficent of thrust, power, torque
# From Q Blade


# data_list=['Betz12p5_0p1_5_100_1000_n30_30.csv',
#             'Betz12p5_0p1_5_1000_30000_n30_30.csv',
#             'Betz12p5_5_30_100_1000_n30_30.csv',
#             'Betz12p5_5_30_1000_30000_n30_30.csv',
#             'BetzBlade12p5_SmallDataSet.csv',
#             'BetzBlade12p5_BigDataSet.csv']


# df_list = []
# for name in data_list:
#     df_list.append(pd.read_csv(name, delimiter=','))

# df = pd.concat(df_list)
# df = pd.read_csv('BetzBlade12p5_BigDataSet.csv', delimiter=',')
# df = pd.read_csv('BetzBlade12p5_0p5-30_200-40000_n30-30.csv', delimiter=',')
df = pd.read_csv('TSR10.csv', delimiter=',')
save_name = 'Resnet_TSR10'
# save_name = 'All_Loss'

# MSE
# L1
# SL1
# Huber
# All_Loss


#%%
class DatasetGen(Dataset):
    def __init__(self, save_name):
        self.save_name = save_name
        thrust = torch.tensor(df['THRUST[N]'].values, dtype=torch.float)
        torque = torch.tensor(df['TORQUE[Nm]'].values, dtype=torch.float)
        power = torch.tensor(df['POWER[kW]'].values, dtype=torch.float)
        diff_thrust_a = torch.diff(thrust[0:-1])
        diff_thrust_b = torch.diff(thrust[1:])
        diff_torque_a = torch.diff(torque[0:-1])
        diff_torque_b = torch.diff(torque[1:])
        diff_power_a = torch.diff(power[0:-1])
        diff_power_b = torch.diff(power[1:])
        lap_thrust = diff_thrust_b - diff_thrust_a
        lap_torque = diff_torque_b - diff_torque_a
        lap_power = diff_power_b - diff_power_a
        self.outlier_loc_thrust = torch.where(torch.abs(lap_thrust)>50)[0]
        self.outlier_loc_torque = torch.where(torch.abs(lap_torque)>50)[0]
        self.outlier_loc_power = torch.where(torch.abs(lap_power)>50)[0]

        # thrust_diff = torch.diff(thurst)
        # self.outlier_loc = torch.where(torch.abs(thrust_diff)>2**7)[0]
        
        self.outlier_mask = torch.ones((self.__len__()), device=device)
        self.outlier_mask[self.outlier_loc_thrust] = 0;
        self.outlier_mask[self.outlier_loc_torque] = 0;
        self.outlier_mask[self.outlier_loc_power] = 0;

        
        self.wind = torch.tensor(df['WIND[m/s]'].values, device=device, dtype=torch.float)
        self.wind_modifier = 1/torch.max(torch.abs(self.wind))
        self.wind_scaled = self.wind*self.wind_modifier
        self.wind_scaled = self.wind_scaled
        
        self.rpm = torch.tensor(df['ROT[rpm]'].values, device=device, dtype=torch.float)
        self.rpm_modifier = 1/torch.max(torch.abs(self.rpm))
        self.rpm_scaled = self.rpm*self.rpm_modifier
        self.rpm_scaled = self.rpm_scaled
        
        self.pitch = torch.tensor(df['PITCH[deg]'].values, device=device, dtype=torch.float)
        self.pitch_modifier = 1/torch.max(torch.abs(self.pitch))
        self.pitch_scaled = self.pitch*self.pitch_modifier
        self.pitch_scaled = self.pitch_scaled
        
        
        self.x_data_size = torch.unique(torch.tensor(df['WIND[m/s]'].values, device=device, dtype=torch.float)).shape[0]
        self.y_data_size = torch.unique(torch.tensor(df['ROT[rpm]'].values, device=device, dtype=torch.float)).shape[0]
        self.z_data_size = torch.unique(torch.tensor(df['PITCH[deg]'].values, device=device, dtype=torch.float)).shape[0]
        
        # self.features = torch.cat( (torch.unsqueeze(self.pitch_scaled, 1), torch.unsqueeze(self.tsr_scaled, 1)), 1)
        self.features = torch.cat( (torch.unsqueeze(self.wind_scaled, 1), torch.unsqueeze(self.rpm_scaled, 1), torch.unsqueeze(self.pitch_scaled, 1)), 1)
        
        self.thrust = torch.tensor(df['THRUST[N]'].values, device=device, dtype=torch.float) * self.outlier_mask
        self.thrust_modifier = 1/torch.max(torch.abs(self.thrust))
        self.thrust_scaled = self.thrust*self.thrust_modifier
        self.thrust_scaled = self.thrust_scaled
        
        self.torque = torch.tensor(df['TORQUE[Nm]'].values, device=device, dtype=torch.float) * self.outlier_mask
        self.torque_modifier = 1/torch.max(torch.abs(self.torque))
        self.torque_scaled = self.torque*self.torque_modifier
        self.torque_scaled = self.torque_scaled
        
        self.save_scales()
        # self.labels = torch.cat( (torch.unsqueeze(self.power_scaled, 1), torch.unsqueeze(self.thrust_scaled, 1), torch.unsqueeze(self.moment_scaled, 1)), 1)

        # self.labels = torch.cat( (torch.unsqueeze(self.thrust_scaled, 1), torch.unsqueeze(self.torque_scaled, 1), torch.unsqueeze(self.outlier_mask, 1)), 1)
        self.labels = torch.cat( (torch.unsqueeze(self.thrust_scaled, 1), torch.unsqueeze(self.torque_scaled, 1), torch.unsqueeze(self.outlier_mask, 1)), 1)
        
    def __len__(self):
        return len(df)
    
    def __getitem__(self, idx):
        return self.features[idx,:], self.labels[idx,:]
    
    def save_scales(self):
        scale_dict = {'wind_modifier':self.wind_modifier.detach().cpu().numpy(),
                      'rpm_modifier':self.rpm_modifier.detach().cpu().numpy(),
                      'pitch_modifier':self.pitch_modifier.detach().cpu().numpy(),
                      'thrust_modifier':self.thrust_modifier.detach().cpu().numpy(),
                      'torque_modifier':self.torque_modifier.detach().cpu().numpy()}
        with open('{}.txt'.format(self.save_name), 'wb') as f:  # Python 3: open(..., 'wb')
            pickle.dump(scale_dict, f)
            
    def read_scales(self):
        with open('{}.txt'.format(self.save_name), 'rb') as f:  # Python 3: open(..., 'rb')
            scale_dict = pickle.load(f)
            self.wind_modifier = torch.tensor(scale_dict['wind_modifier'], device=device)
            self.rpm_modifier = torch.tensor(scale_dict['rpm_modifier'], device=device)
            self.pitch_modifier = torch.tensor(scale_dict['pitch_modifier'], device=device)
            self.thrust_modifier = torch.tensor(scale_dict['thrust_modifier'], device=device)
            self.torque_modifier = torch.tensor(scale_dict['torque_modifier'], device=device)
            # print(scale_dict)

data_gen = DatasetGen(save_name)
data_gen.__getitem__(0)
# plt.close('all')
# fig, ax = plt.subplots()
# ax.plot(data_gen.thrust.cpu().numpy())
# ax.grid(True)

# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# ax.set_xlabel('pitch')
# ax.set_ylabel('rpm')
# ax.set_zlabel('thrust')

#%%

import torch.nn as nn
import torch.nn.functional as F

# Get cpu or gpu device for training.
print(f"Using {device} device")

# Define model
class QBlade_nn(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(3, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 2)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
    
# Define model
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

net = QBlade_nn().to(device)
print(net)
        

mse_loss_fn = nn.MSELoss(reduction='none')
l1_loss_fn = nn.L1Loss(reduction='none')
sl1_loss_fn= nn.SmoothL1Loss(reduction='none')
huber_loss_fn = nn.HuberLoss(reduction='none')

optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

#%%
data_gen = DatasetGen(save_name)
train_dataloader = DataLoader(data_gen, batch_size = int(data_gen.__len__()/100), shuffle=True)
loss_data = []

if(train==True):
    for epoch in range(epochs):
        print(epoch)
        running_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):    
           # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # training_labels = torch.unsqueeze(labels[:, 0], dim=1)
            training_labels = labels[:, 0:2]
            mask = torch.unsqueeze(labels[:, 2], dim=1)
            # zero the parameter gradients
            optimizer.zero_grad()
    
            outputs = net(inputs)
            
            # loss = torch.sum(mse_loss_fn(outputs, training_labels)*maskBetzBlade12p5_BigDataSet)/torch.sum(mask)
            # loss = torch.sum(l1_loss_fn(outputs, training_labels)*mask)/torch.sum(mask)
            # loss = torch.sum(sl1_loss_fn(outputs, training_labels)*mask)/torch.sum(mask)
            # loss = torch.sum(huber_loss_fn(outputs, training_labels)*maBetzBlade12p5_BigDataSetsk)/torch.sum(mask)

            loss1 = torch.sum(mse_loss_fn(outputs, training_labels)*mask)/torch.sum(mask)
            loss2 = torch.sum(l1_loss_fn(outputs, training_labels)*mask)/torch.sum(mask)
            loss3 = torch.sum(sl1_loss_fn(outputs, training_labels)*mask)/torch.sum(mask)
            loss4 = torch.sum(huber_loss_fn(outputs, training_labels)*mask)/torch.sum(mask)
            loss = (loss1+loss2+loss3+loss4)/4
            
            
            loss.backward()
            optimizer.step()
    
            # print statistics
            running_loss += loss.item()
            
            # print(loss)
            loss_data.append(loss.detach().cpu())
        print(running_loss/(data_gen.__len__()/10))
        
        
    print("Model's state_dict:")
    for param_tensor in net.state_dict():
        print(param_tensor, "\t", net.state_dict()[param_tensor].size())
    print('Saving Model')
    torch.save(net, '{}.pth'.format(save_name))

#%%
plt.close('all')
fig, ax = plt.subplots()
ax.plot(torch.log(torch.tensor(loss_data)))

print('Finished Training')


#%%
net = QBlade_nn()
net = torch.load('{}.pth'.format(save_name))
net.eval()



#%%


wind_3d = torch.reshape(data_gen.wind, (data_gen.x_data_size, data_gen.y_data_size, data_gen.z_data_size) )
pitch_3d = torch.reshape(data_gen.pitch, (data_gen.x_data_size, data_gen.y_data_size, data_gen.z_data_size) )
rpm_3d = torch.reshape(data_gen.rpm, (data_gen.x_data_size, data_gen.y_data_size, data_gen.z_data_size) )
thrust_3d = torch.reshape(data_gen.thrust, (data_gen.x_data_size, data_gen.y_data_size, data_gen.z_data_size) )
torque_3d = torch.reshape(data_gen.torque, (data_gen.x_data_size, data_gen.y_data_size, data_gen.z_data_size) )

# for i in range(data_gen.x_data_size):
#     fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
#     ax.set_xlabel('pitch')
#     ax.set_ylabel('rpm')
#     ax.set_zlabel('thrust')
#     ax.contour(pitch_3d[i,:,:].cpu(), rpm_3d[i,:,:].cpu(), thrust_3d[i,:,:].cpu(), 100, cmap=cm.Pastel2)


i = 10
y_max = 40000
y_min = -10
z_max = 1
z_min = -1


in_wind = torch.reshape(wind_3d, (data_gen.x_data_size*data_gen.y_data_size*data_gen.z_data_size,1))*data_gen.wind_modifier
in_rpm = torch.reshape(rpm_3d, (data_gen.x_data_size*data_gen.y_data_size*data_gen.z_data_size,1))*data_gen.rpm_modifier
in_pitch = torch.reshape(pitch_3d, (data_gen.x_data_size*data_gen.y_data_size*data_gen.z_data_size,1))*data_gen.pitch_modifier
in_data = torch.cat((in_wind, in_rpm, in_pitch), dim=1)
del in_wind, in_rpm, in_pitch
out_data = net(in_data)
out_data_3d = torch.reshape(out_data, (data_gen.x_data_size, data_gen.y_data_size, data_gen.z_data_size, 2))
out_thrust_3d = out_data_3d[:,:,:,0]/data_gen.thrust_modifier
out_torque_3d = out_data_3d[:,:,:,1]/data_gen.torque_modifier


plt.close('all')

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.set_xlabel('pitch')
ax.set_ylabel('rpm')
ax.set_zlabel('thrust')
# ax.contour(pitch_3d[i,:,:].cpu(), rpm_3d[i,:,:].cpu(), thrust_3d[i,:,:].cpu(), 100)
ax.plot_surface(pitch_3d[i,:,:].cpu(), rpm_3d[i,:,:].cpu(), thrust_3d[i,:,:].cpu())
# ax.set_ylim(y_min,y_max)
# ax.set_zlim(z_min,z_max)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.set_xlabel('pitch')
ax.set_ylabel('rpm')
ax.set_zlabel('thrust')
# ax.contour(pitch_3d[i,:,:].detach().cpu(), rpm_3d[i,:,:].detach().cpu(), out_thrust.detach().cpu(), 100)
ax.plot_surface(pitch_3d[i,:,:].cpu(), rpm_3d[i,:,:].cpu(), out_thrust_3d[i,:,:].detach().cpu())
# ax.set_ylim(y_min,y_max)
# ax.set_zlim(z_min,z_max)


y_max = 40000
y_min = -10
z_max = 1
z_min = -20
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.set_xlabel('pitch')
ax.set_ylabel('rpm')
ax.set_zlabel('torque')
# ax.contour(pitch_3d[i,:,:].cpu(), rpm_3d[i,:,:].cpu(), torque_3d[i,:,:].cpu(), torch.linspace(-0.5,0.5,100))
ax.plot_surface(pitch_3d[i,:,:].cpu(), rpm_3d[i,:,:].cpu(), torque_3d[i,:,:].cpu())
# ax.set_ylim(y_min,y_max)
# ax.set_zlim(z_min,z_max)

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.set_xlabel('pitch')
ax.set_ylabel('rpm')
ax.set_zlabel('torque')
# ax.contour(pitch_3d[i,:,:].detach().cpu(), rpm_3d[i,:,:].detach().cpu(), out_torque.detach().cpu(), torch.linspace(-0.5,0.5,100))
ax.plot_surface(pitch_3d[i,:,:].cpu(), rpm_3d[i,:,:].cpu(), out_torque_3d[i,:,:].detach().cpu())
# ax.set_ylim(y_min,y_max)
# ax.set_zlim(z_min,z_max)


#%%
print(torch.sum(mse_loss_fn(out_data, data_gen.labels[:,0:2])))

for name, param in net.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")