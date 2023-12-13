#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 17:06:42 2023

@author: tyler
"""

import torch

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

device = "cuda"
#%%

# Define model
import torch.nn as nn
import torch.nn.functional as F

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

    
#%%

print('Loading Model')
new_net = QBlade_nn()
new_net = torch.load('QBlade_LUN.pth')
new_net.eval()


#%%

class LUN():
    def __init__(self):

        model_pth = 'QBlade_LUN.pth'
        self.net = QBlade_nn()
        self.net = torch.load(model_pth)
        # self.net.eval()
        
        self.wind_modifier = 0.0333
        self.rpm_modifier = 2.5000e-05
        self.pitch_modifier = 0.0333
        self.thrust_modifier = 0.0081
        self.torque_modifier = 0.1772
        
    def get_thrust_and_torque(self, i):
        i_scaled = i * torch.tensor((self.wind_modifier, self.rpm_modifier, self.pitch_modifier), device=device )
        # print(',,,,,,,,,,,,,')
        # print(i_scaled)
        
        i = torch.clip(i_scaled, -1., 1.)
        o = self.net(i)
        thrust = o[:,0] / self.thrust_modifier
        torque = o[:,1] / self.torque_modifier
        
        # print(o)
        # thrust = torch.ones_like(thrust, device=device)
        # torque = torch.ones_like(torque, device=device)
        
        return thrust, torque
    

#%%
class Motor():
    # The Thruster handles the propeller and motor sub systems
    # The propeller uses a look up net to determine thrust and moment coefficents
    # The motor can add torque to the system
    # The propeller and motor torques only effect the propeller state and currently do not act on the plane
    def __init__(self, m):
        self.m = m
        self.omega = torch.zeros((m), device=device) #Rads Per Second
        self.I = 1.81e-7 
        
        self.motor_scale = 1.0;
        
        
        # self.motor_speed_constant = 5.58e-3 * 2 * torch.pi / 60
        self.torque_constant = 0.00631      # Nm/A
        self.speed_constant = 1510 * 2 * torch.pi / 60   # rad/s/V


        self.motor_resistance = 0.5
        self.motor_supply_voltage = 12
        self.max_current = self.motor_supply_voltage/self.motor_resistance
        self.max_torque = self.max_current*self.torque_constant
        self.motor_scale = 1.0;
    
    
    def get_torque(self, a, omega):
        self.omega = omega
        # Converts from action range [-1, 1] to motor torque range [?, ?]
        self.requested_torque = a*self.motor_scale
        
        self.emf_voltage = torch.abs(self.omega)/self.speed_constant
        self.available_voltage = self.motor_supply_voltage - self.emf_voltage
        self.available_current = self.available_voltage/self.motor_resistance
        self.available_torque = self.available_current*self.torque_constant
        self.torque_sign = torch.sign(self.requested_torque)
        
        # It is ok to clip like this because we only expect the motor to spin in one direction. There will be no limits on torques in the negitive direction
        self.torque_delivered = torch.clip(self.requested_torque, -self.max_torque*torch.ones((self.m), device=device), self.available_torque)
        # self.torque_delivered = torch.max(torch.min(torch.abs(self.requested_torque), torch.abs(self.available_torque)), torch.zeros((self.m), device=device))        
        
        return self.torque_delivered*self.torque_sign
    
#%%
class Propeller():
    def __init__(self, m):
        self.prop_LUN = LUN()

        self.state = torch.zeros([m,3], device=device)
        self.wind = self.state.view(m,3)[:,0]
        self.rpm = self.state.view(m,3)[:,1]
        self.pitch = self.state.view(m,3)[:,2]
        
        self.pitch_old = torch.zeros((m), device=device) # Previous time step pitch value
        
        
        self.pitch_scale = 1.0;
        self.pitch_offset = 0.0;
        self.radius = 0.12; # 5 inch radius
        self.area = 3.1415*self.radius**2;
        self.I = 3.5e-6;
        self.rho = 1.225;
        
        self.pitch_lp = 0.95
        
    def update_pitch(self, p):
        # Low Pass Filter Pitch Update
        # Takes an action [-1, 1]
        # Unscales the action
        # Low pass filter updates pitch state variable
        # unscaled_pitch = p*self.pitch_scale + self.pitch_offset
        
        self.pitch[:] = self.pitch_lp * self.pitch_old + (1-self.pitch_lp) * p
        self.pitch_old[:] = p
        
    def get_thrust_and_torque(self, omega, V):
        # Get the Force and Moment coefficents
        # Input omega is the rad/s
        # Input V is the wind velocity tensor
        self.wind[:] = V
        self.rpm[:] = omega/(2*torch.pi)*60 #Rad Per Second to RPM
        

        # print('~~~~~~~~~~')
        motor = Motor()
        # print(self.state)
        thrust, torque = self.prop_LUN.get_thrust_and_torque(self.state)

        return thrust, torque
        
#%%
class Thruster():
    # The Thruster handles the propeller and motor sub systems
    # The propeller uses a look up net to determine thrust and moment coefficents
    # The motor can add torque to the system
    # The propeller and motor torques only effect the propeller state and currently do not act on the plane
    def __init__(self, m):
        self.dt = 0.02;
        self.omega = torch.zeros((m), device=device) #Rads Per Second
        self.prop = Propeller(m)
        self.motor = Motor(m)
        self.I = self.prop.I + self.motor.I
        self.thruster_ori = torch.tensor([-1.0, 0.0, 0.0], device=device).repeat(m,1)
        self.damping_constant = 3.13e-6 * 2 * torch.pi

    def update(self, action, obs):
        # Convert Actions to Units
        self.prop.update_pitch(action[:,0])
        
        self.motor_moment = self.motor.get_torque(action[:,1], self.omega)

        wind_vel_apparent = obs
        # # Matmul (m,3,1) (m,1,3) -> (m,1,1) 
        V = -torch.sum(torch.mul(wind_vel_apparent, self.thruster_ori), dim=1)
        
        self.prop_thrust, self.prop_torque = self.prop.get_thrust_and_torque(self.omega, V)
        

        prop_thrust_vec = torch.mul(torch.unsqueeze(self.prop_thrust, dim=1), self.thruster_ori)
        
        # print(self.motor_moment)
        # total_moment = self.prop_torque + self.motor_moment
        # total_moment = self.motor_moment
        total_moment = self.prop_torque
        moment_vec = torch.mul(torch.unsqueeze(total_moment, dim=1), self.thruster_ori)
        print(total_moment)
        #Update prop state (omega)
        alpha = (total_moment  - self.omega * self.damping_constant)/ self.I
        self.omega += alpha*self.dt
        self.omega = torch.clamp(self.omega, torch.zeros_like(self.omega), torch.ones_like(self.omega)*4188)
        return prop_thrust_vec, moment_vec
    
motor = Motor()
    
#%%

motor = Motor(1)

torque_req = torch.tensor([-0.15], device=device)
omega = torch.tensor([1000], device=device) 

torque_del = motor.get_torque(torque_req, omega)


print(torque_req)
print(omega)
print(torque_del)