# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 01:37:05 2023

@author: Plutonium
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 09:58:15 2023

@author: Plutonium
"""


import torch
import matplotlib.pyplot as plt
import numpy as np
import time
from torchviz import make_dot

torch.set_default_device('cuda')
#%%
class Pendulum:
    def __init__(self):
        print('Init Arm')
        self.num_segments = 1
        self.joint_angles = torch.zeros(self.num_segments, requires_grad=True)
        self.angle_offset = 3.1415/2 # So gravity is down 
        self.joint_velocity = torch.zeros(self.num_segments, requires_grad=True)
        self.joint_acceleration = torch.zeros(self.num_segments, requires_grad=True)

        self.link_lengths =  torch.ones(self.num_segments, requires_grad=False)/self.num_segments        
        self.link_mass = torch.ones(self.num_segments, requires_grad=False)
        
        self.xs = torch.zeros(self.num_segments+1, requires_grad=False)
        self.ys = torch.zeros(self.num_segments+1, requires_grad=False)
        self.x_targ=torch.tensor(-0.33, requires_grad=False)
        self.y_targ=torch.tensor(0.44, requires_grad=False)
        
        self.I = (1/3)*self.link_mass[0]*self.link_lengths[0]
        
        
        plt.close('all')
        xp = torch.cat((torch.tensor([0.0]), self.xs)).detach().cpu().numpy()
        yp = torch.cat((torch.tensor([0.0]), self.ys)).detach().cpu().numpy()
        
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.grid(True)
        self.ax.set_xlim([-2, 2])
        self.ax.set_ylim([-2, 2])
        self.line1, = self.ax.plot(xp, yp, 'r-') # Returns a tuple of line objects, thus the comma
        self.line2, = self.ax.plot(self.x_targ.detach().cpu().numpy(),self.y_targ.detach().cpu().numpy(), 'o') # Returns a tuple of line objects, thus the comma
    
    def forward_kinematics(self):
        self.xs = torch.zeros(self.num_segments+1, requires_grad=False)
        self.ys = torch.zeros(self.num_segments+1, requires_grad=False)
        for s in range(1, self.num_segments+1):
            self.xs[s] = self.xs[s-1] + self.link_lengths[s-1]*torch.cos(torch.sum(self.joint_angles[0:s]))
            self.ys[s] = self.ys[s-1] + self.link_lengths[s-1]*torch.sin(torch.sum(self.joint_angles[0:s]))
            
    def ComputeLagrange(self):       
        self.T = 0.5*self.I*self.joint_velocity**2                             # Kinetic Energy
        
        h = self.link_lengths/2 * (1-torch.cos(self.joint_angles+self.angle_offset))
        self.U = self.link_mass*9.81*h;                                        # Potential Energy
        
        self.F = self.joint_velocity * 0.01                                    # Friction Energy Lost
        
        self.L = self.T-self.U                                                 # Legrangeian
        
        
    def EulerLagrange(self, delta_t): #(d/dt)(dL/dthetadot)-(dL/dtheta) = tau 
        # self.tau = self.I*self.joint_acceleration + torch.sin(self.joint_angles)*self.link_mass*9.81*self.link_lengths/2

        # Compute current Lagrangian
        
        self.ComputeLagrange()
        
        # Save current values of dL/dthetadot
        # current_dL_dthetadot = self.I * self.joint_velocity
        self.ClearGrads()

        self.L.backward()

        current_dL_dthetadot = self.joint_velocity.grad.clone()
        manual_dL_dthetadot = self.I*self.joint_velocity
        # print("Automatic differentiation:", current_dL_dthetadot)
        # print("Manual calculation:", manual_dL_dthetadot)
    
        # Update state to t + delta_t
        self.UpdateState(delta_t)
    
        # Compute new Lagrangian at t + delta_t
        self.ComputeLagrange()

        # New value of dL/dthetadot at t + delta_t
        self.ClearGrads()
        self.L.backward()
        # new_dL_dthetadot = self.I * self.joint_velocity
        new_dL_dthetadot = self.joint_velocity.grad.clone()
    
        # Numerical approximation of time derivative
        dL_dthetadot_dt = (new_dL_dthetadot - current_dL_dthetadot) / delta_t
        manual_dL_dthetadot_dt = self.I*self.joint_acceleration
        # print("Automatic differentiation:", dL_dthetadot_dt)
        # print("Manual calculation:", manual_dL_dthetadot_dt)
    
        # dL/dtheta
        dL_dtheta = self.joint_angles.grad
        manual_dL_dtheta = -self.link_mass*9.81*self.link_lengths/2*torch.sin(self.joint_angles)
        # print("Automatic differentiation:", dL_dtheta)
        # print("Manual calculation:", manual_dL_dtheta)
    
        # Euler-Lagrange equation
        self.tau = dL_dthetadot_dt - dL_dtheta
        friction_torque = self.joint_velocity * 0.1
        input_tau = 3
        total_tau = input_tau-self.tau-friction_torque
        self.joint_acceleration = total_tau/self.I
        
        
    def UpdateState(self, delta_t):
        # Update joint angles and velocities using a simple Euler integration
        with torch.no_grad():
            # Update joint angles and velocities using a simple Euler integration
            self.joint_angles += self.joint_velocity * delta_t
            self.joint_velocity += self.joint_acceleration * delta_t

                     
    def ClearGrads(self):
        if self.joint_angles.grad is not None:
            self.joint_angles.grad = None
        if self.joint_velocity.grad is not None:
            self.joint_velocity.grad = None
        if self.joint_acceleration.grad is not None:
            self.joint_acceleration.grad = None
            
    def Plot(self):
        self.forward_kinematics()
        xp = torch.cat((torch.tensor([0.0]), self.xs)).detach().cpu().numpy()
        yp = torch.cat((torch.tensor([0.0]), self.ys)).detach().cpu().numpy()
        self.line1.set_xdata(xp)
        self.line1.set_ydata(yp)
        self.line2.set_xdata(self.x_targ.detach().cpu().numpy())
        self.line2.set_ydata(self.y_targ.detach().cpu().numpy())
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        # plt.connect('motion_notify_event', self.mouse_move)
        
        
    def mouse_move(self, event):
        x, y = event.xdata, event.ydata
        if(x and y):
            self.x_targ = torch.tensor(x, dtype=torch.float, requires_grad=False)
            self.y_targ = torch.tensor(y, dtype=torch.float, requires_grad=False)
        

        
#%%        
env = Pendulum()
env.Plot()
#%%
for _ in range(10000):
    t1 = time.perf_counter()
    env.EulerLagrange(0.01)
    if ( (_ % 10) == 0):
        env.Plot()
        
    t2 = time.perf_counter()
    print(t2-t1)
# print(env.T)
# print(env.U)
# print(env.L)
# print(env.tau)


