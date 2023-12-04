# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 01:37:05 2023

@author: Plutonium
"""
    

import torch
import matplotlib.pyplot as plt
import numpy as np
import time
import torchviz

torch.set_default_device('cpu')
#%%

class PlanarArm:
    def __init__(self, num_segments):
        print('Init Arm')
        self.num_segments = num_segments
        self.joint_angles = torch.zeros(num_segments, requires_grad=True)
        with torch.no_grad():
            self.joint_angles[0] = torch.pi/4
            self.joint_angles[1] = torch.pi/4
        
        self.joint_lengths =  torch.ones(num_segments, requires_grad=False)*1.0
        with torch.no_grad():
            self.joint_lengths[1] = self.joint_lengths[1]/2
            
        self.xs = torch.zeros(num_segments+1, requires_grad=False)
        self.ys = torch.zeros(num_segments+1, requires_grad=False)
        self.x_targ=torch.tensor(-0.33, requires_grad=False)
        self.y_targ=torch.tensor(0.44, requires_grad=False)
        
        self.weights = torch.ones([num_segments,1])
        self.weights[0] = 0
                
        
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
        self.line3, = self.ax.plot([0,0],[0,0], 'm-') # Returns a tuple of line objects, thus the comma
        
    def forward_kinematics(self):
        self.xs = torch.zeros(self.num_segments+1, requires_grad=False)
        self.ys = torch.zeros(self.num_segments+1, requires_grad=False)
        for s in range(1, self.num_segments+1):
            self.xs[s] = self.xs[s-1] + self.joint_lengths[s-1]*torch.cos(torch.sum(self.joint_angles[0:s]))
            self.ys[s] = self.ys[s-1] + self.joint_lengths[s-1]*torch.sin(torch.sum(self.joint_angles[0:s]))
            
                
        
    def get_residual(self):
        self.dx = self.xs[-1] - self.x_targ
        self.dy = self.ys[-1] - self.y_targ
        # error = torch.sqrt(dx**2 + dy**2)
        
    def compute_jacobian(self):
        # Compute forward kinematics
        self.forward_kinematics()
        self.get_residual()


        if self.joint_angles.grad is not None:
            self.joint_angles.grad = None

        self.dx.backward()
        self.jacobian_x = self.joint_angles.grad.clone()
        

        # Zero out the gradients before computing the next one        
        # self.joint_angles.grad = None
        if self.joint_angles.grad is not None:
            self.joint_angles.grad = None

        self.dy.backward()
        self.jacobian_y = self.joint_angles.grad.clone()
        
        self.J = torch.stack((env.jacobian_x, env.jacobian_y))
        
        # Manual 2 segment jacobian calc (Checked out vs torch, it matches)
        # self.test_J = torch.zeros(2,2)
        # self.test_J[0,0]= - self.joint_lengths[0]*torch.sin(self.joint_angles[0]) - self.joint_lengths[1]*torch.sin(self.joint_angles[0] + self.joint_angles[1])
        # self.test_J[0,1]= - self.joint_lengths[1]*torch.sin(self.joint_angles[0] + self.joint_angles[1])
        # self.test_J[1,0]= self.joint_lengths[0]*torch.cos(self.joint_angles[0]) + self.joint_lengths[1]*torch.cos(self.joint_angles[0] + self.joint_angles[1])       
        # self.test_J[1,1]= self.joint_lengths[1]*torch.cos(self.joint_angles[0] + self.joint_angles[1])
        
    


        
    def update_angles(self, dtheta):
        with torch.no_grad():
            # self.joint_angles -= dtheta.view(-1)
            self.joint_angles[1] -= dtheta[1][0]
        
    def plot(self):
        # self.forward_kinematics()
        xp = torch.cat((torch.tensor([0.0]), self.xs)).detach().cpu().numpy()
        yp = torch.cat((torch.tensor([0.0]), self.ys)).detach().cpu().numpy()
        self.line1.set_xdata(xp)
        self.line1.set_ydata(yp)
        self.line2.set_xdata(self.x_targ.detach().cpu().numpy())
        self.line2.set_ydata(self.y_targ.detach().cpu().numpy())
        
        self.line3.set_xdata([xp[-1], xp[-1] + 0.1*self.EndEffector_F[0].detach().cpu().numpy()[0]])
        self.line3.set_ydata([yp[-1], yp[-1] + 0.1*self.EndEffector_F[1].detach().cpu().numpy()[0]])
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.connect('motion_notify_event', self.mouse_move)
     


        
    def control(self):
        m = 2
        n = self.num_segments
        gamma = .5
        
        self.compute_jacobian()
        
        # self.J_inv = torch.linalg.pinv(self.J)
        # self.delta_theta = torch.matmul(self.J_inv, torch.tensor([self.dx, self.dy]))
        # self.update_angles(self.delta_theta)
        
        JJT = torch.matmul(self.J + torch.eye(self.J.shape[0])*0.00001, self.J.permute([1,0]) + torch.eye(self.J.shape[0])*0.00001)
        Im = torch.eye(m)
        R = torch.stack((env.dx, env.dy)).view(-1,1)
        M1 = torch.linalg.solve(JJT, self.J)
        M2 = torch.linalg.solve(JJT+gamma**2*Im, R)
        In = torch.eye(n)
        Zp = In - torch.matmul(self.J.permute([1,0]), M1)
        DeltaThetaPrimary = torch.matmul(self.J.permute([1,0]), M2)
        DeltaThetaSecondary = torch.matmul(Zp, self.joint_angles.view(-1,1) * self.weights)
        DeltaTheta = DeltaThetaPrimary + DeltaThetaSecondary        
        self.update_angles(DeltaTheta)
        
    def mouse_move(self, event):
        x, y = event.xdata, event.ydata
        if(x and y):
            self.x_targ = torch.tensor(x, dtype=torch.float, requires_grad=False)
            self.y_targ = torch.tensor(y, dtype=torch.float, requires_grad=False)
            
    def endeffector_forces(self):

        with torch.no_grad():
            self.J_inv = torch.linalg.pinv(self.J.T + torch.eye(self.J.shape[0])*0.00001)
            # self.J_inv = torch.linalg.pinv(self.J.T)
            
            # Matches the numbers from : 
            # https://studywolf.wordpress.com/2013/09/02/robot-control-jacobians-velocity-and-force/
            end_effector_force = torch.tensor([[1.0],[1.0]])
            joint_torques = torch.matmul(self.J.T, end_effector_force)
            recalc_force = torch.matmul(self.J_inv, joint_torques)
            print(joint_torques)
            print(recalc_force)
            
            demand_force = torch.tensor([[1.0],[0.0]])
            demand_torques = torch.matmul(self.J.T, demand_force)
            # demand_torques = torch.tensor([[3.0],[0.0]])
            calc_forces = torch.matmul(self.J_inv, demand_torques)
            
            # print(self.J)
            # print(self.J_inv)
            print('----')
            print(demand_torques)
            print(calc_forces)
            self.EndEffector_F = calc_forces
            
            pass
        
        
        
        
#%%
env = PlanarArm(2)

#%%

for i in range(10000):
    ang = torch.tensor(i*torch.pi/180)
    # env.control(-2.0, -1.5)
    start = time.perf_counter()
    env.control()
    env.endeffector_forces()
    end = time.perf_counter()
    dt = end-start
    # print(f"Control Time : {dt}")
    # print(f"end effector pos : ({env.xs[-1]},{env.ys[-1]})")
    env.plot()
    

