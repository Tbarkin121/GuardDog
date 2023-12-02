#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 02:54:59 2023

@author: PandorasBox
"""
# https://studywolf.wordpress.com/2013/09/02/robot-control-jacobians-velocity-and-force/

import torch

theta1 = torch.tensor(torch.pi/4)
theta2 = torch.tensor(3*torch.pi/8)*5

t1 = -torch.sin(theta1) - torch.sin(theta1 + theta2)
t2 = torch.cos(theta1) + torch.cos(theta1 + theta2)
t3 = -torch.sin(theta1+theta2)
t4 = torch.cos(theta1+theta2)

J = torch.tensor([[t1,t2],[t3,t4]])
F = torch.tensor([[1.3],[1.0]])
t = torch.matmul(J,F)

print(t)

 