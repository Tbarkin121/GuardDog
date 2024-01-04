#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 13:25:30 2023

@author: tyler
"""

import torch
import torch.nn as nn


class TestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits
    
    
net = TestNet()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)


#%%

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Make Plotting data.
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = R/3**2

# Plot the surface.
# surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       # linewidth=0, antialiased=False)

ax.scatter(X, Y, Z)

# Customize the z axis.
ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

#%%
# Make training data

dt_pts = 1000
x_data = torch.rand((dt_pts,1))
y_data = torch.rand((dt_pts,1))
r = torch.sqrt(x_data**2 + y_data**2)
z_data = r/3**2

