#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 20:39:04 2023

@author: tyler
"""


import os 
from qblade_lun import QBlade_LUN
import torch


lun = QBlade_LUN()
device="cuda"
input = torch.rand((10, 2), device=device)
print(input)
output = lun.get_output(input)
print(output)
