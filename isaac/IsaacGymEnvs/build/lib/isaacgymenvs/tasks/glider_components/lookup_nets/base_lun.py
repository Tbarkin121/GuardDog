import abc
from abc import ABC

import torch
import pickle
import __main__

class LUN:
    def __init__(self, net_name, device):
        print('LUN Init')
        self.net_name = net_name
        self.model_pth = '{}.pth'.format(self.net_name)
        self.device = device


    def load_net(self, net):
        self.network = net
        print(self.model_pth)
        self.network = torch.load(self.model_pth)
        self.network.eval()

    def get_output(self, input):
        """Returns network estimate of the parameters for a given input"""
        """Input and Output should both be torch tensors"""
        output = self.network(input)
        return output
    
    
    

