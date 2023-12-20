import abc
from abc import ABC
import torch

class Manager:
    def __init__(self, device):
        self.device=device
        self.components = []
        print('Component Init')
    
    def add_component(self, component):
        """Add a component to self.components"""
        self.components.add(component)

    @abc.abstractmethod 
    def physics_step(self, states, actions, observations):
        """Returns dY/dt"""