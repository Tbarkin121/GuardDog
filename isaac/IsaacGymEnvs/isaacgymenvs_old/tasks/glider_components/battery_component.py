from .base_component import Component
import torch

class Battery(Component):
    def __init__(self, mass, ori, pos, max_energy, charge_percent):
        print('Battery Init')
        # Set Standard Parameters
        super().__init__(mass, ori, pos)
        # Set Additional Paramaters
        self.max_energy = max_energy
        self.initial_energy = self.max_energy * charge_percent

