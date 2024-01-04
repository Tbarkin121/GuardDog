import torch
from .lookup_nets.qblade_lun import QBlade_LUN

class Propeller:
    def __init__(self):
        print('Propeller Init')

    def __init__(self, m, device):
        self.device=device
        # Load Look Up Net
        self.LUN = QBlade_LUN(self.device)
        
        self.state = torch.zeros([m,3], device=self.device)
        self.wind = self.state.view(m,3)[:,0]
        self.rpm = self.state.view(m,3)[:,1]
        self.pitch = self.state.view(m,3)[:,2]
        
        self.pitch_old = torch.zeros((m), device=self.device) # Previous time step pitch value
                
        self.I = 3.5e-6
        self.rho = 1.225
        
        self.pitch_lp = 0.95
    def update_state(self, wind, omega, pitch_target): 
        self.wind[:] = wind[:,0]
        self.rpm[:] = torch.squeeze(omega)/(2*torch.pi)*60 #Rad Per Second to RPM

        # Low Pass Filter Pitch Update
        self.pitch[:] = self.pitch_lp * self.pitch_old + (1-self.pitch_lp) * torch.squeeze(pitch_target)
        self.pitch_old[:] = self.pitch[:]

        
    def get_thrust_and_torque(self, wind_vel, omega, pitch_action):
        # Get the Force and Moment coefficents
        # Input wind_vel is the wind velocity perpendicular to the rotor face
        # Input omega is the rad/s
        # Input pitch_action is the requested pitch angle of the propeller blades

        self.update_state(wind_vel, omega, pitch_action)
        # print('~~~~~~~~~~')
        # print(self.state)
        thrust, torque = self.LUN.get_thrust_and_torque(self.state)
        self.mechanical_power = omega*torque

        return thrust, torque
