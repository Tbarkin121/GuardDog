import torch

from .base_component import Component
from .motor import Motor
from .propeller import Propeller

class Thruster(Component):
    def __init__(self, num_envs, dt, mass, ori, pos):
        print('Thruster Init')
        super().__init__(mass, ori, pos)
        # Sim Parameters
        self.device="cuda"
        self.dt = dt
        # Load Sub Components
        self.motor = Motor()
        self.prop = Propeller(num_envs, self.device)
        # Thruster State Variable 
        self.omega = torch.zeros((num_envs, 1), device=self.device) #rad/s
        self.I = self.prop.I + self.motor.I
        self.I *= 10 # Increasing I for simulation stability. The numbers were chosen a little arbitrarily anyways
        self.damping_constant = 3.13e-8 * 2 * torch.pi
        self.thruster_ori = ori

        self.motor_scale = 0.05 #+/- 0.05 Nm
        self.pitch_scale = 30.   # [50, -10]
        self.pitch_offset = 20.


    def update(self, wind_speed, pitch_action, motor_action, debug_msg=False):
        
        # # Matmul (m,1,3) (m,3,1) -> (m,1,1) 
        # self.wind_prop_vel = -torch.sum(torch.mul(wind_vec, self.thruster_ori), dim=1)
        self.prop_thrust, self.prop_torque = self.prop.get_thrust_and_torque(wind_speed, self.omega, pitch_action*self.pitch_scale + self.pitch_offset)
        self.motor_torque = self.motor.get_torque(motor_action*self.motor_scale, self.omega)
        
        total_torque = self.prop_torque + self.motor_torque        

        thrust_vec = torch.mul(self.prop_thrust, self.thruster_ori)        
        torque_vec = torch.mul(total_torque, self.thruster_ori)
        
        #Update Omega
        self.damping_torque = -self.damping_constant * self.omega
        self.damping_power = self.damping_torque * self.omega
        alpha = (total_torque  - self.omega * self.damping_constant)/ self.I
        self.omega += alpha*self.dt
        self.omega = torch.clamp(self.omega, torch.zeros_like(self.omega), torch.ones_like(self.omega)*10000)

        # print(self.omega[0,:])
        # print(thrust_vec[0,:])
        
        if(debug_msg==True):
            print('Wind:{:0.2f} | Omega:{:0.2f} | Pitch:{:0.2f} | Elec Pow:{} | \nMotor Pow:{:0.2f} | Prop Pow:{:0.2f} | Damp Pow:{:0.2f} | Thrust:{:0.2f} '.format(wind_speed[0,0].detach().cpu().numpy(), 
                                    self.omega[0,0].detach().cpu().numpy(), 
                                    self.prop.pitch[0].detach().cpu().numpy(), 
                                    self.motor.electrical_power[0,0].detach().cpu().numpy(),
                                    self.motor.mechanical_power[0,0].detach().cpu().numpy(), 
                                    self.prop.mechanical_power[0,0].detach().cpu().numpy(),
                                    self.damping_power[0,0].detach().cpu().numpy(),
                                    self.prop_thrust[0,0].detach().cpu().numpy()))
            # print("Motor Torque : {:0.2f} | Prop Torque : {:0.2f} | Damping Torque : {:0.2f}".format(self.motor_torque[0,0].detach().cpu().numpy(),
            #                                                                                          self.prop_torque[0,0].detach().cpu().numpy(),
            #                                                                                          self.damping_torque[0,0].detach().cpu().numpy()))


        return thrust_vec, torque_vec



    def get_force_and_power(self, action, wind_vel):
        force = action*self.max_forward_thrust
        power = -force*1000.0 
        return force, power
        
