import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QLabel, QSizePolicy, QSlider, QSpacerItem, \
    QVBoxLayout, QWidget
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import numpy as np

#-------------------------
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

import time
import pickle

device = "cuda"
m = 2 # Number of Environments
#%%

# Define model
import torch.nn as nn
import torch.nn.functional as F

# Define model
class QBlade_nn(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(3, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 2)
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

 
#%%
save_name = 'BetzBlade12p5cm_lap_outlier'

class LUN():
    def __init__(self, save_name):
        self.save_name = save_name
        model_pth = '{}.pth'.format(self.save_name)
        self.net = QBlade_nn()
        self.net = torch.load(model_pth)
        # self.net.eval()
        
        self.read_scales()

        
    def get_thrust_and_torque(self, i):
        i_scaled = i * torch.tensor((self.wind_modifier, self.rpm_modifier, self.pitch_modifier), device=device )
        # print(',,,,,,,,,,,,,')
        # print(i_scaled)
        
        i = torch.clip(i_scaled, -1., 1.)
        o = self.net(i)
        thrust = o[:,0] / self.thrust_modifier
        torque = o[:,1] / self.torque_modifier
        
        # print(o)
        # thrust = torch.ones_like(thrust, device=device)
        # torque = torch.ones_like(torque, device=device)
        
        return thrust, torque
    
    def read_scales(self):
        with open('{}.txt'.format(self.save_name), 'rb') as f:  # Python 3: open(..., 'rb')
            scale_dict = pickle.load(f)
            self.wind_modifier = torch.tensor(scale_dict['wind_modifier'], device=device)
            self.rpm_modifier = torch.tensor(scale_dict['rpm_modifier'], device=device)
            self.pitch_modifier = torch.tensor(scale_dict['pitch_modifier'], device=device)
            self.thrust_modifier = torch.tensor(scale_dict['thrust_modifier'], device=device)
            self.torque_modifier = torch.tensor(scale_dict['torque_modifier'], device=device)
            # print(scale_dict)
    
l = LUN(save_name)
in_t = torch.unsqueeze(torch.tensor([18.0, 6800.0, 6.0], device=device), dim=0)
thrust, torque = l.get_thrust_and_torque(in_t)

print(thrust)
print(torque)
#%%
class Motor():
    # https://support.maxongroup.com/hc/en-us/articles/360004496254-maxon-Motors-as-Generators
    # The Thruster handles the propeller and motor sub systems
    # The propeller uses a look up net to determine thrust and moment coefficents
    # The motor can add torque to the system
    # The propeller and motor torques only effect the propeller state and currently do not act on the plane
    def __init__(self, m):
        # State Variables
        self.omega = torch.zeros((m), device=device)     # (rad/s)
        self.motor_supply_voltage = 12                   # V
        
        # Motor Constants (https://www.maxongroup.us/medias/sys_master/root/8816803676190/15-205-EN.pdf)
        self.I = 1.81e-7 
        self.speed_constant = 1510 * 2 * torch.pi / 60   # (rad/s)/V
        self.generator_constant = 1/self.speed_constant  # V/(rad/s)
        self.torque_constant = 1/self.speed_constant     # Nm/A
        self.motor_resistance = 0.421                    # Ohms
        self.max_efficiency = 0.92                       # 
        
        self.I_noload = -0.147                            # A
        
        self.I_load   = 0.0                              # A
        self.speed_torque_gradient = 101e3               #rpm/Nm (rpm/mNm in maxon datasheet)
        
        # Useful Variables
        self.rads2rpm = 30/torch.pi
        self.max_current = self.motor_supply_voltage/self.motor_resistance
        self.max_torque = self.max_current*self.torque_constant
        self.motor_scale = 1.0
    
    def calc_no_load_voltage(self):
        return self.omega*self.generator_constant
    
    def calc_generator_voltage(self, Il):
        # Il is load current
        # print(self.motor_resistance*Il)
        
        # print('~~~~~~~~~~~~~~~~~~~~')
        # print(self.motor_resistance)
        # print(Il)
        # print(Il[0])
        # print(self.motor_resistance*Il)
        # print('{} : {}'.format(self.omega[0]*self.generator_constant, Il[0]*self.motor_resistance))
        # return self.calc_no_load_voltage() - self.motor_resistance*Il
        return self.calc_no_load_voltage() + self.motor_resistance*Il
    
    def calc_max_load_current(self):
        return self.calc_no_load_voltage() / self.motor_resistance
    
    def calc_torque(self, Il):
        # Il is load current
        return self.torque_constant * (Il - torch.sign(Il)*self.I_noload)
    
    def calc_elec_power(self, Il):
        # Il is load current
        # Power is in Watts
        # +P = power into the battery
        # -P = power out of the battery
        return self.calc_generator_voltage(Il) * Il
    
    def calc_mech_power(self, Il):
        # Power is in Watts
        return self.omega*self.calc_torque(Il)
        
    
        
        
    def get_torque(self, torque_request, omega):
        # Update Omega from thruster sim
        self.omega = omega
        self.I_load_request = torque_request / self.torque_constant
        self.generator_voltage = self.calc_generator_voltage(self.I_load) # Using the previous load, calculate the generator induced voltage

        
        max_available_voltage = self.motor_supply_voltage - self.generator_voltage
        max_available_voltage = torch.where(max_available_voltage < 0, torch.zeros_like(max_available_voltage), max_available_voltage)

        max_available_current = max_available_voltage/self.motor_resistance
        
        min_available_voltage = -self.motor_supply_voltage - self.generator_voltage
        min_available_voltage = torch.where(min_available_voltage > 0, torch.zeros_like(min_available_voltage), min_available_voltage)
        min_available_current = min_available_voltage/self.motor_resistance
        
        self.I_load = (torch.clip(self.I_load_request, min_available_current, max_available_current) + self.I_load)/2
        self.torque_delivered = self.I_load * self.torque_constant
        
        self.electrical_power = self.calc_elec_power(self.I_load)
        self.mechanical_power = self.calc_mech_power(self.I_load)
        # print('```````````````')
        # print(torque_request)
        # print(self.torque_delivered)
        # print('I_REQ {} \nI_MIN{} \nIMAX{} \nI_DEL{}'.format(self.I_load_request, min_available_current, max_available_current, self.I_load))
        # print(self.I_load)
        return self.torque_delivered
        # self.torque_delivered = torch.max(torch.min(torch.abs(self.requested_torque), torch.abs(self.available_torque)), torch.zeros((m), device=device))        
    
#%%
class Propeller():
    def __init__(self, m):
        self.prop_LUN = LUN(save_name)

        self.state = torch.zeros([m,3], device=device)
        self.wind = self.state.view(m,3)[:,0]
        self.rpm = self.state.view(m,3)[:,1]
        self.pitch = self.state.view(m,3)[:,2]
        
        self.pitch_old = torch.zeros((m), device=device) # Previous time step pitch value
        
        
        self.pitch_scale = 1.0;
        self.pitch_offset = 0.0;
        self.radius = 0.12; # 5 inch radius
        self.area = 3.1415*self.radius**2;
        self.I = 3.5e-6;
        self.rho = 1.225;
        
        self.pitch_lp = 0.95
        
    def update_pitch(self, p):
        # Low Pass Filter Pitch Update
        # Takes an action [-1, 1]
        # Unscales the action
        # Low pass filter updates pitch state variable
        # unscaled_pitch = p*self.pitch_scale + self.pitch_offset
        
        self.pitch[:] = self.pitch_lp * self.pitch_old + (1-self.pitch_lp) * p
        self.pitch_old[:] = p
        
    def get_thrust_and_torque(self, omega, V):
        # Get the Force and Moment coefficents
        # Input omega is the rad/s
        # Input V is the wind velocity tensor
        self.wind[:] = V
        self.rpm[:] = omega/(2*torch.pi)*60 #Rad Per Second to RPM
        

        # print('~~~~~~~~~~')
        # print(self.state)
        thrust, torque = self.prop_LUN.get_thrust_and_torque(self.state)

        self.mechanical_power = omega*torque

        return thrust, torque
        
#%%
class Thruster():
    # The Thruster handles the propeller and motor sub systems
    # The propeller uses a look up net to determine thrust and moment coefficents
    # The motor can add torque to the system
    # The propeller and motor torques only effect the propeller state and currently do not act on the plane
    def __init__(self, m):
        self.dt = 0.02
        self.omega = torch.zeros((m), device=device) #Rads Per Second
        self.prop = Propeller(m)
        self.motor = Motor(m)
        self.I = self.prop.I + self.motor.I
        self.I *= 10
        self.thruster_ori = torch.tensor([-1.0, 0.0, 0.0], device=device).repeat(m,1)
        self.damping_constant = 3.13e-8 * 2 * torch.pi

    def update(self, action, obs, debug_torque):
        # Convert Actions to Units
        self.prop.update_pitch(action[:,0])
        
        self.motor_torque = self.motor.get_torque(action[:,1], self.omega)

        wind_vel_apparent = obs
        # # Matmul (m,3,1) (m,1,3) -> (m,1,1) 
        V = -torch.sum(torch.mul(wind_vel_apparent, self.thruster_ori), dim=1)
        
        self.prop_thrust, self.prop_torque = self.prop.get_thrust_and_torque(self.omega, V)
        

        prop_thrust_vec = torch.mul(torch.unsqueeze(self.prop_thrust, dim=1), self.thruster_ori)
        
        # print(self.motor_torque)
        total_torque = self.prop_torque + self.motor_torque + debug_torque
        # total_torque = self.motor_torque + debug_torque
        # total_torque = self.prop_torque
        
        moment_vec = torch.mul(torch.unsqueeze(total_torque, dim=1), self.thruster_ori)
        # print(total_torque)

        #Update prop state (omega)
        self.damping_power = -self.damping_constant * self.omega**2
        alpha = (total_torque  - self.omega * self.damping_constant)/ self.I
        self.omega += alpha*self.dt
        self.omega = torch.clamp(self.omega, torch.zeros_like(self.omega), torch.ones_like(self.omega)*10000)

        return prop_thrust_vec, moment_vec

    
#%%
class Slider(QWidget):
    def __init__(self, minimum, maximum, parent=None):
        super(Slider, self).__init__(parent=parent)
        self.verticalLayout = QVBoxLayout(self)
        self.label = QLabel(self)
        self.verticalLayout.addWidget(self.label)
        self.horizontalLayout = QHBoxLayout()
        spacerItem = QSpacerItem(0, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.slider = QSlider(self)
        self.slider.setOrientation(Qt.Vertical)
        self.horizontalLayout.addWidget(self.slider)
        spacerItem1 = QSpacerItem(0, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.resize(self.sizeHint())

        self.minimum = minimum
        self.maximum = maximum
        self.slider.valueChanged.connect(self.setLabelValue)
        self.x = None
        self.setLabelValue(self.slider.value())


    def setLabelValue(self, value):
        self.x = self.minimum + (float(value) / (self.slider.maximum() - self.slider.minimum())) * (
        self.maximum - self.minimum)
        self.label.setText("{0:.4g}".format(self.x))


class Widget(QWidget):
    def __init__(self, parent=None):
        super(Widget, self).__init__(parent=parent)
        self.t = 0
        self.horizontalLayout = QHBoxLayout(self)
        
        self.ranges = np.array([[-30.,30.],[0.5,30.],[-0.15,0.15],[-0.15, 0.15]])  #<------------ set ranges here
        
        self.num_sliders = self.ranges.shape[0]
        self.w = []
        
        for i in range(self.num_sliders):
            self.w.append(Slider(self.ranges[i,0], self.ranges[i,1]))
            self.horizontalLayout.addWidget(self.w[i])
            self.w[i].x = 0.0
            
        
        self.win = pg.GraphicsWindow(title="title", size=[1600, 1600])
        self.horizontalLayout.addWidget(self.win)

        # Row 1 : Inputs 
        self.p1 = self.win.addPlot(row=0, col=0, title="Pitch") #<------------ setup subplot structure
        self.p2 = self.win.addPlot(row=0, col=1, title="Wind Vel")
        self.p3 = self.win.addPlot(row=0, col=2, title="Requested Motor Torque")
        self.p4 = self.win.addPlot(row=0, col=3, title="Debug Magic Torque")
        # Row 2 : State Variables
        self.p5 = self.win.addPlot(row=1, col=0, title="Omega") 
        self.p6 = self.win.addPlot(row=1, col=1, title="Prop Thrust")
        self.p7 = self.win.addPlot(row=1, col=2, title="Prop Torque")
        self.p8 = self.win.addPlot(row=1, col=3, title="Motor Torque")
        # Row 3 : Power
        self.p9 = self.win.addPlot(row=2, col=0, title="Motor Mechanical Power")
        self.p10 = self.win.addPlot(row=2, col=1, title="Motor Electrical Power")
        self.p11 = self.win.addPlot(row=2, col=2, title="Prop Mechanical Power")
        self.p12 = self.win.addPlot(row=2, col=3, title="Damping Power Loss")
        
        self.curve1 = self.p1.plot(pen='r')
        self.curve2 = self.p2.plot(pen='g')
        self.curve3 = self.p3.plot(pen='b')
        self.curve4 = self.p4.plot(pen='w')
        self.curve5 = self.p5.plot(pen='c')
        self.curve6 = self.p6.plot(pen='y')
        self.curve7 = self.p7.plot(pen='m')
        self.curve8 = self.p8.plot(pen=(255, 0.0, 255))
        self.curve9 = self.p9.plot(pen=(255, 128, 255))
        self.curve10 = self.p10.plot(pen=(255, 255, 128))
        self.curve11 = self.p11.plot(pen=(128, 255, 128))
        self.curve12 = self.p12.plot(pen=(128, 255, 128))
        self.thruster = Thruster(m);
        
        self.data1 = torch.zeros((100))
        self.data2 = torch.zeros((100))
        self.data3 = torch.zeros((100))
        self.data4 = torch.zeros((100))
        self.data5 = torch.zeros((100))
        self.data6 = torch.zeros((100))
        self.data7 = torch.zeros((100))
        self.data8 = torch.zeros((100))
        self.data9 = torch.zeros((100))
        self.data10 = torch.zeros((100))
        self.data11 = torch.zeros((100))
        self.data12 = torch.zeros((100))
        
        self.update_plot()
        
        
        
        # for i in range(self.num_sliders):
        #     self.w[i].slider.valueChanged.connect(self.update_plot)

    
        # QtCore.QTimer.singleShot(0, self.update_plot)
    
    def update_plot(self):
        # then = time.perf_counter()
        # self.t = self.t+self.dt
        
        self.a = self.w[0].x                      #<------------ get slider vals
        self.b = self.w[1].x
        self.c = self.w[2].x
        self.d = self.w[3].x
        
        self.action = torch.tensor((self.a,self.c), device=device).repeat(m,1)
        self.obs = torch.tensor((self.b,0,0), device=device).repeat(m,1)
        self.thruster.update(self.action, self.obs, self.d)
        
        self.data1 = torch.roll(self.data1, -1)
        self.data1[-1] = self.thruster.prop.pitch[0]
        self.data2 = torch.roll(self.data2, -1)
        self.data2[-1] = self.thruster.prop.wind[0]
        self.data3 = torch.roll(self.data3, -1)
        self.data3[-1] = self.c
        self.data4 = torch.roll(self.data4, -1)
        self.data4[-1] = self.d
        
        self.data5 = torch.roll(self.data5, -1)
        self.data5[-1] = self.thruster.omega[0]
        self.data6 = torch.roll(self.data6, -1)
        self.data6[-1] = self.thruster.prop_thrust[0]
        self.data7 = torch.roll(self.data7, -1)
        self.data7[-1] = self.thruster.prop_torque[0]
        self.data8 = torch.roll(self.data8, -1)
        self.data8[-1] = self.thruster.motor_torque[0]
        
        self.data9 = torch.roll(self.data9, -1)
        self.data9[-1] = self.thruster.motor.mechanical_power[0]
        self.data10 = torch.roll(self.data10, -1)
        self.data10[-1] = self.thruster.motor.electrical_power[0]
        
        
        self.power = self.thruster.omega * self.thruster.motor_torque
        self.data11 = torch.roll(self.data11, -1)
        self.data11[-1] = self.thruster.prop.mechanical_power[0]
        self.data12 = torch.roll(self.data12, -1)
        self.data12[-1] = self.thruster.damping_power[0]
        
        
        self.curve1.setData(self.data1.detach().numpy())
        self.curve2.setData(self.data2.detach().numpy())
        self.curve3.setData(self.data3.detach().numpy())
        self.curve4.setData(self.data4.detach().numpy())
        self.curve5.setData(self.data5.detach().numpy())
        self.curve6.setData(self.data6.detach().numpy())
        self.curve7.setData(self.data7.detach().numpy())
        self.curve8.setData(self.data8.detach().numpy())
        self.curve9.setData(self.data9.detach().numpy())
        self.curve10.setData(self.data10.detach().numpy())
        self.curve11.setData(self.data11.detach().numpy())
        self.curve12.setData(self.data12.detach().numpy())
        
        QtCore.QTimer.singleShot(0, self.update_plot)

        # time = time.perf_counter() - then
        # print(time)        
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = Widget()
    w.show()
    sys.exit(app.exec_())