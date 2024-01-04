
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
import gc

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

print('Loading Model')
new_net = QBlade_nn()
new_net = torch.load('QBlade_LUN.pth')
new_net.eval()


#%%
save_name = '10cmBlade_no_outlier'
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
        print(i_scaled)
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
    

in_t = torch.unsqueeze(torch.tensor([19., 10400.0, -12.0], device=device), dim=0)
lookup_net = LUN(save_name)
thrust, torque = lookup_net.get_thrust_and_torque(in_t)
print('{} : {}'.format(thrust.detach().cpu().numpy(), torque.detach().cpu().numpy()))

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
        
        self.ranges = np.array([[-30.,30.],[200., 40000.],[0.5, 30.]])  #<------------ set ranges here
        
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
        self.p2 = self.win.addPlot(row=0, col=1, title="RPM")
        self.p3 = self.win.addPlot(row=0, col=2, title="WindVel")
        
        # Row 2 : State Variables
        self.p4 = self.win.addPlot(row=1, col=0, title="Prop Thrust") 
        self.p5 = self.win.addPlot(row=1, col=1, title="Prop Torque")

       
        
        
        self.curve1 = self.p1.plot(pen='r')
        self.curve2 = self.p2.plot(pen='g')
        self.curve3 = self.p3.plot(pen='b')
        self.curve4 = self.p4.plot(pen='w')
        self.curve5 = self.p5.plot(pen='c')
      
        
        self.data1 = torch.zeros((1000))
        self.data2 = torch.zeros((1000))
        self.data3 = torch.zeros((1000))
        self.data4 = torch.zeros((1000))
        self.data5 = torch.zeros((1000))

        self.lookup_net = LUN()
        
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
        
     
        
        self.data1 = torch.roll(self.data1, -1)
        self.data1[-1] = self.a
        self.data2 = torch.roll(self.data2, -1)
        self.data2[-1] = self.b
        self.data3 = torch.roll(self.data3, -1)
        self.data3[-1] = self.c
        
        in_t = torch.unsqueeze(torch.tensor([self.a, self.b, self.c], device=device, dtype=torch.float), dim=0)
        thrust, torque = self.lookup_net.get_thrust_and_torque(in_t)
        self.data4 = torch.roll(self.data4, -1)
        self.data4[-1] = thrust
        
        self.data5 = torch.roll(self.data5, -1)
        self.data5[-1] = torque
        
        
        
        self.curve1.setData(self.data1.detach().numpy())
        self.curve2.setData(self.data2.detach().numpy())
        self.curve3.setData(self.data3.detach().numpy())
        self.curve4.setData(self.data4.detach().numpy())
        self.curve5.setData(self.data5.detach().numpy())
        
        QtCore.QTimer.singleShot(0, self.update_plot)

        # time = time.perf_counter() - then
        # print(time)        
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = Widget()
    w.show()
    sys.exit(app.exec_())