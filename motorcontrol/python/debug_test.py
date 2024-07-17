
import serial
import struct
import numpy as np
import matplotlib.pyplot as plt
from IPython import get_ipython
import time

import threading
from queue import Queue, Empty

from registers import *
from functions import *
from datastructs import *
from constants import *
from serial_thread import *


DEBUG_PRINT = False


#%%

class MotorController:
    def __init__(self, port, baudrate=921600):
        
        self.shared_data = {
            'timestamp': 0,
            'motor_speed': 0,
            'motor_position': 0,
            'datalog_flag': 0,
            'beacon_flag': 0,
            'beacon_data': None,
            'ping_flag': 0,
            'ping_data': None
        }
        
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        
        try:
            self.ser = serial.Serial(
                port=port,
                baudrate=baudrate,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                bytesize=serial.EIGHTBITS,
                timeout=0,
            )
            
            print(f"Connected to: {self.ser.portstr}")
        except serial.SerialException as e:
            print(f"Failed to connect to {port}: {e}")
            self.ser = None
            return
        
        
        self.header = HeaderUnion()
        self.beaconheader = BeaconHeaderUnion()
        self.pingheader = PingHeaderUnion()
        self.errorheader = ErrorHeaderUnion()
        self.requestheader = RequestHeaderUnion()
        self.responseheader = ResponseHeaderUnion()
        self.asyncheader = AsyncHeaderUnion()
        self.motorstate = MotorStateUnion()
        
        # Functions to Init With
        self.start_thread()
        
        self.MCU_Handshake()
        self.Request_Async_Data()
        self.Set_PID_Gains()

    def start_thread(self):
        self.thread = threading.Thread(target=read_from_port, args=(self.ser, self.lock, self.stop_event, self.shared_data))
        self.thread.daemon = True  # This allows the thread to exit when the main program exits
        self.thread.start()

    def stop_thread(self):
        self.stop_event.set()  # Signal the thread to stop
        self.thread.join()  # Wait for the thread to finish
        
    def close(self):
        self.ser.close()
        print(f"Connection to {self.ser.portstr} closed.")
        
    def MCU_Handshake(self):
        version, DATA_CRC, RX_maxSize, TXS_maxSize, TXA_maxSize = getBEACON(self.ser, self.shared_data, self.lock)
        print(f"{version}, {DATA_CRC}, {RX_maxSize}, {TXS_maxSize}, {TXA_maxSize}")
        version, DATA_CRC, RX_maxSize, TXS_maxSize, TXA_maxSize = getBEACON(self.ser, self.shared_data, self.lock, version, RX_maxSize, TXS_maxSize, TXA_maxSize)
        print(f"{version}, {DATA_CRC}, {RX_maxSize}, {TXS_maxSize}, {TXA_maxSize}")
        time.sleep(.1)

        packetNumber, ipID, cbit, Nbit = getPING(self.ser, self.shared_data, self.lock, packetNumber=0)
        time.sleep(.1)
        
    def Request_Async_Data(self, hf_rate=255, mark=1):
        motorID = 1
        Buf_Len = 14
        HF_Rate = hf_rate
        HF_Num = 2
        MF_Rate = 254
        MF_Num = 1
        Reg1 = MC_REG_ENCODER_EL_ANGLE[0]
        Reg1 |= motorID
        Reg2 = MC_REG_ENCODER_SPEED[0]
        Reg2 |= motorID
        Reg3 = MC_REG_CURRENT_POSITION[0]
        Reg3 |= motorID
        Mark = mark
        Format = 'HBBBBHHHB'
        packet = createDATA_PACKET(setREG([MC_REG_ASYNC_UARTA], [[Buf_Len, HF_Rate, HF_Num, MF_Rate, MF_Num, Reg1, Reg2, Reg3, Mark]], motorID=1, dataraw_format=Format))

        arr = sendManyBytesToSerial(self.ser, packet)
        time.sleep(.1)
        
    
        
    def Set_PID_Gains(self, P_Gain=5000, I_Gain=500, D_Gain=100):
        packet = createDATA_PACKET(setREG([MC_REG_POSITION_KP, MC_REG_POSITION_KI, MC_REG_POSITION_KD], [P_Gain, I_Gain, D_Gain], motorID=1))
        # packet = createDATA_PACKET(setREG([MC_REG_POSITION_KP, MC_REG_POSITION_KI, MC_REG_POSITION_KD], [10, 0, 1], motorID=1))
        arr = sendManyBytesToSerial(self.ser, packet)
        time.sleep(.1)
        
    def Start_Motor(self):
        decodeCommandResult(sendManyBytesToSerial(self.ser, createDATA_PACKET(getCOMMAND(START_MOTOR[0]))))
        time.sleep(.1)

    
    def Stop_Motor(self):
        decodeCommandResult(sendManyBytesToSerial(self.ser, createDATA_PACKET(getCOMMAND(STOP_MOTOR[0]))))
        time.sleep(.1)
    
    def Move_To_Position(self, pos, traj_time):
        arr = sendManyBytesToSerial(self.ser, createDATA_PACKET(setREG([MC_REG_POSITION_RAMP], [[pos, traj_time]], motorID=1, dataraw_format='ff')))
        time.sleep(traj_time)

#%%

# M1 = MotorController(port='COM3')
M2 = MotorController(port='COM4')

#%%
# M1.Start_Motor()
M2.Start_Motor()

#%%

M2.Request_Async_Data(158, 2)
#%%
start_time = time.perf_counter()
dt_list = []
for _ in range(250):    

    with M2.lock:
        if(M2.shared_data['datalog_flag']):
            dt = time.perf_counter() - start_time
            
                
            M2.shared_data['datalog_flag'] = 0
            dt_list.append(1/dt)
            # print(M2.shared_data['timestamp'])
            
            start_time = time.perf_counter()
            
        
    time.sleep(0.0001)
    
print(dt_list)

#%%
# M2.Stop_Motor()
#%%
# M2.stop_thread()

#%%
# import time
# import serial

# ser = serial.Serial(
#     port='COM4',
#     baudrate=921600,
#     parity=serial.PARITY_NONE,
#     stopbits=serial.STOPBITS_ONE,
#     bytesize=serial.EIGHTBITS,
#     timeout=0
#     )

# #%%

# ser.reset_input_buffer()
# time_start = time.perf_counter()
# for _ in range(1000):
    
#     if(ser.in_waiting):
#         print(f"data in queue {data_in_waiting}")
#         dt = time.perf_counter()-time_start
#         print(f"freq : {1/dt}")
#         # print(M2.ser.in_waiting)
#         # ser.reset_input_buffer()
#         data=ser.read(ser.in_waiting)
#         time_start = time.perf_counter()
#     time.sleep(0.001)

# # prev_data_in_waiting = 0
# # for _ in range(1000):
# #     data_in_waiting = ser.in_waiting
# #     if(not data_in_waiting == prev_data_in_waiting):
# #         print(f"data in queue {data_in_waiting}")
# #         prev_data_in_waiting = data_in_waiting
# #         dt = time.perf_counter()-time_start
# #         print(f"freq : {1/dt}")
# #         time_start = time.perf_counter()
# #     time.sleep(0.001)
    
# #%%
# ser.close()
#%%
# #%%
# # buffer = bytearray()

# time_array_2 = []


# for _ in range(100000):
#     if M2.ser.in_waiting > 0:

#         time_array_2.append(time.perf_counter())
        
#         M2.ser.reset_input_buffer()

                            
#     time.sleep(0.001)  # Small delay to prevent high CPU usage (Up to 1000 Hz ignoring the time other stuff takes)
        
    
#%%
