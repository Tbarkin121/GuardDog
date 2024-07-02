
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
while(1): #Just a spyder thing
    pass


#%%

class MotorController:
    def __init__(self, port, baudrate=921600):
        
        self.shared_data = {
            'timestamp': 0,
            'motor_speed': 0,
            'motor_position': 0,
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
                timeout=0
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
        
    def Request_Async_Data(self, mark=1):
        motorID = 1
        Buf_Len = 14
        HF_Rate = 255
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

M1 = MotorController(port='COM4')
M2 = MotorController(port='COM3')

#%%
M1.Start_Motor()
M2.Start_Motor()

#%%
M1.Set_PID_Gains(12000, 1000, 250)

#%%
for _ in range(3):
    M1.Move_To_Position(-6.0, 1.5)
    M1.Move_To_Position(2.0, 1.5)
M1.Move_To_Position(0.0, 0.3)

for _ in range(4):
    M2.Move_To_Position(-10.0, 0.5)
    M2.Move_To_Position(5.0, 0.5)
M2.Move_To_Position(0.0, 0.5)

#%%
M1.Stop_Motor()
M2.Stop_Motor()

#%%
M1.ser.close()
M2.ser.close()
#%%

# Shared data dictionary to hold motor speed, position, etc.
shared_data = {'timestamp':0,
               'motor_speed': 0, 
               'motor_position': 0,
               'beacon_flag':0,
               'beacon_data':None,
               'ping_flag':0,
               'ping_data':None}


#%%

ser = serial.Serial(
port='COM4',\
baudrate=921600,\
parity=serial.PARITY_NONE,\
stopbits=serial.STOPBITS_ONE,\
bytesize=serial.EIGHTBITS,\
timeout=0)
print("connected to: " + ser.portstr)


       
#%%


# Start the thread to read from the serial port and update shared data
thread = threading.Thread(target=read_from_port, args=(ser, lock, stop_event, shared_data))
thread.daemon = True  # This allows the thread to exit when the main program exits
thread.start()


#%%
stop_event.set()  # Signal the thread to stop
thread.join()  # Wait for the thread to finish



#%%
version, DATA_CRC, RX_maxSize, TXS_maxSize, TXA_maxSize = getBEACON(ser, shared_data, lock)
print(f"{version}, {DATA_CRC}, {RX_maxSize}, {TXS_maxSize}, {TXA_maxSize}")
version, DATA_CRC, RX_maxSize, TXS_maxSize, TXA_maxSize = getBEACON(ser, shared_data, lock, version, RX_maxSize, TXS_maxSize, TXA_maxSize)
print(f"{version}, {DATA_CRC}, {RX_maxSize}, {TXS_maxSize}, {TXA_maxSize}")
time.sleep(.1)

packetNumber, ipID, cbit, Nbit = getPING(ser, shared_data, lock, packetNumber=0)
time.sleep(.1)

#%%
packetNumber, ipID, cbit, Nbit = getPING(ser, shared_data, lock, packetNumber=0)
time.sleep(.1)




#%%

packet = createDATA_PACKET(setREG([MC_REG_POSITION_KP, MC_REG_POSITION_KI, MC_REG_POSITION_KD], [1000, 100, 100], motorID=1))
# packet = createDATA_PACKET(setREG([MC_REG_POSITION_KP, MC_REG_POSITION_KI, MC_REG_POSITION_KD], [10, 0, 1], motorID=1))
arr = sendManyBytesToSerial(ser, packet)
time.sleep(.1)

#%%
decodeCommandResult(sendManyBytesToSerial(ser, createDATA_PACKET(getCOMMAND(START_MOTOR[0]))))
time.sleep(.1)

#%%
# What do we need to put in this? 
# Buffer Size (2 Bytes)
# HF Rate, HF Num, MF Rate, MF Num (4 Bytes Total)
# HF Reg ID (2 Bytes Each) (0x0B11) Encoder Speed (0x0AD1) Electrical angle... 
# Mark (Random Number not 0, 1 Byte)

motorID = 1
Buf_Len = 14
HF_Rate = 255
HF_Num = 2
MF_Rate = 254
MF_Num = 1
Reg1 = MC_REG_ENCODER_EL_ANGLE[0]
Reg1 |= motorID
Reg2 = MC_REG_ENCODER_SPEED[0]
Reg2 |= motorID
Reg3 = MC_REG_CURRENT_POSITION[0]
Reg3 |= motorID
Mark = 1
Format = 'HBBBBHHHB'
packet = createDATA_PACKET(setREG([MC_REG_ASYNC_UARTA], [[Buf_Len, HF_Rate, HF_Num, MF_Rate, MF_Num, Reg1, Reg2, Reg3, Mark]], motorID=1, dataraw_format=Format))

arr = sendManyBytesToSerial(ser, packet)
time.sleep(.1)

#%%

for _ in range(1):
    arr = sendManyBytesToSerial(ser, createDATA_PACKET(setREG([MC_REG_POSITION_RAMP], [[-10.000, 0.250]], motorID=1, dataraw_format='ff')))
    time.sleep(1.2)
    arr = sendManyBytesToSerial(ser, createDATA_PACKET(setREG([MC_REG_POSITION_RAMP], [[10.000, 0.500]], motorID=1, dataraw_format='ff')))
    time.sleep(1.2)
    
arr = sendManyBytesToSerial(ser, createDATA_PACKET(setREG([MC_REG_POSITION_RAMP], [[0.000, 1.00]], motorID=1, dataraw_format='ff')))
time.sleep(1.2)
    
#%%
decodeCommandResult(sendManyBytesToSerial(ser, createDATA_PACKET(getCOMMAND(STOP_MOTOR[0]))))
time.sleep(.1)





    
#%%
# Main program execution (simulating other tasks)
try:
    while True:
        with lock:
            print(f"Motor Speed: {shared_data['motor_speed']}")
            print(f"Motor Position: {shared_data['motor_position']}")
        time.sleep(.1)  # Simulate doing other tasks
except KeyboardInterrupt:
    print("Exiting program.")
    stop_event.set()  # Signal the thread to stop
    thread.join()  # Wait for the thread to finish
finally:
    pass
    # ser.close()  # Close the serial port
    




#%%
   
ser.close()

#%%

# time.sleep(.1)
# decodeCommandResult(sendManyBytesToSerial(ser, createDATA_PACKET(getCOMMAND(GET_MCP_VERSION[0]))), GET_MCP_VERSION[1])
# time.sleep(.1)
# decodeCommandResult(sendManyBytesToSerial(ser, createDATA_PACKET(getCOMMAND(STOP_RAMP[0]))))
# time.sleep(.1)
# decodeCommandResult(sendManyBytesToSerial(ser, createDATA_PACKET(getCOMMAND(FAULT_ACK[0]))))
# time.sleep(.1)
# decodeCommandResult(sendManyBytesToSerial(ser, createDATA_PACKET(getCOMMAND(IQDREF_CLEAR[0]))))

# comm = getCOMMAND(command=GET_MCP_VERSION)
# createDATA_PACKET(comm)
# pack = createDATA_PACKET(setREG(
#     [MC_REG_CONTROL_MODE, MC_REG_SPEED_KP, MC_REG_SPEED_REF], [STC_SPEED_MODE, 500, 300]))


# arr = sendManyBytesToSerial(ser, createDATA_PACKET(setREG(
#     [MC_REG_CONTROL_MODE, MC_REG_SPEED_KP, MC_REG_SPEED_REF], [STC_SPEED_MODE, 500, 300])))

# arr = sendManyBytesToSerial(ser, createDATA_PACKET(getREG(
#     [MC_REG_CONTROL_MODE,  MC_REG_SPEED_KP, MC_REG_SPEED_REF])))

# results = decodeRegValues(arr, [MC_REG_CONTROL_MODE,  MC_REG_SPEED_KP, MC_REG_SPEED_REF])

# print(results)
