
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
    def __init__(self, port, baudrate=921600, motor_name=""):
        
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
        
        self.Name = motor_name
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.time_start = time.perf_counter()
        
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
        self.Current_Limit = 1.0
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_demand_current = 0.0
        self.current_lpf = 0.9

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
        self.P = P_Gain
        self.I = I_Gain
        self.D = D_Gain
        packet = createDATA_PACKET(setREG([MC_REG_POSITION_KP, MC_REG_POSITION_KI, MC_REG_POSITION_KD], [P_Gain, I_Gain, D_Gain], motorID=1))
        arr = sendManyBytesToSerial(self.ser, packet)
        time.sleep(.1)
        
    def Start_Motor(self):
        decodeCommandResult(sendManyBytesToSerial(self.ser, createDATA_PACKET(getCOMMAND(START_MOTOR[0]))))
        time.sleep(.1)

    
    def Stop_Motor(self):
        decodeCommandResult(sendManyBytesToSerial(self.ser, createDATA_PACKET(getCOMMAND(STOP_MOTOR[0]))))
        time.sleep(.1)
        
    def Fault_Ack(self):
        decodeCommandResult(sendManyBytesToSerial(self.ser, createDATA_PACKET(getCOMMAND(FAULT_ACK[0]))))
        time.sleep(.1)
        
    
    def Move_To_Position(self, pos, traj_time):
        arr = sendManyBytesToSerial(self.ser, createDATA_PACKET(setREG([MC_REG_POSITION_RAMP], [[pos, traj_time]], motorID=1, dataraw_format='ff')))
        time.sleep(traj_time)
        
    def Set_Current(self, current_amp, duration):
        RSHUNT = 0.007
        AMPLIFICATION_GAIN = 20
        ADC_REFERENCE_VOLTAGE = 3.3
        CURRENT_CONV_FACTOR = int((65536.0 * RSHUNT * AMPLIFICATION_GAIN)/ADC_REFERENCE_VOLTAGE)
        current_digit = int(current_amp*CURRENT_CONV_FACTOR)
        arr = sendManyBytesToSerial(self.ser, createDATA_PACKET(setREG([MC_REG_TORQUE_RAMP], [[current_digit, duration]], motorID=1, dataraw_format='iH')))
        
        
    def PID_Pos_CTRL(self, targ):
        
        if(self.shared_data['datalog_flag']):
            
            self.shared_data['datalog_flag'] = 0
            
            error = targ - self.shared_data['motor_position']
            self.integral += error * (time.perf_counter() - self.time_start)
           
            
            derivative = (error - self.prev_error) / (time.perf_counter() - self.time_start)
            
            # Compute control signal
            P_current = self.P * error
            I_current = self.I * self.integral
            # integral clipping
            I_current = np.clip(I_current, -1.0, 1.0)
            D_current = self.D * derivative
            
            demand_current = P_current + I_current + D_current
            # print(demand_current)
            # demand_current = self.prev_demand_current*self.current_lpf + demand_current*(1-self.current_lpf)
            # print(demand_current)
            demand_current = np.clip(demand_current, -self.Current_Limit, self.Current_Limit)  # Saturate the control signal to the current limit
            # print(demand_current)

            self.Set_Current(demand_current, 0)
            self.prev_demand_current = demand_current            
            # Prepare for the next iteration
            self.prev_error = error
            
            
            if(DEBUG_PRINT):
                # print(self.shared_data['timestamp'])
                # print(self.shared_data['motor_position'])
                # print(self.shared_data['motor_speed'])
                # print(self.shared_data['datalog_flag'])
                print(f"{self.Name} : {demand_current}")
                print(f'Update Rate : {1/(time.perf_counter() - self.time_start)}')
                
            self.time_start = time.perf_counter()

while(1): #Just a spyder thing
    pass

    
#%%

M1 = MotorController(port='COM3', motor_name = "M1")
M2 = MotorController(port='COM4', motor_name = "M2")

# for _ in range(100):
#     if(M1.shared_data['datalog_flag']):
#         print(f'Update Rate : {1/(time.perf_counter() - M1.time_start)}')
#         M1.shared_data['datalog_flag'] = 0
#         M1.time_start = time.perf_counter()
#     time.sleep(0.001)

#%%
M1.Start_Motor()
M2.Start_Motor()

#%%

HFRate = int((1/1000)/(1/16000))
M1.Request_Async_Data(HFRate,3)
M2.Request_Async_Data(HFRate,3)



#%%
M1_targ_pos = 0.0
M2_targ_pos = 0.0

M1_targ_step = 0.005
M2_targ_step = 0.005

# M1.Set_PID_Gains(1, 0, 0)
# M2.Set_PID_Gains(1, 0, 0)

M1.P = 4.0
M1.I = 1.0
M1.D = 0.03

M2.P = 4.0
M2.I = 1.0
M2.D = 0.03



M1_GR=5
M2_GR=5

#%%

M1.Current_Limit = 11.5
M2.Current_Limit = 11.5

DataLog = []
for _ in range(10000):
    # print(f"M1:{M1_targ_pos}. M2:{M2_targ_pos}")
    
    M1_targ_pos += M1_targ_step    
    M2_targ_pos += M2_targ_step
    
    if(M1_targ_pos > 0 or M1_targ_pos < -M1_GR*np.pi/2):
        M1_targ_step = -M1_targ_step
    
    if(M2_targ_pos > 0 or M2_targ_pos < -M2_GR*np.pi/2):
        M2_targ_step = -M2_targ_step
        
    M1.PID_Pos_CTRL(M1_targ_pos)
    M2.PID_Pos_CTRL(M2_targ_pos)
    DataLog.append([M1.prev_demand_current, M2.prev_demand_current])
    print(f"M1:{M1.prev_demand_current}. M2:{M2.prev_demand_current}")
        
    time.sleep(0.001)

#%%
M1.Set_Current(0.0, 0)
M2.Set_Current(0.0, 0)

#%%    

M1.Current_Limit = 11.5
M2.Current_Limit = 11.5

M1_targ_pos = -5.0
M2_targ_pos = -10.0
M1_targ_step = 0.015
M2_targ_step = M1_targ_step*2

M1.P = 8.0
M1.I = 2.5
M1.D = 0.02

M2.P = 8.0
M2.I = 2.5
M2.D = 0.02

DataLog = []
for _ in range(2000):
    # print(f"M1:{M1_targ_pos}. M2:{M2_targ_pos}")
    
    M1_targ_pos += M1_targ_step    
    M2_targ_pos += M2_targ_step
    
    if(M1_targ_pos > -1 or M1_targ_pos < -5):
        M1_targ_step = -M1_targ_step
    
    if(M2_targ_pos > -2 or M2_targ_pos < -10):
        M2_targ_step = -M2_targ_step
        
    M1.PID_Pos_CTRL(M1_targ_pos)
    M2.PID_Pos_CTRL(M2_targ_pos)
    DataLog.append([M1.prev_demand_current, 
                    M2.prev_demand_current, 
                    M1.shared_data['motor_position'], 
                    M2.shared_data['motor_position'],
                    M1_targ_pos,
                    M2_targ_pos])
    print(f"M1:{M1.prev_demand_current}. M2:{M2.prev_demand_current}")
        
    time.sleep(0.001)


M1.Set_Current(0.0, 0)
M2.Set_Current(0.0, 0)

 #%%
DataLog_np = np.array(DataLog)
fig, axs = plt.subplots(3, 1, figsize=(12, 18))
for p in range(2):
    axs[0].plot(DataLog_np[:,p], label=f'M{p+1}')
    axs[1].plot(DataLog_np[:,p+2], label=f'M{p+1}')
    axs[2].plot(DataLog_np[:,p+4], label=f'M{p+1}')
    
for i in range(3):
    axs[i].set_xlabel('Sample')
    axs[i].set_ylabel('Data')
    axs[i].set_title('Data Traces')
    axs[i].legend()
    axs[i].grid(True)
plt.show()
    
#%%
M1.Stop_Motor()
M2.Stop_Motor()
#%%



M1_targ_pos = -7.0
M2_targ_pos = -8.0
for _ in range(100000):
    
    M1.PID_Pos_CTRL(M1_targ_pos)
    M2.PID_Pos_CTRL(M2_targ_pos)
    print(f"M1:{M1.prev_demand_current}. M2:{M2.prev_demand_current}")
    time.sleep(0.001)

#%%
M1.Set_Current(0.0, 0)
M2.Set_Current(0.0, 0)


#%%
M1.Set_PID_Gains(12000, 1000, 250)

#%%

for _ in range(3):
    M1.Move_To_Position(-2.0, 1.0)
    M1.Move_To_Position(2.0, 1.0)
M1.Move_To_Position(0.0, 1.0)

for _ in range(4):
    M2.Move_To_Position(-2.0, 0.5)
    M2.Move_To_Position(2.0, 0.5)
M2.Move_To_Position(0.0, 0.5)

#%%
M1.Move_To_Position(-6.0, 1.0)
M2.Move_To_Position(2.0, 0.5)


#%%
M1.Stop_Motor()
M2.Stop_Motor()

#%%
M1.ser.close()
M2.ser.close()


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
