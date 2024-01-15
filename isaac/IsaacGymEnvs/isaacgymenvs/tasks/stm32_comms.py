# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 12:52:12 2024

@author: tylerbarkin
"""

import serial
import struct
import time
import numpy as np
import platform

class MCU_Comms():
    def __init__(self, enabled=1):
        if(enabled):
            self.act_data = np.zeros(12)
            self.obs_data = np.zeros(48)

            if platform.system() == 'Windows':
                self.port = 'COM6'
            else:
                self.port = '/dev/ttyACM1'
                    
            print('Using Port : {}'.format(self.port))
            self.open_port()
        
        
    
    def open_port(self):
        # Configure the serial connection
        self.ser = serial.Serial(
                        port=self.port,                      # Serial port
                        baudrate=460800,                     # Baud rate, should match STM32 setting
                        parity=serial.PARITY_NONE,
                        stopbits=serial.STOPBITS_ONE,
                        bytesize=serial.EIGHTBITS,
                        timeout=1                            # Read timeout in seconds
                    )
    
    def close_port(self):
        if self.ser.is_open:
            self.ser.close()
            print("Serial port closed")
    
        
    def read_data(self):
        try:
            # Read 16 bytes from the serial port (size of 4 floats)
            data = self.ser.read(12 * 4)
            
            # Check if we received 48 bytes
            if len(data) == 48:
                # Unpack the bytes to four floats
                float_values = struct.unpack('12f', data)
                self.act_data = np.array(float_values)
                # print(f"Received floats: {float_values}")
            else:
                print("Incomplete data received")
        
        except KeyboardInterrupt:
            print("Exiting...")
    

            
    def write_data(self):
        # Pack the floats into bytes
        data_to_send = struct.pack('48f', *self.obs_data)
        
        try:
            # Send the packed bytes over the serial connection
            self.ser.write(data_to_send)
            # print("Data sent")
        
        except Exception as e:
            print(f"Error: {e}")
        
    def __del__(self):
        # Destructor: close the serial port
        self.close_port()
        

# comm_obj = MCU_Comms()

# comm_obj.obs_data = np.zeros((48))
# for i in range(48):
#     comm_obj.obs_data[i] = i    

# for _ in range(1):
#     start_time = time.perf_counter()
#     comm_obj.write_data()
#     comm_obj.read_data()
#     elapsed_time = time.perf_counter() - start_time
#     print('Total Time = {}'.format(elapsed_time))

# comm_obj.close_port()
    
    

