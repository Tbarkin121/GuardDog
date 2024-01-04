# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 12:52:12 2024

@author: tylerbarkin
"""

import serial
import struct
import time
import numpy as np

class MCU_Comms():
    def __init__(self):
        self.in_data = np.zeros(4)
        self.out_data = np.zeros(4)
        self.open_port()
    
    def open_port(self):
        # Configure the serial connection
        self.ser = serial.Serial(
                        port='COM6',                         # Serial port
                        baudrate=115200,                     # Baud rate, should match STM32 setting
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
            data = self.ser.read(4 * 4)
            
            # Check if we received 16 bytes
            if len(data) == 16:
                # Unpack the bytes to four floats
                float_values = struct.unpack('4f', data)
                self.in_data = np.array(float_values)
                print(f"Received floats: {float_values}")
            else:
                print("Incomplete data received")
        
        except KeyboardInterrupt:
            print("Exiting...")
    

            
    def write_data(self):
        # Pack the floats into bytes
        data_to_send = struct.pack('4f', *self.out_data)
        
        try:
            # Send the packed bytes over the serial connection
            self.ser.write(data_to_send)
            print("Data sent")
            time.sleep(0.5)
        
        except Exception as e:
            print(f"Error: {e}")
        
    def __del__(self):
        # Destructor: close the serial port
        self.close_port()
        

comm_obj = MCU_Comms()

comm_obj.out_data = np.array([1.2, 2.3, 3.4, 4.5])
comm_obj.write_data()
comm_obj.read_data()

comm_obj.out_data = np.array([1.2, 2.3, 3.4, 4.5])*2.0
comm_obj.write_data()
comm_obj.read_data()

comm_obj.out_data = np.array([1.2, 2.3, 3.4, 4.5])*3.0
comm_obj.write_data()
comm_obj.read_data()

comm_obj.close_port()
    
    

