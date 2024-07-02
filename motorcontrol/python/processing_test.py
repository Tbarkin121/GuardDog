
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


#%%
#######################################################################################

# comm = getCOMMAND(command=GET_MCP_VERSION)
# createDATA_PACKET(comm)
# pack = createDATA_PACKET(setREG(
#     [MC_REG_CONTROL_MODE, MC_REG_SPEED_KP, MC_REG_SPEED_REF], [STC_SPEED_MODE, 500, 300]))
ser = serial.Serial(
port='COM4',\
baudrate=921600,\
parity=serial.PARITY_NONE,\
stopbits=serial.STOPBITS_ONE,\
bytesize=serial.EIGHTBITS,\
timeout=0)
print("connected to: " + ser.portstr)

time.sleep(0.1)

#%%

def read_from_port(ser, lock, stop_event, shared_data):
    buffer = bytearray()
    
    while not stop_event.is_set():
        if ser.in_waiting > 0:
            data = ser.read(ser.in_waiting)
            buffer.extend(data)

            # Parse complete messages from the buffer
            buffer, messages = parse_messages(buffer)
            
            print(len(messages))
            ## Below here should just be used when updating the shared variables
            # with lock:
            #     for message in messages:
            #         data_array = np.frombuffer(message, dtype=np.uint8)
            #         update_shared_data(data_array, shared_data)
                    
        time.sleep(0.01)  # Small delay to prevent high CPU usage

def parse_messages(buffer):
    messages = []
    header = HeaderUnion()
    motorstate = MotorStateUnion()
    while len(buffer) >= 4: # Need at least 4 bytes to compose the header
        print('------')    
        print('Buffer:')
        for byte in buffer:
            print(byte, end=' ')
        print('\n------')    
        header.raw = int.from_bytes(buffer[0:4], byteorder='little')
        if(CheckHeaderCRC(header.raw)):
            print(f"Type: {header.bits.type}")
            print(f"Payload Length: {header.bits.payload_length}")
            print(f"Reserved: {header.bits.reserved}")
            print(f"CRC: {header.bits.crc}")
            
            if(len(buffer) >= 4+header.bits.payload_length):
            
                if(header.bits.type == MCTL_ASYNC):
                    payload_raw = buffer[4:4+header.bits.payload_length]
                    # motorstate.raw = int.from_bytes(payload_raw, byteorder='little')
                    # Use ctypes.memmove to copy the bytes into the raw field
                    source = (ctypes.c_ubyte * len(payload_raw)).from_buffer_copy(payload_raw)
                    ctypes.memmove(ctypes.byref(motorstate.raw), source, len(example_bytes))
                    print(f"Time Stamp: {motorstate.bits.Timestamp}")
                    print(f"Position: {motorstate.bits.Pos}")
                    print(f"Velocity: {motorstate.bits.Vel}")
                    buffer = buffer[4+header.bits.payload_length:] #Remove processed Data from Buffer
                    messages.append(motorstate)
                else:
                    print('Unprocessed Message Type, Remove Header and Continue')
                    buffer = buffer[4:] #Remove processed header 
            else:
                print('We have a header but the message payload is not complete.')
                return buffer, messages
            
        else:
            print("CRCH Error, Step Buffer Forward 1 Byte")
            buffer = buffer[1:] #Step the buffer forward 1 byte
        # message_type = databyte & 0x0F
        # if(message_type == )
        # print(message_type)
    print('Buffer Smaller Than Header Size')
    return buffer, messages


# # Shared data dictionary to hold motor speed, position, etc.
# shared_data = {'motor_speed': 0, 'motor_position': 0}

# # Lock to ensure thread safety
# lock = threading.Lock()

# # Create an event to signal the thread to stop
# stop_event = threading.Event()

# # Start the thread to read from the serial port and update shared data
# thread = threading.Thread(target=read_from_port, args=(ser, lock, stop_event, shared_data))
# thread.daemon = True  # This allows the thread to exit when the main program exits
# thread.start()


# #%%
# # Stop Thread Execution 
# stop_event.set()
# thread.join()



# #%%
# # Main program execution (simulating other tasks)
# try:
#     while True:
#         with lock:
#             print(f"Motor Speed: {shared_data['motor_speed']}")
#             print(f"Motor Position: {shared_data['motor_position']}")
#         time.sleep(1)  # Simulate doing other tasks
# except KeyboardInterrupt:
#     print("Exiting program.")
#     stop_event.set()  # Signal the thread to stop
#     thread.join()  # Wait for the thread to finish
# finally:
#     # ser.close()  # Close the serial port
    
    
    
#%%


buffer = bytearray()
print(ser.in_waiting)
if ser.in_waiting > 0:
    data = ser.read(ser.in_waiting)
    buffer.extend(data)
    print(buffer)
    
    
buffer_keep = buffer

#%%

buffer = buffer_keep

buffer, messages = parse_messages(buffer)

ser.close()