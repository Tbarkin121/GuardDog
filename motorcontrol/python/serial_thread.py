# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 00:22:10 2024

@author: Plutonium
"""

from datastructs import *
from functions import *
from constants import *

import time
import threading
from queue import Queue, Empty

DEBUG_PRINT = False

def read_from_port(ser, lock, stop_event, shared_data):
    buffer = bytearray()
    
    while not stop_event.is_set():
        
        if ser.in_waiting > 0:
            # data = ser.read(ser.in_waiting)
            data = ser.read(ser.in_waiting)
            buffer.extend(data)

            
            # Parse complete messages from the buffer
            buffer, messages = parse_messages(buffer)

            
            if(DEBUG_PRINT):
                print(f"Message Count : {len(messages)}")
                for msg in messages:
                    print(msg)
                
            # Below here should just be used when updating the shared variables
            if(len(messages) > 0):
                
                
                with lock:
                    for message in messages:
                        if(DEBUG_PRINT):
                            print(f"Message Debug : {message}")
                            
                        if (type(message) == MotorStateUnion):
                            shared_data['timestamp'] = message.bits.Timestamp
                            shared_data['motor_speed'] = message.bits.Vel
                            shared_data['motor_position'] = message.bits.Mech_Ang/RADTOS16
                            shared_data['datalog_flag'] = 1
                            
                        elif (type(message) == BeaconHeaderUnion):
                            shared_data['beacon_flag'] = 1
                            shared_data['beacon_data'] = message
                            
                        elif (type(message) == PingHeaderUnion):
                            shared_data['ping_flag'] = 1
                            shared_data['ping_data'] = message
                            

                            
        time.sleep(0.001)  # Small delay to prevent high CPU usage (Up to 1000 Hz ignoring the time other stuff takes)


def parse_messages(buffer):
    messages = []

    while len(buffer) >= 4: # Need at least 4 bytes to compose the header
        if(DEBUG_PRINT):
            print('------')    
            print('Buffer:')
            for byte in buffer:
                print(byte, end=' ')
            print('\n------')    
            
        header.raw = int.from_bytes(buffer[0:4], byteorder='little')

        if(CheckHeaderCRC(header.raw)):
            
            
            if(header.bits.type == MCTL_ASYNC): #Request and Async have the same code
                asyncheader.raw = int.from_bytes(buffer[0:4], byteorder='little')
                if(DEBUG_PRINT):
                    print(f"Type: {asyncheader.bits.type}")
                    print(f"Payload Length: {asyncheader.bits.payload_length}")
                    print(f"Reserved: {asyncheader.bits.reserved}")
                    print(f"CRC: {asyncheader.bits.crch}")
                    
                if(len(buffer) >= 4+asyncheader.bits.payload_length):
                    payload_raw = buffer[4:4+asyncheader.bits.payload_length]
                    
                    # Use ctypes.memmove to copy the bytes into the raw field
                    source = (ctypes.c_ubyte * len(payload_raw)).from_buffer_copy(payload_raw)
                    ctypes.memmove(ctypes.byref(motorstate.raw), source, len(source))
                    if(DEBUG_PRINT):
                        print("MCTL_ASYNC")
                        print(len(payload_raw))
                        print(f"Time Stamp: {motorstate.bits.Timestamp}")
                        print(f"Electric Angle: {motorstate.bits.Elec_Ang}")
                        print(f"Mechanical Angle: {motorstate.bits.Mech_Ang/RADTOS16}")
                        print(f"Velocity: {motorstate.bits.Vel}")

                    buffer = buffer[4+asyncheader.bits.payload_length:] #Remove processed Data from Buffer
                    messages.append(motorstate)
                    
                else:
                    if(DEBUG_PRINT):
                        print('We have a header but the message payload is not complete.')
                    return buffer, messages
                
            
            elif(header.bits.type == MCTL_RESPONSE):
                responseheader.raw = int.from_bytes(buffer[0:4], byteorder='little')
                
                payload_raw = buffer[4:4+responseheader.bits.payload_length]
                if(DEBUG_PRINT):
                    print(payload_raw)
                source = (ctypes.c_ubyte * len(payload_raw)).from_buffer_copy(payload_raw)
                buffer = buffer[4+responseheader.bits.payload_length:] #Remove processed Data from Buffer
                
                if(DEBUG_PRINT):
                    print("MCTL_RESPONSE")
                    
                
            elif(header.bits.type == MCTL_BEACON):
                beaconheader.raw = int.from_bytes(buffer[0:4], byteorder='little')
                messages.append(beaconheader)
                buffer = buffer[4:] #Remove processed header 
                if(DEBUG_PRINT):
                    print("MCTL_BEACON")
                    
            elif(header.bits.type == MCTL_PING):
                pingheader.raw = int.from_bytes(buffer[0:4], byteorder='little')
                messages.append(pingheader)
                buffer = buffer[4:] #Remove processed header 
                if(DEBUG_PRINT):
                    print("MCTL_PING")
                
            
            else:
                if(DEBUG_PRINT | 1):
                    print('Unprocessed Message Type, Remove Header and Continue')
                buffer = buffer[4:] #Remove processed header 

            
        else:
            if(DEBUG_PRINT):
                print("CRCH Error, Step Buffer Forward 1 Byte")
            buffer = buffer[1:] #Step the buffer forward 1 byte
        # message_type = databyte & 0x0F
        # if(message_type == )
        # print(message_type)
    if(DEBUG_PRINT):
        print('Buffer Smaller Than Header Size')
    return buffer, messages



#%%

# # Lock to ensure thread safety
# lock = threading.Lock()

# # Create an event to signal the thread to stop
# stop_event = threading.Event()