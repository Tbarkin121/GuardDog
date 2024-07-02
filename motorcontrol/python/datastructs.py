# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 12:58:40 2024

@author: Plutonium
"""

import ctypes

class HeaderBits(ctypes.LittleEndianStructure):
    _fields_ = [
        ("type", ctypes.c_uint32, 4),          
        ("the_rest", ctypes.c_uint32, 28)           
    ]

class HeaderUnion(ctypes.Union):
    _fields_ = [
        ("bits", HeaderBits),
        ("raw", ctypes.c_uint32)
    ]

class BeaconHeaderBits(ctypes.LittleEndianStructure):
    _fields_ = [
        ("type", ctypes.c_uint32, 4),          
        ("version", ctypes.c_uint32, 3),  
        ("crc", ctypes.c_uint32, 1),   
        ("rxs_max", ctypes.c_uint32, 6),          
        ("txs_max", ctypes.c_uint32, 7),           
        ("txa_max", ctypes.c_uint32, 7),            
        ("crch", ctypes.c_uint32, 4)           
    ]

class BeaconHeaderUnion(ctypes.Union):
    _fields_ = [
        ("bits", BeaconHeaderBits),
        ("raw", ctypes.c_uint32)
    ]

class PingHeaderBits(ctypes.LittleEndianStructure):
    _fields_ = [
        ("type", ctypes.c_uint32, 4),          
        ("c1", ctypes.c_uint32, 1),     
        ("c2", ctypes.c_uint32, 1),     
        ("n1", ctypes.c_uint32, 1),           
        ("n2", ctypes.c_uint32, 1),
        ("llid", ctypes.c_uint32, 4),
        ("packet_number", ctypes.c_uint32, 16),
        ("crch", ctypes.c_uint32, 4)           
    ]

class PingHeaderUnion(ctypes.Union):
    _fields_ = [
        ("bits", PingHeaderBits),
        ("raw", ctypes.c_uint32)
    ]

class ErrorHeaderBits(ctypes.LittleEndianStructure):
    _fields_ = [
        ("type", ctypes.c_uint32, 4),         
        ("reserved1", ctypes.c_uint32, 4),  
        ("error_code1", ctypes.c_uint32, 8),     
        ("error_code2", ctypes.c_uint32, 8),
        ("reserved2", ctypes.c_uint32, 4),
        ("crch", ctypes.c_uint32, 4)          
    ]

class ErrorHeaderUnion(ctypes.Union):
    _fields_ = [
        ("bits", ErrorHeaderBits),
        ("raw", ctypes.c_uint32)
    ]

class RequestHeaderBits(ctypes.LittleEndianStructure):
    _fields_ = [
        ("type", ctypes.c_uint32, 4),          # bits [0-3]
        ("payload_length", ctypes.c_uint32, 14),  # bits [4-16]
        ("reserved", ctypes.c_uint32, 11),     # bits [17-27]
        ("crch", ctypes.c_uint32, 4)            # bits [28-31]
    ]

class RequestHeaderUnion(ctypes.Union):
    _fields_ = [
        ("bits", RequestHeaderBits),
        ("raw", ctypes.c_uint32)
    ]

class ResponseHeaderBits(ctypes.LittleEndianStructure):
    _fields_ = [
        ("type", ctypes.c_uint32, 4),          # bits [0-3]
        ("payload_length", ctypes.c_uint32, 13),  # bits [4-16]
        ("reserved", ctypes.c_uint32, 11),     # bits [17-27]
        ("crc", ctypes.c_uint32, 4)            # bits [28-31]
    ]

class ResponseHeaderUnion(ctypes.Union):
    _fields_ = [
        ("bits", ResponseHeaderBits),
        ("raw", ctypes.c_uint32)
    ]
    
class AsyncHeaderBits(ctypes.LittleEndianStructure):
    _fields_ = [
        ("type", ctypes.c_uint32, 4),          # bits [0-3]
        ("payload_length", ctypes.c_uint32, 13),  # bits [4-16]
        ("reserved", ctypes.c_uint32, 11),     # bits [17-27]
        ("crch", ctypes.c_uint32, 4)            # bits [28-31]
    ]

class AsyncHeaderUnion(ctypes.Union):
    _fields_ = [
        ("bits", AsyncHeaderBits),
        ("raw", ctypes.c_uint32)
    ]

# Packing data returns I define is probably a good place to start using the Mark Variable
class MotorStateBits(ctypes.LittleEndianStructure):
    _fields_ = [
        ("Timestamp", ctypes.c_uint32, 32),          
        ("Elec_Ang", ctypes.c_int16),
        ("Vel", ctypes.c_int16),
        ("Mech_Ang", ctypes.c_int32)
    ]

class MotorStateUnion(ctypes.Union):
    _fields_ = [
        ("bits", MotorStateBits),
        ("raw", ctypes.c_ubyte * ctypes.sizeof(MotorStateBits))
    ]

header = HeaderUnion()
beaconheader = BeaconHeaderUnion()
pingheader = PingHeaderUnion()
errorheader = ErrorHeaderUnion()
requestheader = RequestHeaderUnion()
responseheader = ResponseHeaderUnion()
asyncheader = AsyncHeaderUnion()
motorstate = MotorStateUnion()

#%%

# print(ctypes.c_ubyte * ctypes.sizeof(MotorStateBits))
# # Example bytes to write into the union
# example_bytes = bytearray([137, 0, 0, 32, 158, 73, 254, 10, 0, 0, 77, 0 ])

# # Create a MotorStateUnion instance
# motor_state = MotorStateUnion()

# # Use ctypes.memmove to copy the bytes into the raw field
# # Convert the bytearray to a ctypes array of the appropriate type
# source = (ctypes.c_ubyte * len(example_bytes)).from_buffer_copy(example_bytes)
# ctypes.memmove(ctypes.byref(motor_state.raw), source, len(example_bytes))

# # Print the structure fields to verify the values
# print(f"Timestamp: {motor_state.bits.Timestamp}")
# print(f"Position: {motor_state.bits.Elec_Ang}")
# print(f"Velocity: {motor_state.bits.Vel}")

# # Print the raw bytes to verify the bytes
# print("Raw bytes:", list(motor_state.raw))