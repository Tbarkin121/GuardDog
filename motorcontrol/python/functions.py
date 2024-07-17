# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 11:32:57 2024

@author: Plutonium
"""
from registers import *
import time
import struct
#######################################################################################
def int32_to_int8(n):
    mask = (1 << 8) - 1
    return [(n >> k) & mask for k in range(0, 32, 8)]
#######################################################################################
def int16_to_int8(n):
    mask = (1 << 8) - 1
    return [(n >> k) & mask for k in range(0, 16, 8)]
#######################################################################################
# Function to convert float to bytes
def data_raw_formatter(data, data_format):
    packed_data = struct.pack(data_format, *data)
    byte_array = np.frombuffer(packed_data, dtype=np.uint8)
    array_size = np.frombuffer(np.array(byte_array.size, dtype=np.uint16), dtype=np.uint8)
    
    # print(byte_array)
    # print(array_size)
    
    return byte_array, array_size
#######################################################################################
def ComputeHeaderCRC(header):
  crc = np.uint8(0)

  header &= 0x0fffffff

  crc = CRC4_Lookup8[crc ^ np.uint8(header         & 0xff)]
  crc = CRC4_Lookup8[crc ^ np.uint8((header >> 8)  & 0xff)]
  crc = CRC4_Lookup8[crc ^ np.uint8((header >> 16) & 0xff)]
  crc = CRC4_Lookup4[crc ^ np.uint8((header >> 24) & 0x0f)]

  header |= np.uint32(crc) << 28
  return header
#######################################################################################
def CheckHeaderCRC(header):
    crc = 0
    crc = CRC4_Lookup8[crc ^ np.uint8(header & 0xff)]
    crc = CRC4_Lookup8[crc ^ np.uint8((header >> 8) & 0xff)]
    crc = CRC4_Lookup8[crc ^ np.uint8((header >> 16) & 0xff)]
    crc = CRC4_Lookup8[crc ^ np.uint8((header >> 24) & 0xff)]
    return crc == 0
#######################################################################################
def computeCRC8(bb):

    sum = 0
    for b in bb:
        sum = CRC4_Lookup8[sum ^ b & 0xFF]

    res = sum & 0xFF
    res += sum >> 8
    return res
#######################################################################################
def getByteArray(arr):
    data=np.array(arr, np.uint8)
    data = (np.append(data,computeCRC8(data))).tolist()
    databy = bytearray(data)
    return databy
#######################################################################################
def readUINT32fromSerial(ser):
    res = ser.read(4)
    line = []
    for c in res:
        line.append(c)
    arr = np.array(line, np.uint8)
    res = arr.view(np.uint32)[0]
    return res
#######################################################################################
def send4BytesToSerial(ser, dataList, shared_data):
    ser.write(dataList)
    time.sleep(.1)
    res = readUINT32fromSerial(ser) 

    if CheckHeaderCRC(res):
        return res
    return []
#######################################################################################
def sendManyBytesToSerial(ser, dataList):
    ser.write(dataList)
    # time.sleep(.1)
    # res = readUINT32fromSerial(ser) 
    # if not CheckHeaderCRC(res):
    #     return []

    # dataLength, sync = getDataLength(res)
    # print(dataLength)
    # data = ser.read(dataLength)
    # line = []
    # for c in data:
    #     line.append(c)
    # arr = np.array(line, np.uint8)

    # return arr
#######################################################################################
def decodeRegValues(arr, regs):
    regResults = []
    ind = 0
    for reg in regs:
        if reg[1] == TYPE_DATA_8BIT:
            regResults.append(arr[ind])
            ind += 1
        elif reg[1] == TYPE_DATA_16BIT:
            regResults.append(arr[ind:ind+2].view(np.uint16)[0])
            ind += 2
        elif reg[1] == TYPE_DATA_32BIT:
            regResults.append(arr[ind:ind+4].view(np.uint32)[0])
            ind += 4
        else:    
            print ('Register cannot be included in the list and must be decoded separately')
            raise 'Register cannot be included in the list and must be decoded separately'
    return regResults
#######################################################################################
def decodeCommandResult(arr, type=TYPE_DATA_8BIT):
    if type == TYPE_DATA_8BIT:
        return arr
    elif type == TYPE_DATA_32BIT:
        res = arr[0:(len(arr)//4)*4].view(np.uint32)
        return res
#######################################################################################
def getDataLength(res32):
    length = (res32 >> 4) & 0x1FFF
    syncasync =  res32 & 0b1111
    return length, syncasync
#######################################################################################
def getBEACON(ser, shared_data, lock, version = 0, RX_maxSize = 7, TXS_maxSize = 7, TXA_maxSize = 32):
    beacon = 0x05
    beacon |= version << 4
    beacon |= RX_maxSize << 8
    beacon |= TXS_maxSize << 14
    beacon |= TXA_maxSize << 21
    
    packet = np.array(int32_to_int8(ComputeHeaderCRC(np.uint32(beacon))),np.uint8)
    
    ser.write(packet)
    time.sleep(0.1)
    while(not shared_data['beacon_flag']):
        time.sleep(0.01)
    res = shared_data['beacon_data']
    with lock:
        shared_data['beacon_flag']=0
        shared_data['beacon_data']=None
    # print(res.raw)
    if CheckHeaderCRC(res.raw):
        return res.bits.version, res.bits.crc, res.bits.rxs_max, res.bits.txs_max, res.bits.txa_max
    
    return []

#######################################################################################
def getPING(ser, shared_data, lock, cbit = 0, Nbit = 0, ipID = 0, packetNumber = 0):
    ping = 0x06
    ping |= cbit << 4
    ping |= cbit << 5
    ping |= Nbit << 6
    ping |= Nbit << 7
    ping |= ipID << 8
    ping |= packetNumber << 12   
    
    packet = np.array(int32_to_int8(ComputeHeaderCRC(np.uint32(ping))),np.uint8)
    
    ser.write(packet)
    time.sleep(0.1)
    while(not shared_data['ping_flag']):
        time.sleep(0.01)
    res = shared_data['ping_data']
    with lock:
        shared_data['ping_flag']=0
        shared_data['ping_data']=None
        
    if CheckHeaderCRC(res.raw):
        return res.bits.packet_number, res.bits.c1, res.bits.n1, res.bits.llid
    
    return []

#######################################################################################
def createDATA_PACKET( command):
    data_packet = 0x09
    data_packet |= (len(command) << 4) & 0x1FFF0
    header = np.array(int32_to_int8(ComputeHeaderCRC(np.uint32(data_packet))), np.uint8)
    return np.append(header,command)
#######################################################################################
def getCOMMAND(command, motorID = 1):
    command |= motorID
    return np.array(int16_to_int8(command), np.uint8)
#######################################################################################
def getREG(regs, motorID = 1):
    regRequest = getCOMMAND(GET_DATA_ELEMENT[0], motorID)
    for reg in regs:
        reg[0] |= motorID
        reg[0] |= reg[1]
        regRequest = np.append(regRequest,np.array(int16_to_int8(reg[0]), np.uint8))
    return regRequest
#######################################################################################
def setREG(regs, values, motorID = 1, dataraw_format=''):
    regRequest = getCOMMAND(SET_DATA_ELEMENT[0], motorID)
    for ind in range(len(regs)):
        reg = regs[ind]
        reg[0] |= motorID
        reg[0] |= reg[1]
        regRequest = np.append(regRequest,np.array(int16_to_int8(reg[0]), np.uint8))

        if reg[1] == TYPE_DATA_8BIT:
            regRequest = np.append(regRequest, values[ind])

        elif reg[1] == TYPE_DATA_16BIT:
            regRequest = np.append(regRequest, np.array(int16_to_int8(values[ind]), np.uint8))

        elif reg[1] == TYPE_DATA_32BIT:
            regRequest = np.append(regRequest, np.array(int32_to_int8(values[ind]), np.uint8))

        elif reg[1] == TYPE_DATA_STRING:
            regRequest = np.append(regRequest, np.array(values[ind], np.uint8))

        elif reg[1] == TYPE_DATA_RAW:
            byte_array, array_size = data_raw_formatter(values[ind], dataraw_format)
            regRequest = np.append(regRequest, array_size)
            regRequest = np.append(regRequest, byte_array)
            
            # Debugging Stuff
            if(0):
                print(array_size)
                print(byte_array)
                print(regRequest)
                print(type(regRequest))            
                # Convert to hexadecimal representation
                hex_vals = [f'0x{byte:02x}' for byte in regRequest]
                # Print the result
                print("Byte values (hexadecimal):", hex_vals)

    return np.array(regRequest, np.uint8)