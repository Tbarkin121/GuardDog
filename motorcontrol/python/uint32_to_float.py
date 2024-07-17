import struct

# Hexadecimal number
hex_numbers = [0x41200000, 0x3F800000, 0xC1200000, 0x3F800000,0x00000000,0x3F000000, 0x000003E8]

for num in hex_numbers:
    # Pack the hex number into a bytes object
    bytes_object = struct.pack('>I', num)
    
    # Unpack the bytes object into a float
    float_number = struct.unpack('>f', bytes_object)[0]
    
    print(float_number)
    
#%%
0XE8
0X03
0X00
0X00