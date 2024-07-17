# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 10:31:41 2024

@author: Plutonium
"""

import time
import serial
import threading

port1 = 'COM3'  
port2 = 'COM4'  
baud_rate = 921600
packet_size = 14  # Size of each packet in bytes
packet_interval = 0.001  # Interval between packets in seconds (10 ms)
number_of_packets = 100
# Generate test data to send in each packet
line_ending = b'\n'
test_packet = b'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:packet_size - len(line_ending)] + line_ending

#%%


def sender():
    with serial.Serial(port1, baud_rate, timeout=1) as ser:
        start_time = 0
        for _ in range(number_of_packets):
            ser.write(test_packet)
            ser.flush()
            time_taken = time.perf_counter() - start_time
            # print(f"Sent {len(test_packet)} bytes in {time_taken:.6f} seconds. {1/time_taken:0.6f} Hz")

            start_time = time.perf_counter()
            # sleep_time = max(packet_interval-time_taken, 0)
            time.sleep(packet_interval)
            

# def receiver():
#     with serial.Serial(port2, baud_rate, timeout=0) as ser:
#         received_data = b''
#         start_time = time.perf_counter()
#         while True:
#             if(ser.in_waiting):
#                 data = ser.read(packet_size)
#                 if data:
#                     received_data += data
#                     print(data)
#                     print(f"Received {len(data)} bytes in {time.perf_counter() - start_time:.6f} seconds. {1/(time.perf_counter()-start_time):.6f} Hz")
#                     start_time = time.perf_counter()

def receiver():
    with serial.Serial(port2, baud_rate, timeout=1) as ser:
        ser.reset_input_buffer()
        received_data = b''
        start_time = time.perf_counter()
        prev_data_in_waiting = 0
        while True:
            data_in_waiting = ser.in_waiting
            if(not data_in_waiting == prev_data_in_waiting):
                print(f"data in queue {data_in_waiting}")
                prev_data_in_waiting = data_in_waiting
                dt = time.perf_counter()-start_time
                print(f"freq : {1/dt}")
                start_time = time.perf_counter()
                
            time.sleep(0.001)




#%%
# Start sender and receiver in separate threads
start_time = time.perf_counter()
receiver_thread = threading.Thread(target=receiver)
sender_thread = threading.Thread(target=sender)

receiver_thread.start()
sender_thread.start()


receiver_thread.join()
sender_thread.join()



total_time = time.time() - start_time
print(f"Total time for data transmission: {total_time:.2f} seconds")


