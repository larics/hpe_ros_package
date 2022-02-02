#!/opt/conda/bin/python3


import serial 

def read_serial(port_name, baud_rate, timeout):

    with serial.Serial(port_name, baud_rate, timeout) as ser:
        line =  ser.readline()
        print(line)


if __name__ == "__main__":

    read_serial("/dev/ttyUSB0", 115200, 1)