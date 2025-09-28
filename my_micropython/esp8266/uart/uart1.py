

from machine import UART


# #uart 0 used for repl
# uart = UART(0, baudrate=115200)

# D4, tx ,work
#uart = UART(1, baudrate=115200)

# no uart2
uart = UART(2, baudrate=115200)

uart.write('hello')
#uart.read(5) # read up to 5 bytes

print("end")
