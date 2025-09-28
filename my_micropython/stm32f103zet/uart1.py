from pyb import UART
from pyb import elapsed_micros


uart = UART(1, 9600)
uart.write('hello')

print(elapsed_micros(0))

#print(dir(pyb))