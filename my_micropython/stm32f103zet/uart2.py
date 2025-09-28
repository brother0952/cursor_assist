from pyb import UART
from pyb import elapsed_micros


uart = UART(2, 9600)
uart.write('hello')

print(elapsed_micros(0))

print(dir(pyb))