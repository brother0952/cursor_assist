
from pyb import CAN
import time

can=CAN(1,mode=CAN.NORMAL,baudrate=500000)
#can.setfilter(0, CAN.LIST16, 0, (0x321,0x320,))
#can.setfilter(0, CAN.LIST16, 0, (801,800)) # 

while True:
    can.send(b'\x00\x01ssage!', 0x322)
    res=can.recv(0)
    print(res)
    time.sleep(0.1)
    
    break
    
print("end")
