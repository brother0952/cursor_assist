'''
from pyb import CAN
#can = CAN(1, CAN.LOOPBACK)
can = CAN(1, CAN.NORMAL)
can.setfilter(0, CAN.LIST16, 0, (123, 124, 125, 126))  # set a filter to receive messages with id=123, 124, 125 and 126
can.send(b'\x00\x01ssage!', 0x322)   # send a message with id 123

'''

import pyb
from pyb import CAN
# 初始化 CAN 和温度传感器
can = pyb.CAN(1) # 假设使用 CAN 1 总线
can.init(pyb.CAN.NORMAL, prescaler=4, sjw=1, bs1=14, bs2=6) # 设置通信参数
# maybe 84/8*(21) 500k

    
    
#can.init(pyb.CAN.NORMAL, prescaler=16, sjw=1, bs1=14, bs2=6)
# maybe 250k bps



can.setfilter(0, CAN.LIST16, 0, (0x321, 0x320, 125, 126))

#can.send('\x01\x01ssage!',0x322)
can.send('\x00\x00ssage!',0x322)
