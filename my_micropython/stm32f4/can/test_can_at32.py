# work
import pyb
import time
from pyb import LED

print("start")

print(pyb.freq())
# 初始化 CAN 和温度传感器
can = pyb.CAN(1,pyb.CAN.NORMAL) # 假设使用 CAN 1 总线
can.init(pyb.CAN.NORMAL, prescaler=4, sjw=1, bs1=14, bs2=6) # 设置通信参数 500K
# maybe 84/8*(21) 500k


can.setfilter(0, pyb.CAN.LIST16, 0, (0x321, 124, 125, 126))


led = LED(1) # 1=red, 2=green, 3=yellow, 4=blue
led2 = LED(2)
led.toggle()
led2.toggle()
time.sleep(1)
led.toggle()
led2.toggle()

a=True

while True:
    try:
        #res=can.recv(0)
        #print(res)
        if a:
            can.send('message!', 0x322)
            a=False
        else:
            can.send(b'm\x00ssage!', 0x322)
            a=True
        #print("sent")
    except Exception as e:
        print(e)
        pass
    time.sleep_ms(200)
    led2.toggle()


print("end")

