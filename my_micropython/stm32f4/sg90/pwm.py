# -*- coding: utf-8 -*-

from pyb import Pin, Timer
import time

p = Pin('B6') # X1 has TIM2, CH1
tim = Timer(4, freq=50)
ch = tim.channel(1, Timer.PWM, pin=p)
ch.pulse_width_percent(10)

percent=0
delta=1
ch.pulse_width_percent(0)
time.sleep(1)
for x in range(100):
    for i in range(22,125):
        ch.pulse_width_percent(i/10)
        time.sleep(0.01)
    for i in reversed(range(22,125)):
        ch.pulse_width_percent(i/10)
        time.sleep(0.01)    
ch.pulse_width_percent(0)

