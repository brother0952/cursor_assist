# -*- coding: utf-8 -*-

from pyb import Pin, Timer
import time

p = Pin('B6') # X1 has TIM2, CH1
tim = Timer(4, freq=1000)
ch = tim.channel(1, Timer.PWM, pin=p)
ch.pulse_width_percent(10)

percent=0
delta=1
while True:
    ch.pulse_width_percent(percent)
    percent+=delta
    if percent>=100:
        percent=100
        delta=-1
    elif percent<=10:
        percent=10
        delta=1
    time.sleep(0.01)    
