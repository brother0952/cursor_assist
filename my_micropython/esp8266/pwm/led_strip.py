# -*- coding: utf-8 -*-

from machine import Pin, PWM
import time


# D3
pwm0 = PWM(Pin(0))      # 从1个引脚中创建 PWM 对象
pwm0.freq()             # 获取当前频率
pwm0.freq(1000)         # 设置频率
pwm0.duty()             # 获取当前占空比
#pwm0.duty(200)          # 设置占空比

def set_pwm_percent(per):
    pwm0.duty(int(per*1024/100))

percent=0
delta=1
while True:
    set_pwm_percent(percent)
    pwm0
    percent+=delta
    if percent>=100:
        percent=100
        delta=-1
    elif percent<=10:
        percent=10
        delta=1
    time.sleep(0.01)    
    
    
