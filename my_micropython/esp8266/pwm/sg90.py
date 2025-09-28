# -*- coding: utf-8 -*-

from machine import Pin, PWM
import time


# D3 # if run . disconnect with repl
#pwm0 = PWM(Pin(0))      # 从1个引脚中创建 PWM 对象

# D2 pin4 .work
pwm0 = PWM(Pin(4))      # 从1个引脚中创建 PWM 对象

pwm0.freq()             # 获取当前频率
pwm0.freq(50)         # 设置频率
pwm0.duty()             # 获取当前占空比
#pwm0.duty(200)          # 设置占空比

def set_pwm_percent(per):
    pwm0.duty(int(per*1024/100))

percent=0
delta=1
set_pwm_percent(0)
time.sleep(1)


for x in range(5):
    for i in range(22,125):
        set_pwm_percent(i/10)
        time.sleep(0.01)
    for i in reversed(range(22,125)):
        set_pwm_percent(i/10)
        time.sleep(0.01)
        
set_pwm_percent(0)

