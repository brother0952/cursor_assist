
from machine import Pin, PWM

# D3
pwm0 = PWM(Pin(0))      # 从1个引脚中创建 PWM 对象
pwm0.freq()             # 获取当前频率
pwm0.freq(50)         # 设置频率
pwm0.duty()             # 获取当前占空比
pwm0.duty(200)          # 设置占空比
#pwm0.deinit()           # 关闭引脚的 PWM

#pwm2 = PWM(Pin(2), freq=500, duty=512) # 在同一语句下创建和配置 PWM

