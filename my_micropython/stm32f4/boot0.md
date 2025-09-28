boot0  connect vdd .
enter dfu

device manager can find it .
then launch dfu.exe


D:\Users\xianyuchao\Downloads\dfu-util-0.6-win32.zip\dfu-util-0.6-win32\dfu-util-0.6

cmd:
# not work
dfu-util.exe STM32F4DISC-20240222-v1.22.2.dfu

# work
D:\Program Files (x86)\STMicroelectronics\Software\DfuSe v3.0.6\Bin
DfuSeDemo.exe
GUI


通用串行总线控制器-》 STM Device in DFU Mode


dfu 烧录micropython
不能用 STM32CubeProgrammer  . 想用，但是登陆不了官网，游客也下载不了。输入账号密码登陆后，重新回到登陆页面， 游客则收不到邮件。
幽默st
