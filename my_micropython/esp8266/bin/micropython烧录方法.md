

有micropython固件,下载链接
https://micropython.org/download/ESP8266_GENERIC/

类似这个
ESP8266_GENERIC-20170108-v1.8.7.bin



pip install esptool


esptool.py --port /dev/ttyUSB0 erase_flash
python -m esptool --port COM6 erase_flash


esptool.py --port /dev/ttyUSB0 --baud 460800 write_flash --flash_size=detect 0 esp8266-20170108-v1.8.7.bin

esptool.py --port /dev/ttyUSB0 --baud 460800 write_flash --flash_size=detect -fm dout 0 esp8266-20170108-v1.8.7.bin

esptool.py --port COM6 write_flash xx
python -m esptool --port COM6 --baud 460800 write_flash --flash_size=detect 0 xx

确保8266 的供电。可以运行mp

webrepl 可以腾出串口

