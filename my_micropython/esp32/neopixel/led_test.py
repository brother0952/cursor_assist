

from machine import Pin
import neopixel
import time


led=Pin(38,Pin.OUT) # pin 38
np=neopixel.NeoPixel(led,1)


def lightOn():
    np[0]=(10,10,20)
    np.write()

def lightOff():
    np[0]=(0,0,0)
    np.write()

lightOn()
time.sleep_ms(500)
lightOff()
time.sleep_ms(500)
lightOn()
time.sleep_ms(500)
lightOff()


