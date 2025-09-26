import neopixel
import time
from machine import Pin

import random

np=neopixel.NeoPixel(Pin(2),1)# pin38

np[0]=(0,0,0)
np.write()


np2=neopixel.NeoPixel(Pin(2),256)# pin6

np2[0]=(0,0,0)
np2.write()

