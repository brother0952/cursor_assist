# neo pixel

import machine, neopixel
np = neopixel.NeoPixel(machine.Pin(4), 8)
np[0] = (30, 0, 0) # set to red, full brightness
#np[1] = (0, 128, 0) # set to green, half brightness
#np[2] = (0, 0, 64)  # set to blue, quarter brightness


np[5] = (0, 0, 30)

print("end")

np.write()


