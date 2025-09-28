import machine, neopixel

# gpio4 "D2"
np = neopixel.NeoPixel(machine.Pin(4), 8, bpp=4)

# gpio0 "D3"
# np = neopixel.NeoPixel(machine.Pin(0), 10, bpp=4)

np[0] = (255, 0, 0, 128) # Orange in an RGBY Setup
np[1] = (0, 255, 0, 128) # Yellow-green in an RGBY Setup
np[2] = (0, 0, 255, 128) # Green-blue in an RGBY Setup
np[3] = (255, 255, 255, 128) # Green-blue in an RGBY Setup

np.write()
