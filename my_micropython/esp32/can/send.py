from machine import CAN

# not work . need special firmware

dev = CAN(0,extframe=False,mode=CAN.SILENT_LOOPBACK,baudrate=500,tx_io=5,rx_io=4,auto_restart=False)

dev.any() # if True ,has data to read

dev.send([1,2,3,4,5,6,7,8],0x322) # 100 is id
dev.recv() # recv tuple
