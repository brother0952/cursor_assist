from pyb import CAN
can = CAN(1, CAN.LOOPBACK)
can.setfilter(0, CAN.LIST16, 0, (0x322, 124, 125, 126))  # set a filter to receive messages with id=123, 124, 125 and 126
can.send('message!', 123)   # send a message with id 123
can.recv(0)                 # receive message on FIFO 0

