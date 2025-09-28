from pyb import CAN
can = CAN(1, CAN.LOOPBACK)
can.setfilter(0, CAN.LIST16, 0, (123, 124, 125, 126))  # set a filter to receive messages with id=123, 124, 125 and 126
can.send('message!', 123)   # send a message with id 123
can.recv(0)                 # receive message on FIFO 0





can.deinit()
can.restart()
can.state()
can.any(0) # filter 0

can.recv(0,2000)

buf = bytearray(8)
lst = [0, 0, 0, 0, memoryview(buf)]
# No heap memory is allocated in the following call
can.recv(0, lst) # timeout 2000




    
