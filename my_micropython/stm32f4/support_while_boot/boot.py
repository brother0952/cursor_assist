from machine import Pin
import os
 
btn = Pin('PC15', mode=Pin.IN_PULLUP)
 
if 0==btn():
    print('skip main.py')
else:
    if 'main.py' in os.listdir():
        # import main
        print("run main")
        pass
    else:
        print('no main.py in filesystem')
 
 