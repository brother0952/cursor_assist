def do_connect():
    import network
    
    ssid='HUAWEI-P107NL'
    key='12871034'
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    if not wlan.isconnected():
        print('connecting to network...')
        # wlan.connect('ssid', 'key')
        wlan.connect(ssid, key)
        while not wlan.isconnected():
            print("connected")
            pass
    #print('network config:', wlan.ipconfig('addr4'))
    print(dir(wlan))
    print(wlan.status())
    #print('network config:', wlan.ifconfig('addr4'))
    
do_connect()

#while True:
    #pass

