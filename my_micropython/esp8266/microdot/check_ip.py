import network

sta_if = network.WLAN(network.STA_IF)

print(sta_if.ifconfig()[0])
      
      
#print((sta_if.ifconfig('addr4')[0]))

#sta_if.ipconfig('addr4')