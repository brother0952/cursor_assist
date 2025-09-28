import network



wlan=network.WLAN(network.AP_IF)
res=wlan.active(True)
# wlan.config(essid=ssid, password=password)
wlan.config(essid="HEYJUDE")

print("web conf:",wlan.ifconfig())
#print(wlan.config(essid))
