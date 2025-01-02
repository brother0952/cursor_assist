baud_rates = [
    "9600", "115200", "256000", "460800", "921600",
    "1000000", "1500000", "2000000", "3000000"
]
self.baud_box = ttk.Combobox(baud_row, textvariable=self.baud_var, 
                            values=baud_rates,
                            width=15) 