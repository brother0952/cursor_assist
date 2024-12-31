ser = serial.Serial(
    port,
    baudrate,
    timeout=1,
    write_timeout=1,
    # 增加缓冲区大小
    write_buffer_size=65536,
    read_buffer_size=65536,
    # 禁用流控制以减少开销
    xonxoff=False,    # 禁用软件流控
    rtscts=False,     # 禁用硬件流控
    dsrdtr=False      # 禁用DSR/DTR
) 