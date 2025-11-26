import time
import socket

HOST = "192.168.56.101"
PORT = 30002

cmd = "set_digital_out(0, True)\n"

try:
    s = socket.create_connection((HOST, PORT), timeout=5)
    s.sendall(cmd.encode("utf-8"))
    time.sleep(0.5)  # 等待命令执行
    s.close()
    print("Sent digital output command")
except Exception as e:
    print("Error:", e)
