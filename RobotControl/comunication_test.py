import socket

HOST = "192.168.65.128"
PORT = 29999  # Dashboard端口

commands = [
    "power on",
    "brake release"
]

try:
    s = socket.create_connection((HOST, PORT), timeout=5)
    
    for cmd in commands:
        s.sendall((cmd + "\n").encode())
        response = s.recv(1024).decode()
        print(f"Sent: {cmd}, Response: {response.strip()}")
    
    s.close()
except Exception as e:
    print("Error:", e)
