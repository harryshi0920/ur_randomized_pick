import socket
import time

HOST = "192.168.65.128"  # 根据你的URSim实际IP调整
PORT = 30002  # URScript指令端口

def send_urscript_command(script):
    with socket.create_connection((HOST, PORT), timeout=5) as s:
        s.sendall(script.encode())
        print(f"[URScript] Sent command:\n{script}")

if __name__ == "__main__":
    # 示例：MoveJ到一个关节角度位置
    movej_command = """
    def move_example():
        movej([0.0, -1.57, 1.57, 0.0, 1.57, 0.0], a=1.2, v=0.25)
    end
    """
    send_urscript_command(movej_command)

    # 等待机器人运动完成（根据距离长短调整时间）
    time.sleep(3)

    # 示例：MoveL到一个工具坐标位置
    movel_command = """
    def move_example():
        movel(p[0.3, -0.3, 0.3, 0.0, 3.14, 0.0], a=1.2, v=0.25)
    end
    """
    send_urscript_command(movel_command)
