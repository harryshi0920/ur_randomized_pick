import cv2

# 打开摄像头
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# 设置初始参数
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)          # 自动对焦

# 获取摄像头分辨率
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"width: {width}, height: {height}")

ret, frame = cap.read()
if ret:
    h, w = frame.shape[:2]
    print(f"Image Size: {w}x{h}")

# 设置窗口大小
cv2.namedWindow("Camera Focus Control", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Camera Focus Control", width, height)

print("ESC to Exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 获取当前焦距值
    current_focus = cap.get(cv2.CAP_PROP_FOCUS)
    print(f"Focus: {int(current_focus)}")

    cv2.imshow("Camera Focus Control", frame)

    key = cv2.waitKey(50) & 0xFF    
    # waitKey() 阻塞50ms，如果没反应就继续播放下一帧
    # & 0xFF是做位掩码操作，保证拿到的是8位键码
    if key == 27:  # ESC键退出
        break

cap.release()
cv2.destroyAllWindows()
