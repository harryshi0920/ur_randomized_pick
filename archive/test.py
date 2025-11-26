import cv2
import cv2.aruco as aruco
import numpy as np

# 打开摄像头
cap = cv2.VideoCapture(0)

# 设置摄像头分辨率 (可选，提高检测效果)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# 选择 ArUco 字典
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
parameters = aruco.DetectorParameters()

# 用于存储定格画面
snapshot = None
captured = False
printed_once = False  # 控制输出标志位

def detect_aruco_and_draw(img):
    """检测 ArUco 标记并在图像上绘制标注"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        # 绘制检测结果
        aruco.drawDetectedMarkers(img, corners, ids)
        for i, marker_id in enumerate(ids):
            corner = corners[i][0]
            # 在 ArUco 标记中心绘制 ID
            center = tuple(np.mean(corner, axis=0).astype(int))
            cv2.putText(img, f"ID: {marker_id[0]}", center, 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return img, True
    else:
        # 未检测到标记时显示提示
        cv2.putText(img, "No ArUco marker detected", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        return img, False

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法读取摄像头画面")
        break

    # 如果按下 "S" 键，则检测定格画面
    if captured and snapshot is not None:
        display_frame, detected = detect_aruco_and_draw(snapshot.copy())

        # 仅在第一次检测结果时打印一次
        if not printed_once:
            if detected:
                print("✅ ArUco marker detected in snapshot!")
            else:
                print("❌ No ArUco marker detected in snapshot.")
            printed_once = True

        cv2.imshow("Snapshot - ArUco Detection", display_frame)
    
    else:
        # 实时显示摄像头画面
        display_frame, _ = detect_aruco_and_draw(frame.copy())
        cv2.imshow("Live ArUco Detection", display_frame)

    # 按键检测
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        snapshot = frame.copy()
        captured = True
        printed_once = False  # 每次拍摄重新允许输出
        print("📸 已拍摄定格动画！")

cap.release()
cv2.destroyAllWindows()