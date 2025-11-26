import cv2
import cv2.aruco as aruco
import numpy as np

# Camera Interface
cap = cv2.VideoCapture(0)

# 选择 ArUco 字典
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 转为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 检测 ArUco 标记
    corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    # 绘制检测到的标记
    if ids is not None:
        aruco.drawDetectedMarkers(frame, corners, ids)

        # 🎯 标注角点
        for i, corner_set in enumerate(corners):
            corner_points = corner_set[0]  # 获取4个角点
            for j, point in enumerate(corner_points):
                x, y = int(point[0]), int(point[1])
                # 1️⃣ 在角点绘制小圆点
                cv2.circle(frame, (x, y), 6, (0, 255, 0), -1)  # 绿色圆点
                # 2️⃣ 标注角点编号
                cv2.putText(frame, f'{j}', (x + 5, y - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                            (255, 0, 0), 2, cv2.LINE_AA)

                # 打印角点坐标信息
                print(f"Marker ID: {ids[i][0]} | Corner {j}: ({x}, {y})")

    # 显示结果
    cv2.imshow("ArUco Corners", frame)

    # 按下 'q' 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
