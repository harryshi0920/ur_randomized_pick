import cv2
import cv2.aruco as aruco
import numpy as np

# 摄像头接口
cap = cv2.VideoCapture(0)

# 选择 ArUco 字典
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()

# ✅ 定义相机内参和畸变系数 (你需要修改为你的相机标定数据)
camera_matrix = np.array([[1000, 0, 640],
                          [0, 1000, 360],
                          [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((5, 1))  # 假设无畸变，可换成你的相机畸变参数

# ✅ 定义 ArUco 标记的世界坐标 (单位：米)
marker_length = 0.05  # 5cm 的 ArUco 标记

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # BGR 转 灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 检测 ArUco 标记
    corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    # 绘制 ArUco 标记
    if ids is not None:
        aruco.drawDetectedMarkers(frame, corners, ids)

        # 估计姿态 (获取旋转和平移矩阵)
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)

        for i in range(len(ids)):
            # ✅ 画 3D 坐标轴
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.05)

            # ✅ 显示 ID 和坐标 (标记中心点)
            cX, cY = int(corners[i][0][:, 0].mean()), int(corners[i][0][:, 1].mean())
            cv2.putText(frame, f"ID:{ids[i][0]}", (cX, cY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            print(f"Marker ID: {ids[i][0]}, Rotation Vector: {rvecs[i].flatten()}, Translation Vector: {tvecs[i].flatten()}")

    cv2.imshow("ArUco Detection with Axes", frame)

    # 按下 'q' 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
