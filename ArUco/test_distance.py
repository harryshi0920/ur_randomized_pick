import cv2
import cv2.aruco as aruco
import numpy as np

# 摄像头接口
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # 打开默认摄像头
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)          # 自动对焦
cap.set(cv2.CAP_PROP_FOCUS, 40)             # 焦距设置为40

# 选择 ArUco 字典
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()

# 相机内参 (需使用相机标定结果替换)
"""
camera_matrix = np.array([[1000, 0, 640],
                          [0, 1000, 360],
                          [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((5, 1))  # 默认假设无畸变（需要标定相机）
"""
# 从 .npz 文件加载相机内参和畸变系数
params = np.load("camera_params.npz")
camera_matrix = params["mtx"]
dist_coeffs = params["dist"]

# ArUco 标记实际边长 (单位：米)
marker_length = 0.05  # 5cm

# 用于存储标记的位置信息和角度信息
aruco_positions = {}
aruco_angles = {}  # 存储XY平面角度信息

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # BGR 转 灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 检测 ArUco 标记
    corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        aruco.drawDetectedMarkers(frame, corners, ids)

        # 估计姿态
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)

        # 保存每个 ArUco 的位置和角度
        id_to_index = {int(ids[i][0]): i for i in range(len(ids))}
        for i, marker_id in enumerate(ids.flatten()):
            # 绘制坐标轴
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.03)

            # 保存位置信息
            aruco_positions[marker_id] = tvecs[i][0]

            # 计算角度（XY 平面 Yaw 角度）
            R, _ = cv2.Rodrigues(rvecs[i])  # 旋转向量转换为旋转矩阵
            yaw_angle = np.degrees(np.arctan2(R[1, 0], R[0, 0]))  # 提取XY平面的偏航角
            aruco_angles[marker_id] = yaw_angle

            # 显示 ID
            cX = int(corners[i][0][:, 0].mean())
            cY = int(corners[i][0][:, 1].mean())
            cv2.putText(frame, f"ID:{marker_id} {yaw_angle:.1f} deg", (cX, cY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 计算并显示两个 ArUco 之间的距离 & 角度差
        if len(aruco_positions) >= 2:
            ids_list = list(aruco_positions.keys())

            for i in range(len(ids_list)):
                for j in range(i + 1, len(ids_list)):
                    id1, id2 = ids_list[i], ids_list[j]

                    # 计算欧氏距离
                    pos1, pos2 = aruco_positions[id1], aruco_positions[id2]
                    distance = np.linalg.norm(pos1 - pos2)

                    # 计算角度差
                    angle_diff = abs(aruco_angles[id1] - aruco_angles[id2])

                    # 根据 ID 找到角点，计算标记之间的中点
                    index1 = id_to_index[id1]
                    index2 = id_to_index[id2]

                    midX = int((corners[index1][0][:, 0].mean() + corners[index2][0][:, 0].mean()) / 2)
                    midY = int((corners[index1][0][:, 1].mean() + corners[index2][0][:, 1].mean()) / 2)

                    # 显示距离 & 角度差
                    cv2.putText(frame, f"Dist {id1}-{id2}: {distance:.3f} m",
                                (midX, midY),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    
                    cv2.putText(frame, f"Angle {id1}-{id2}: {angle_diff:.1f} deg",
                                (midX, midY + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

                    print(f"Distance between {id1} and {id2}: {distance:.3f} meters")
                    print(f"Angle difference (XY plane) between {id1} and {id2}: {angle_diff:.1f} degrees")

    cv2.imshow("ArUco Distance & Angle Measurement", frame)

    # 按下 'q' 退出
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
