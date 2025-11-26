import cv2
import cv2.aruco as aruco
import numpy as np

# 摄像头初始化
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
cap.set(cv2.CAP_PROP_FOCUS, 40)
cap.set(cv2.CAP_PROP_FPS, 30)



# 定义相机内参和畸变系数 (需要标定相机)
"""camera_matrix = np.array([[1000, 0, 640],
                          [0, 1000, 360],
                          [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((5, 1))  # 默认假设无畸变（需要标定相机）"""



# 定义相机内参和畸变系数 (需要标定相机)
"""camera_matrix = np.array([
    [627.32911975,     0.0,            309.07601879],
    [0.0,              627.96552459,   230.05588573],
    [0.0,              0.0,            1.0]
], dtype=np.float32)

dist_coeffs = np.array(
    [4.91158427e-02, -1.99719913e-01, -1.66224400e-04, -3.31318376e-03, 1.47963074e-01],
    dtype=np.float32
)   
"""

# 加载相机内参
params = np.load("camera_params.npz")
camera_matrix = params["mtx"]
dist_coeffs = params["dist"]

# ArUco 字典与参数
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()

marker_length = 0.0495  # 单位：米

print("按空格键拍照进行距离测量, ESC退出")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()
    cv2.putText(display, "Press SPACE to capture & measure", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("Live", display)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC退出
        break
    elif key == 32:  # SPACE拍照
        print("拍照中...")
        img = frame.copy()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if ids is not None:
            aruco.drawDetectedMarkers(img, corners, ids)
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                corners, marker_length, camera_matrix, dist_coeffs)

            # ID到索引映射
            id_to_index = {int(ids[i][0]): i for i in range(len(ids))}
            positions = {}

            for i, marker_id in enumerate(ids.flatten()):
                positions[marker_id] = tvecs[i][0]
                cv2.drawFrameAxes(img, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.03)

                # 画 Z 轴方向端点的小蓝点
                axis = np.float32([[0, 0, 0], [0, 0, 0.05]])  # 起点+Z轴方向点（单位：米）
                imgpts, _ = cv2.projectPoints(axis, rvecs[i], tvecs[i], camera_matrix, dist_coeffs)

                p1 = tuple(imgpts[0].ravel().astype(int))   # 起点（就是标记中心）
                p2 = tuple(imgpts[1].ravel().astype(int))   # Z轴方向的终点
                cv2.circle(img, p2, 5, (255, 0, 0), -1)     # 蓝色圆点 = Z 轴方向终点

                cX = int(corners[i][0][:, 0].mean())
                cY = int(corners[i][0][:, 1].mean())
                cv2.putText(img, f"ID:{marker_id}", (cX, cY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 计算两两距离
            if len(positions) >= 2:
                ids_list = list(positions.keys())
                for i in range(len(ids_list)):
                    for j in range(i + 1, len(ids_list)):
                        id1, id2 = ids_list[i], ids_list[j]
                        pos1, pos2 = positions[id1], positions[id2]
                        distance = np.linalg.norm(pos1 - pos2)

                        # 显示在中点
                        index1 = id_to_index[id1]
                        index2 = id_to_index[id2]
                        midX = int((corners[index1][0][:, 0].mean() + corners[index2][0][:, 0].mean()) / 2)
                        midY = int((corners[index1][0][:, 1].mean() + corners[index2][0][:, 1].mean()) / 2)

                        cv2.putText(img, f"{id1}-{id2} dist: {distance:.5f} m",
                                    (midX, midY),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                        print(f"{id1}-{id2} 距离: {distance:.5f} 米")
        else:
            print("未检测到 ArUco")

        # 展示分析结果
        cv2.imshow("Snapshot Result", img)
        cv2.waitKey(0)  # 等待任意键关闭结果图像
        cv2.destroyWindow("Snapshot Result")

cap.release()
cv2.destroyAllWindows()
