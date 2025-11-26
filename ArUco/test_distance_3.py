import cv2
import cv2.aruco as aruco
import numpy as np

# 摄像头初始化
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
cap.set(cv2.CAP_PROP_FOCUS, 40)
cap.set(cv2.CAP_PROP_FPS, 30)

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
    elif key == 32:  # 空格
        print("拍照中...")

        # 拍照并做矫正
        raw_img = frame.copy()
        undistorted = cv2.undistort(raw_img, camera_matrix, dist_coeffs, None)

        gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if ids is not None:
            aruco.drawDetectedMarkers(undistorted, corners, ids)
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                corners, marker_length, camera_matrix, dist_coeffs)

            id_to_index = {int(ids[i][0]): i for i in range(len(ids))}
            positions = {}

            for i, marker_id in enumerate(ids.flatten()):
                positions[marker_id] = tvecs[i][0]
                cv2.drawFrameAxes(undistorted, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.03)

                # Z 轴方向蓝点
                axis = np.float32([[0, 0, 0], [0, 0, 0.05]])
                imgpts, _ = cv2.projectPoints(axis, rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
                p2 = tuple(imgpts[1].ravel().astype(int))
                cv2.circle(undistorted, p2, 5, (255, 0, 0), -1)

                # ID 标记
                cX = int(corners[i][0][:, 0].mean())
                cY = int(corners[i][0][:, 1].mean())
                cv2.putText(undistorted, f"ID:{marker_id}", (cX, cY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 距离计算
            if len(positions) >= 2:
                ids_list = list(positions.keys())
                for i in range(len(ids_list)):
                    for j in range(i + 1, len(ids_list)):
                        id1, id2 = ids_list[i], ids_list[j]
                        pos1, pos2 = positions[id1], positions[id2]
                        distance = np.linalg.norm(pos1 - pos2)

                        index1 = id_to_index[id1]
                        index2 = id_to_index[id2]
                        midX = int((corners[index1][0][:, 0].mean() + corners[index2][0][:, 0].mean()) / 2)
                        midY = int((corners[index1][0][:, 1].mean() + corners[index2][0][:, 1].mean()) / 2)

                        cv2.putText(undistorted, f"{id1}-{id2} dist: {distance:.3f} m",
                                    (midX, midY),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                        print(f"{id1}-{id2} 距离: {distance:.3f} 米")
        else:
            print("⚠️ 未检测到 ArUco")

        cv2.imshow("Snapshot Result", undistorted)
        cv2.waitKey(0)
        cv2.destroyWindow("Snapshot Result")

cap.release()
cv2.destroyAllWindows()
