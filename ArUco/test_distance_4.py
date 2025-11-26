import cv2
import cv2.aruco as aruco
import numpy as np

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920) 
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
cap.set(cv2.CAP_PROP_FOCUS, 40)
cap.set(cv2.CAP_PROP_FPS, 30)

# 加载标定参数
params = np.load("camera_params.npz")
camera_matrix = params["mtx"]
dist_coeffs = params["dist"]

# ArUco 设置
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_50)
parameters = aruco.DetectorParameters()
marker_length = 18.94     # mm

print("空格拍照并测量（使用 IPPE 解算) ESC 退出")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()
    cv2.putText(display, "Press SPACE to capture & measure", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    cv2.imshow("Live", display)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == 32:
        print("拍照中...")

        img = frame.copy()  # 使用原始图像，不做矫正
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if ids is not None:
            aruco.drawDetectedMarkers(img, corners, ids)
            positions = {}
            id_to_index = {}

            for i, marker_id in enumerate(ids.flatten()):
                objp = np.array([
                    [-marker_length / 2,  marker_length / 2, 0],
                    [ marker_length / 2,  marker_length / 2, 0],
                    [ marker_length / 2, -marker_length / 2, 0],
                    [-marker_length / 2, -marker_length / 2, 0]
                ], dtype=np.float32)

                imgp = corners[i][0].astype(np.float32)

                success, rvecs, tvecs, _ = cv2.solvePnPGeneric(
                    objp, imgp, camera_matrix, dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE
                )

                if success:
                    rvec = rvecs[0]
                    tvec = tvecs[0]
                    positions[marker_id] = tvec
                    id_to_index[marker_id] = i

                    cv2.drawFrameAxes(img, camera_matrix, dist_coeffs, rvec, tvec, 0.03)

                    axis = np.float32([[0, 0, 0], [0, 0, 0.05]])
                    imgpts, _ = cv2.projectPoints(axis, rvec, tvec, camera_matrix, dist_coeffs)
                    p2 = tuple(imgpts[1].ravel().astype(int))
                    cv2.circle(img, p2, 5, (255, 0, 0), -1)

                    cX = int(imgp[:, 0].mean())
                    cY = int(imgp[:, 1].mean())
                    cv2.putText(img, f"ID:{marker_id}", (cX, cY - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    print(f"Marker ID {marker_id} solvePnP failed.")

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

                        cv2.putText(img, f"{id1}-{id2}: {distance:.4f}m",
                                    (midX, midY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                        print(f"{id1}-{id2} 距离: {distance:.4f} m")
        else:
            print("未检测到 ArUco")

        cv2.imshow("Snapshot Result", img)
        cv2.waitKey(0)
        cv2.destroyWindow("Snapshot Result")

cap.release()
cv2.destroyAllWindows()
