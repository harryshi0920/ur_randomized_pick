import cv2
import cv2.aruco as aruco
import numpy as np
import socket
import math
import time

def rigid_transform_3D(A, B):
    assert A.shape == B.shape           # A & B are same shape
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B
    H = AA.T @ BB
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
    t = centroid_B.T - R @ centroid_A.T

    return R, t


# Network Global Variable
HOST = "192.168.1.50" 
PORT = 30002

def send_urscript_command(script):
    with socket.create_connection((HOST, PORT), timeout=5) as s:
        s.sendall(script.encode())
        print(f"[URScript] Sent:\n{script}")

# UrScript Global Variable
feature_pose = [0.43882, 0.09855, -0.04081, 2.509, 1.912, 0.009]
pre_jig = [-0.01439, 0.35937, -0.40379, 0, 0, 6.101]
pre_jig_j = [0.1117, -1.6104, -1.6047, -1.4959, 1.5818, -2.7222]
pre_asmb = [0.40168, -0.41071, -0.63009, 0.000, 0.000, 0.034]
pre_asmb_j = [1.6116, -1.3966, -1.5385, -1.7667, 1.5731, -1.2224]
camera = [0.29176, 0.0823, -0.51551, 0, 0, 1.568]
camera_j = [0.7775, -1.5258, -1.6301, -1.5493, 1.5799, -0.5224]
feature_jig = [0.6181, -0.2326, 0.05606, 3.147, 0.506, 4.144]

def ur_move(id, marker_pose):
    door_rf = f"""
def move_to_marker_absolute():
  FeaturePose = p[{', '.join(f'{v:.5f}' for v in feature_pose)}]
  MarkerPose = p[{', '.join(f'{v:.5f}' for v in marker_pose)}]
  PreJig = [{', '.join(f'{v:.5f}' for v in pre_jig_j)}]
  PreAsmb = [{', '.join(f'{v:.5f}' for v in pre_asmb_j)}]
  Camera = [{', '.join(f'{v:.5f}' for v in camera_j)}]
  FeatureJig = p[{', '.join(f'{v:.5f}' for v in feature_jig)}]

  movel(pose_trans(FeaturePose, MarkerPose), a=1.5, v=2)
  sync()
  set_digital_out(0, True)
  MarkerPick = pose_add(MarkerPose, p[0, 0, 0.033, 0, 0, 0])
  movel(pose_trans(FeaturePose, MarkerPick), a=0.8, v=0.15)
  sync()

  movej(PreJig, a=1.5, v=6)
  sync()
  Jig = p[0.05349, 0.08194, 0.04597, 2.118, -2.248, 0.065]
  movel(pose_trans(FeatureJig, Jig), a=0.8, v=0.25)
  sync()
  set_digital_out(0, False)
  movej(PreJig, a=1.5, v=6)
  sync()
  PostJig = p[0.04963, 0.1492, 0.04391, 2.232, -2.117, 0.115]
  movel(pose_trans(FeatureJig, PostJig), a=0.8, v=0.25)
  set_digital_out(0, True)
  sync()
  movej(PreJig, a=1.5, v=6)
  sync()

  movej(PreAsmb, a=1.5, v=6)
  sync()
  Aiming1 = p[0.4321, 0.29484, 0.34605, 1.05, 2.971, -0.003]
  movel(Aiming1, a=0.8, v=1)
  sync()
  Aiming2 = [0.8877, -1.8604, -2.3349, -1.2851, 3.1304, -3.9427]
  movej(Aiming2, a=0.8, v=0.25)
  sync()
  Assem = [0.9397, -1.9131, -2.2236, -1.7108, 3.1807, -4.3611]
  movej(Assem, a=0.8, v=0.25)
  sync()
  set_digital_out(0, False)
  movej (Aiming2, a=1.5, v=6)
  sync()
  movel(Aiming1, a=0.8, v=0.1)
  sync()
  movej(Camera, a=1.5, v=6)
  sync()
end
"""


    door_rr = f"""
def move_to_marker_absolute():
  FeaturePose = p[{', '.join(f'{v:.5f}' for v in feature_pose)}]
  MarkerPose = p[{', '.join(f'{v:.5f}' for v in marker_pose)}]
  PreJig = [{', '.join(f'{v:.5f}' for v in pre_jig_j)}]
  PreAsmb = [{', '.join(f'{v:.5f}' for v in pre_asmb_j)}]
  Camera = [{', '.join(f'{v:.5f}' for v in camera_j)}]
  FeatureJig = p[{', '.join(f'{v:.5f}' for v in feature_jig)}]

  movel(pose_trans(FeaturePose, MarkerPose), a=0.8, v=2)
  sync()
  set_digital_out(0, True)
  MarkerPick = pose_add(MarkerPose, p[0, 0, 0.033, 0, 0, 0])
  movel(pose_trans(FeaturePose, MarkerPick), a=0.8, v=0.15)
  sync()

   movej(PreJig, a=1.5, v=6)
  sync()
  Jig = p[0.06956, 0.14059, 0.04354, 2.222, -2.156, 0.076]
  movel(pose_trans(FeatureJig, Jig), a=0.8, v=0.25)
  sync()
  set_digital_out(0, False)
   movej(PreJig, a=1.5, v=6)
  sync()
  PostJig = p[0.02833, 0.14571, 0.0429, 2.276, -2.071, 0.245]
  movel(pose_trans(FeatureJig, PostJig), a=0.8, v=0.25)
  set_digital_out(0, True)
  sync()
   movej(PreJig, a=1.5, v=6)
  sync()

  movej(PreAsmb, a=1.5, v=6)
  sync()
  Aiming1 = p[0.32341, 0.29302, 0.35527, 3.192, 1.121, 0.003]
  movel(Aiming1, a=0.8, v=1)
  sync()
  Aiming2 = [0.9249, -1.8534, -2.6389, -0.1086, 3.011, -3.055]
  movej(Aiming2, a=0.8, v=0.3)
  sync()
  Assem = [0.9678, -2.0021, -2.4878, -0.2983, 3.1639, -3.2767]
  movej(Assem, a=0.8, v=0.15)
  sync()
  set_digital_out(0, False)
  movej(Aiming2, a=1.5, v=6)
  sync()
  movej(Camera, a=1.5, v=6)
  sync()
end
"""


    door_lf = f"""
def move_to_marker_absolute():
  FeaturePose = p[{', '.join(f'{v:.5f}' for v in feature_pose)}]
  MarkerPose = p[{', '.join(f'{v:.5f}' for v in marker_pose)}]
  PreJig = [{', '.join(f'{v:.5f}' for v in pre_jig_j)}]
  PreAsmb = [{', '.join(f'{v:.5f}' for v in pre_asmb_j)}]
  Camera = [{', '.join(f'{v:.5f}' for v in camera_j)}]
  FeatureJig = p[{', '.join(f'{v:.5f}' for v in feature_jig)}]

  movel(pose_trans(FeaturePose, MarkerPose), a=0.8, v=0.15)
  sync()
  set_digital_out(0, True)
  MarkerPick = pose_add(MarkerPose, p[0, 0, 0.035, 0, 0, 0])
  movel(pose_trans(FeaturePose, MarkerPick), a=0.8, v=0.15)
  sync()

  movej(PreJig, a=0.8, v=0.8)
  sync()
  Jig = p[0.03249, 0.10050, 0.03899, 3.151, -0.381, 0.253]
  movel(pose_trans(FeatureJig, Jig), a=0.8, v=0.15)
  sync()
  set_digital_out(0, False)
  movej(PreJig, a=0.8, v=0.8)
  sync()
  PostJig = p[0.02649, 0.11945, 0.04039, 2.065, 2.152, 0.107]
  movel(pose_trans(FeatureJig, PostJig), a=0.8, v=0.15)
  set_digital_out(0, True)
  sync()
  movej(PreJig, a=0.8, v=0.8)
  sync()

  movej(PreAsmb, a=0.8, v=0.8)
  sync()
  Aiming = p[0.04949, 0.69614, 0.20413, 1.517, 0.379, 0.308]
  movel(Aiming, a=0.8, v=0.15)
  sync()
  Assem = p[0.14996, 0.66054, 0.18089, 1.482, 0.469, 0.397]
  movel(Assem, a=0.8, v=0.15)
  sync()
  set_digital_out(0, False)
  movel(Aiming, a=0.8, v=0.15)
  sync()
  movej(PreAsmb, a=0.8, v=0.8)
  sync()
  movej(Camera, a=0.8, v=0.8)
  sync()
end
"""
    


    door_lr = f"""
def move_to_marker_absolute():
  FeaturePose = p[{', '.join(f'{v:.5f}' for v in feature_pose)}]
  MarkerPose = p[{', '.join(f'{v:.5f}' for v in marker_pose)}]
  PreJig = [{', '.join(f'{v:.5f}' for v in pre_jig_j)}]
  PreAsmb = [{', '.join(f'{v:.5f}' for v in pre_asmb_j)}]
  Camera = [{', '.join(f'{v:.5f}' for v in camera_j)}]
  FeatureJig = p[{', '.join(f'{v:.5f}' for v in feature_jig)}]

  movel(pose_trans(FeaturePose, MarkerPose), a=0.8, v=0.15)
  sync()
  set_digital_out(0, True)
  MarkerPick = pose_add(MarkerPose, p[0, 0, 0.033, 0, 0, 0])
  movel(pose_trans(FeaturePose, MarkerPick), a=0.8, v=0.15)
  sync()

  movej(PreJig, a=0.8, v=0.8)
  sync()
  Jig = p[0.03250, 0.10049, 0.03896, 3.151, -0.381, 0.253]
  movel(pose_trans(FeatureJig, Jig), a=0.8, v=0.15)
  sync()
  set_digital_out(0, False)
  movej(PreJig, a=0.8, v=0.8)
  sync()
  PostJig = p[0.0287, 0.14, 0.042, 0.017, -3.176, 0.014]
  movel(pose_trans(FeatureJig, PostJig), a=0.8, v=0.15)
  set_digital_out(0, True)
  sync()
  movej(PreJig, a=0.8, v=0.8)
  sync()

  movej(PreAsmb, a=0.8, v=0.8)
  sync()
  Aiming = p[0.004, 0.58637, 0.19186, 0.779, 1.389, -0.863]
  movel(Aiming, a=0.8, v=0.15)
  sync()
  Assem = p[0.08191, 0.60508, 0.17883, 0.832, 1.367, -0.763]
  movel(Assem, a=0.8, v=0.15)
  set_digital_out(0, False)
  movel(Aiming, a=0.8, v=0.15)
  sync()
  movej(PreAsmb, a=0.8, v=0.8)
  sync()
  movej(Camera, a=0.8, v=0.3)
  sync()
end
"""
    

    
    hood = f"""
def move_to_marker_absolute():
  FeaturePose = p[{', '.join(f'{v:.5f}' for v in feature_pose)}]
  MarkerPose = p[{', '.join(f'{v:.5f}' for v in marker_pose)}]
  PreJig = [{', '.join(f'{v:.5f}' for v in pre_jig_j)}]
  PreAsmb = [{', '.join(f'{v:.5f}' for v in pre_asmb_j)}]
  Camera = [{', '.join(f'{v:.5f}' for v in camera_j)}]
  FeatureJig = p[{', '.join(f'{v:.5f}' for v in feature_jig)}]

  movel(pose_trans(FeaturePose, MarkerPose), a=0.8, v=0.15)
  sync()
  set_digital_out(0, True)
  MarkerPick = pose_add(MarkerPose, p[0, 0, 0.038, 0, 0, 0])
  movel(pose_trans(FeaturePose, MarkerPick), a=0.8, v=0.15)
  sync()

  movej(PreJig, a=0.8, v=0.8)
  sync()
  Jig = p[0.09326, 0.10878, 0.03491, 2.272, -2.120, 0.041]
  movel(pose_trans(FeatureJig, Jig), a=0.8, v=0.15) 
  sync()
  set_digital_out(0, False)
  movej(PreJig, a=0.8, v=0.8)
  sync()
  PostJig = p[0.07149, 0.14172, 0.03506, 2.338, -2.251, -0.047]
  movel(pose_trans(FeatureJig, PostJig), a=0.8, v=0.15)
  set_digital_out(0, True)
  sync()
  movej(PreJig, a=0.8, v=0.8)
  sync()

  movej(PreAsmb, a=0.8, v=0.8)
  sync()
  Aiming = p[0.30662, 0.69531, 0.24299, 2.875, 0.939, -0.374]
  movel(Aiming, a=0.8, v=0.15)
  sync()
  Assem = p[0.28478, 0.67123, 0.1861, 2.9, 1.029, -0.363]
  movel(Assem, a=0.8, v=0.15)
  sync()
  set_digital_out(0, False)
  movel(Aiming, a=0.8, v=0.15)
  sync()
  movej(PreAsmb, a=0.8, v=0.8)
  sync()
  movej(Camera, a=0.8, v=0.8)
  sync()
end
"""
    
    roof = f"""
def move_to_marker_absolute():
  FeaturePose = p[{', '.join(f'{v:.5f}' for v in feature_pose)}]
  MarkerPose = p[{', '.join(f'{v:.5f}' for v in marker_pose)}]
  PreJig = [{', '.join(f'{v:.5f}' for v in pre_jig_j)}]
  PreAsmb = [{', '.join(f'{v:.5f}' for v in pre_asmb_j)}]
  Camera = [{', '.join(f'{v:.5f}' for v in camera_j)}]
  FeatureJig = p[{', '.join(f'{v:.5f}' for v in feature_jig)}]

  movel(pose_trans(FeaturePose, MarkerPose), a=0.8, v=0.15)
  sync()
  set_digital_out(0, True)
  MarkerPick = pose_add(MarkerPose, p[0, 0, 0.038, 0, 0, 0])
  movel(pose_trans(FeaturePose, MarkerPick), a=0.8, v=0.15)
  sync()

  movej(PreJig, a=0.8, v=0.8)
  sync()
  Jig = p[0.08233, 0.08245, 0.03884, 0.139, 3.184, 0.045]
  movel(pose_trans(FeatureJig, Jig), a=0.8, v=0.15) 
  sync()
  set_digital_out(0, False)
  movej(PreJig, a=0.8, v=0.8)
  sync()
  PostJig = p[0.04326, 0.09634, 0.03363, 3.118, 0.049, 0.007]
  movel(pose_trans(FeatureJig, PostJig), a=0.8, v=0.15)
  set_digital_out(0, True)
  sync()
  PostJigClear = p[0.09895, 0.07753, 0.10152, 3.223, -0.104, 0.115]
  movel(pose_trans(FeatureJig, PostJigClear), a=0.8, v=0.15)
  sync()
  movej(PreJig, a=0.8, v=0.8)
  sync()

  movej(PreAsmb, a=0.8, v=0.8)
  sync()
  Aiming = [1.6645, -1.3628, -2.1464, -1.1655, 1.5350, 0.9109]
  movej(Aiming, a=0.8, v=0.8)
  sync()
  Assem = [1.6645, -1.3989, -2.2003, -1.0752, 1.535, 0.9062]
  movej(Assem, a=0.8, v=0.15)
  sync()
  set_digital_out(0, False)
  movej(Aiming, a=0.8, v=0.15)
  sync()
  movej(PreAsmb, a=0.8, v=0.8)
  sync()
  movej(Camera, a=0.8, v=0.8)
  sync()
end
"""

    trunk = f"""
def move_to_marker_absolute():
  FeaturePose = p[{', '.join(f'{v:.5f}' for v in feature_pose)}]
  MarkerPose = p[{', '.join(f'{v:.5f}' for v in marker_pose)}]
  PreJig = [{', '.join(f'{v:.5f}' for v in pre_jig_j)}]
  PreAsmb = [{', '.join(f'{v:.5f}' for v in pre_asmb_j)}]
  Camera = [{', '.join(f'{v:.5f}' for v in camera_j)}]
  FeatureJig = p[{', '.join(f'{v:.5f}' for v in feature_jig)}]

  movel(pose_trans(FeaturePose, MarkerPose), a=0.8, v=0.15)
  sync()
  set_digital_out(0, True)
  MarkerPick = pose_add(MarkerPose, p[0, 0, 0.042, 0, -5.9, 0])
  movel(pose_trans(FeaturePose, MarkerPick), a=0.8, v=0.15)
  sync()

  movej(PreJig, a=0.8, v=0.8)
  sync()
  Jig = p[0.0642, 0.10129, 0.03127, 2.438, 2.41, 0.619]
  movel(pose_trans(FeatureJig, Jig), a=0.8, v=0.15) 
  sync()
  set_digital_out(0, False)
  movej(PreJig, a=0.8, v=0.8)
  sync()
  PostJig = p[0.05381, 0.11281, 0.02951, 2.36, 2.403, 0.379]
  movel(pose_trans(FeatureJig, PostJig), a=0.8, v=0.15)
  set_digital_out(0, True)
  sync()
  movej(PreJig, a=0.8, v=0.8)
  sync()

  movej(PreAsmb, a=0.8, v=0.8)
  sync()
  Aiming = [1.8295, -1.247, -2.2361, -1.24, 1.5865, -3.6062]
  movej(Aiming, a=0.8, v=0.8)
  sync()
  Assem = [1.8413, -1.2882, -2.3606, -1.0538, 1.5718, -3.5331]
  movej(Assem, a=0.8, v=0.8)
  sync()
  set_digital_out(0, False)
  movej(Aiming, a=0.8, v=0.8)
  sync()
  movej(PreAsmb, a=0.8, v=0.8)
  sync()
  movej(Camera, a=0.8, v=0.3)
  sync()
end
"""
    if id == 3:
        send_urscript_command(door_rf)
    if id == 4:
        send_urscript_command(door_rr)
    if id == 5:
        send_urscript_command(door_lf)
    if id == 6:
        send_urscript_command(door_lr)
    if id == 7:
        send_urscript_command(hood)
    if id == 8:
        send_urscript_command(trunk)
    if id == 9:
        send_urscript_command(roof)



# Camera Global Variable
desktop_coords = {
    0: np.array([0, 0, 0], dtype=np.float32),
    1: np.array([139.2, 0, 0], dtype=np.float32),
    2: np.array([0, 127, 0], dtype=np.float32),
}
params = np.load("camera_params.npz")       # Camera Calibration Template
camera_matrix = params["mtx"]
dist_coeffs = params["dist"]



def distance_measure(cap):
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_50)
    parameters = aruco.DetectorParameters()
    marker_length = 20  # mm

    ret, frame = cap.read()
    if not ret:
        return

    img = frame.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is None:
        print("No ArUco Markers Detected")
        return

    aruco.drawDetectedMarkers(img, corners, ids)
    positions, rotations, id_to_index = {}, {}, {}

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
            flags=cv2.SOLVEPNP_ITERATIVE)

        if success:
            rvec, tvec = rvecs[0], tvecs[0]
            positions[int(marker_id)] = tvec
            rotations[int(marker_id)] = rvec
            id_to_index[int(marker_id)] = i

    if not all(k in positions for k in [0, 1, 2]):
        print("[WARNING] Missing ID=0,1,2: Cannot compute transform")
        return

    cam_pts = [positions[k].reshape(3) for k in [0, 1, 2]]
    desk_pts = [desktop_coords[k] for k in [0, 1, 2]]
    R, t = rigid_transform_3D(np.array(cam_pts), np.array(desk_pts))

    marker_pose_dict = {}

    for marker_id, tvec in positions.items():
        if marker_id in [0, 1, 2]:
            continue

        T = tvec.reshape(3)
        T_desk = R @ T + t
        dx, dy, dz = T_desk.flatten()

        R_marker_cam, _ = cv2.Rodrigues(rotations[marker_id])
        R0_cam, _ = cv2.Rodrigues(rotations[0])
        R_marker_desk = R @ R_marker_cam
        R0_desk = R @ R0_cam
        R_rel = R0_desk.T @ R_marker_desk
        yaw_rad = np.arctan2(R_rel[1, 0], R_rel[0, 0])

        rz_base = 1.451
        rz_final = (rz_base + yaw_rad) % (2 * math.pi)

        # print(f"Marker {marker_id} in Desktop Frame:")
        # print(f"ΔX: {dx:.2f} mm, ΔY: {dy:.2f} mm, ΔZ: {dz:.2f} mm")
        # print(f"Yaw Angle (0~360°): {np.degrees(yaw_rad):.2f}°")

        i = id_to_index[marker_id]
        cX = int(corners[i][0][:, 0].mean())
        cY = int(corners[i][0][:, 1].mean())
        cv2.putText(img, f"dx:{dx:.1f} dy:{dy:.1f}", (cX, cY + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)

        marker_pose = [float(v) for v in [dx / 1000, dy / 1000, -0.05, 0, 0, -rz_final]]

        marker_pose_dict[marker_id] = marker_pose

    cv2.namedWindow("Snapshot Result", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Snapshot Result", 1280, 720)
    cv2.imshow("Snapshot Result", img)
    cv2.waitKey(2000)
    cv2.destroyWindow("Snapshot Result")

    return marker_pose_dict

def main():
    # Camera Initialization
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_FOCUS, 388)

    print("Press SPACE to start sequence | ESC to exit")
    cv2.namedWindow("Distance Measurement", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Distance Measurement", 1280, 720)

    # Move robot to home camera position at startup
    home = f"""
def move_to_home():
  Camera = [{', '.join(f'{v:.5f}' for v in camera_j)}]
  movej(Camera, a=0.8, v=0.8)
  sync()
end
move_to_home()
"""
    send_urscript_command(home)
    print("[INFO] Sent robot to home position")
    time.sleep(5)

    started = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        message = "Press SPACE to start" if not started else "Running..."
        cv2.putText(display, message, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("Distance Measurement", display)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == 32 and not started:
            started = True
            for current_id in range(3, 10):
                marker_pose_dict = distance_measure(cap)
                if current_id in marker_pose_dict:
                    print(f"[INFO] Moving to ID {current_id}: {marker_pose_dict[current_id]}")
                    ur_move(current_id, marker_pose_dict[current_id])
                    time.sleep(48)
                    print(f"[INFO] Finished move for ID {current_id}")
                else:
                    print(f"[WARNING] ID {current_id} not detected")

            print("[INFO] Sequence finished.")
            break

    cap.release()
    cv2.destroyAllWindows(1)


if __name__ == "__main__":
    main()