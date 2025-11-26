import numpy as np
import cv2 as cv
import glob

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

chessboard_size = (9, 6)
square_size = 22.5  # 每个格子的边长，单位mm（你自己测量）

objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []
imgpoints = []  

images = glob.glob('Calibration_1080P/calib_imgs/*.jpg ')

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, corners = cv.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        cv.drawChessboardCorners(img, chessboard_size, corners2, ret)
        cv.imshow('Corners', img)
        cv.waitKey(0)

cv.destroyAllWindows()

# 执行相机标定
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None)
 
print("finished")
print("相机内参矩阵:\n", camera_matrix)
print("畸变系数:\n", dist_coeffs)

# 保存标定结果
np.savez("camera_params.npz", mtx=camera_matrix, dist=dist_coeffs)
