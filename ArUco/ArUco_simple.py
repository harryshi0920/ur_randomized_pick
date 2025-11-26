import cv2
import cv2.aruco as aruco
import numpy as np

# Camera Interface
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # 打开默认摄像头
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)          # 自动对焦
cap.set(cv2.CAP_PROP_FOCUS, 40)             # 焦距设置为40

"""
# 定义相机内参和畸变系数 (需要标定相机)
camera_matrix = np.array([[1000, 0, 640],
                          [0, 1000, 360],
                          [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((5, 1))  # 默认假设无畸变（需要标定相机）
"""


# 定义相机内参和畸变系数 (需要标定相机)
"""camera_matrix = np.array([
    [627.32911975,     0.0,            309.07601879],
    [0.0,              627.96552459,   230.05588573],
    [0.0,              0.0,            1.0]
], dtype=np.float32)

dist_coeffs = np.array(
    [4.91158427e-02, -1.99719913e-01, -1.66224400e-04, -3.31318376e-03, 1.47963074e-01],
    dtype=np.float32
)   """


# 从 .npz 文件加载相机内参和畸变系数
params = np.load("camera_params.npz")
camera_matrix = params["mtx"]
dist_coeffs = params["dist"]

# ArUco 标记实际边长 (Unit: m)
marker_length = 0.02

# Choose ArUco dictionary
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_50)
"""
Example of ArUco Dictionary

Dictionary Name         Bits        Tag#
DICT_4x4_50             4x4         50
DICT_4x4_100            4x4         100
DICT_4x4_250            4x4         250
DICT_5x5_50             5x5         50
DICT_6x6_1000           6x6         1000  
"""
parameters = aruco.DetectorParameters()
"""
All the parameters for aruco detection is is defect mode
To modifed the parameters:
parameters.adaptiveThreshConstant = 5.0  # 自适应阈值常数
parameters.minMarkerDistanceRate = 0.02  # 最小标记间距比率
parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX  # 启用亚像素角点检测
parameters.cornerRefinementWinSize = 7  # 亚像素窗口大小
parameters.cornerRefinementMaxIterations = 50  # 亚像素最大迭代次数
"""

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # BGR to Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    """
    为什么要转化为灰度图 (GrayScale)，而二值图 (Binary) 通常不适合直接检测？
    (1) 丢失关键信息
        - 二值图只保留 黑白两种状态，导致 边缘模糊、噪声扩大、角点丢失。
        - 对于 光照变化、模糊或反光 场景，二值化结果极易 失真。
    (2) ArUco 检测依赖自适应阈值
        - ArUco 检测中的 adaptiveThreshConstant 等参数，是 在灰度图上 进行自适应阈值处理。
        - 如果你 提前用 cv2.threshold() 二值化，会破坏 ArUco 内部的阈值算法。
    (3) 无法进行角点细化
        - 二值图中边缘为 硬性二值，无法进行 亚像素角点检测 (subpixel refinement)，精度会大大降低。
    """

    # Detect ArUco Markers
    corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters = parameters)
    """
    1. corners （角点坐标）
        - 是一个 列表 (list)，其中每个元素是 检测到的 ArUco 标记的 4 个角点坐标。
        - 每个角点用 二维坐标 表示 (x, y)。
        - 坐标顺序：从左上角开始，逆时针排列。
    2. ids
        - 是一个 NumPy 数组，包含 检测到的 ArUco 标记 ID。
        - 如果，只检测到一个 ArUco 码，那么列表中只有一个 ID ，例如：[[3]]。
        - 如果，检测到多个 ArUco 码，例如：[[4], [3], [1]]。
    3. rejected
        - 标记四边形区域，但是因为不符合编码规则，被归类为拒绝
    """
    # 姿态估计 
    rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)       # 返回 每个标记在相机坐标系中的位姿（translation 和 rotation）
    """
    rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, markerLength, cameraMatrix, distCoeffs)
    1. 参数解析
        参数名          类型                            解释
        corners         list or numpy.ndarray           detectMarkers()返回的角点坐标，格式为 [n, 1, 4, 2] ( n是标记数量 4是四个角点  2是(x,y) )
        markerLength	float                           ArUco标记的实际尺寸
        cameraMatrix    numpy.ndarray                   相机内参矩阵 (3x3) 一般是相机标定的结果
        distCoeffs      numpy.ndarray                   畸变系数 (1x5或1x8)，同样来自相机标定结果。
    
    2. 返回值解析
        返回值          类型                解释
        rvecs           numpy.ndarray       旋转向量 (n, 1, 3)：表示标记在相机坐标系中的旋转。
        tvecs           numpy.ndarray       平移向量 (n, 1, 3)：表示标记在相机坐标系中的位置信息。
        _               None                返回的标记对象索引 (多标记时一般用不到)。
    """
    # 绘制检测到的标记
    if ids is not None:
        aruco.drawDetectedMarkers(frame, corners, ids)
        """
        正确的调用格式：
        cv2.aruco.drawDetectedMarkers(image, corners, ids=None, borderColor=(0, 255, 0))
            - image: 图像，必填
            - corners: 每个 ArUco 标记的角点坐标数组， detectMarkers() 输出，必填
            - ids: ArUco 标记对应的 ID 列表， detectMarkers() 输出，选填
            - borderColor: 标记边框颜色，默认 (0, 255, 0) (绿色)，选填
        """
        for i, marker_id in enumerate(ids):
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], marker_length, thickness = 3)
            print(f"Marker ID: {marker_id[0]} | Corners: {corners[i][0]}")

    cv2.imshow("ArUco Detection", frame)

    # 按下 'q' 退出
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

"""
如何提高测量精度？
    1. 进行相机标定：使用 calib3d 标定摄像头，提高 camera_matrix 精度。
    2. 滤波平滑距离数据：使用多帧均值或卡尔曼滤波。
    3. 调整检测参数：优化 DetectorParameters() 提高检测准确度。
"""