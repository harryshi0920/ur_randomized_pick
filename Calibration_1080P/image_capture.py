import cv2
import os
"""
Python 的 os 模块是“操作系统接口”，
用 Python 控制文件、路径、进程、环境变量等底层操作系统功能。
"""

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # 打开默认摄像头
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)          # 自动对焦
cap.set(cv2.CAP_PROP_FOCUS, 40)             # 焦距设置为40

save_dir = "Calibration_1080P/calib_imgs"
os.makedirs(save_dir, exist_ok=True)
"""
如果 data/、images/、results/ 都不存在，会依次全部创建
如果已经存在，不会报错，程序继续运行
"""

count = 0
print("按 's' 拍照，按 'ESC' 退出")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Camera", frame)
    key = cv2.waitKey(20) & 0xFF

    if key == ord('s'):
        filename = os.path.join(save_dir, f"img_{count:02d}.png")       # 路径拼接函数
        """
        - img_: 固定前缀
        - {count:02d}: 用两位数字表示 count, 不够两位前面补 0
            部分    含义
            {...}	f-string 语法，把变量嵌入字符串
            count	变量名（你要格式化的值）
                :	格式控制符的开始
                0	前面补零
                2	总宽度为 2
                d	十进制整数(d = decimal)
 
        - .jpg: 图片后缀
        """
        cv2.imwrite(filename, frame)
        print(f"已保存: {filename}")
        count = count + 1

    elif key == 27: 
        break

cap.release()
cv2.destroyAllWindows()
