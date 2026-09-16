"""
数字图像处理作业：暗通道先验雾图去雾
功能：
1. 读取有雾图像
2. 计算暗通道图像
3. 估计大气光值A
4. 估算透射率t
5. 根据成像模型恢复无雾图像
6. 展示：原图、暗通道图、去雾图
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

def dark_channel(img, window_size=15):
    """
    计算暗通道图像
    :param img: 输入RGB图像(H,W,3)
    :param window_size: 局部窗口大小
    :return: dark: 暗通道灰度图
    """
    # 取R G B三通道最小值
    min_rgb = np.min(img, axis=-1)
    # 最小值滤波（腐蚀操作）
    kernel = np.ones((window_size, window_size), np.uint8)
    dark = cv2.erode(min_rgb, kernel)
    return dark

def estimate_atmospheric_light(img, dark, percent=0.001):
    """估计大气光A，取暗通道最亮的前0.1%像素在原图中的值"""
    h, w = dark.shape
    num_pixels = h * w
    # 找到暗通道中亮度最高的前0.1%像素索引
    top_pixels = int(max(num_pixels * percent, 1))
    dark_vec = dark.reshape(-1)
    indices = np.argsort(dark_vec)[-top_pixels:]
    img_vec = img.reshape(-1, 3)
    A = np.max(img_vec[indices], axis=0)
    return A

def estimate_transmission(img, A, omega=0.95, window_size=15):
    """估算透射率 t，omega保留少量雾气"""
    img_norm = img / A
    min_norm = np.min(img_norm, axis=-1)
    kernel = np.ones((window_size, window_size), np.uint8)
    t = 1 - omega * cv2.erode(min_norm, kernel)
    return t

def recover_image(img, t, A, t0=0.1):
    """无雾图像恢复，限制t下限防止噪声放大"""
    t = np.maximum(t, t0)
    t = np.expand_dims(t, axis=-1)
    J = (img - A) / t + A
    J = np.clip(J, 0, 1)
    return J

if __name__ == "__main__":
    # 读取雾图，请将雾图命名 fog.jpg 放在代码同目录
    img_bgr = cv2.imread("fog.jpg")
    if img_bgr is None:
        print("图片读取失败！请检查 fog.jpg 是否在当前文件夹")
    else:
        # 转为RGB，并归一化到0~1
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = img_rgb.astype(np.float32) / 255.0

        # 1.计算暗通道
        dark = dark_channel(img, window_size=15)
        # 2.估计大气光
        A = estimate_atmospheric_light(img, dark)
        # 3.估计透射率
        t = estimate_transmission(img, A, omega=0.95, window_size=15)
        # 4.恢复去雾图像
        J = recover_image(img, t, A)

        # ========绘图展示：原图、暗通道图、去雾图========
        plt.figure(figsize=(15, 6))

        plt.subplot(1, 3, 1)
        plt.title("Original Hazy Image")
        plt.imshow(img_rgb)
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.title("Dark Channel Image")
        plt.imshow(dark, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.title("Dehazed Image")
        plt.imshow(J)
        plt.axis("off")

        plt.tight_layout()
        plt.show()
