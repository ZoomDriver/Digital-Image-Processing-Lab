import cv2
import matplotlib.pyplot as plt

# 1. 读取城堡图像，opencv读入是BGR
img_bgr = cv2.imread("castle.jpg")
if img_bgr is None:
    print("找不到图片！请把castle.jpg放到代码同一文件夹")
else:
    # 转为RGB格式
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # 2. split分离R, G, B三个通道
    r, g, b = cv2.split(img_rgb)

    # 3. 分别对R、G、B通道做直方图均衡化
    r_eq = cv2.equalizeHist(r)
    g_eq = cv2.equalizeHist(g)
    b_eq = cv2.equalizeHist(b)

    # 4. merge合并均衡后的三个通道，得到均衡化彩色图
    img_eq_rgb = cv2.merge([r_eq, g_eq, b_eq])

    # 5. 绘图展示所有结果
    plt.figure(figsize=(16, 10))

    plt.subplot(3, 3, 1)
    plt.title("Original RGB Image")
    plt.imshow(img_rgb)
    plt.axis("off")

    plt.subplot(3, 3, 2)
    plt.title("Original R channel")
    plt.imshow(r, cmap="gray")
    plt.axis("off")

    plt.subplot(3, 3, 3)
    plt.title("Original G channel")
    plt.imshow(g, cmap="gray")
    plt.axis("off")

    plt.subplot(3, 3, 4)
    plt.title("Original B channel")
    plt.imshow(b, cmap="gray")
    plt.axis("off")

    plt.subplot(3, 3, 5)
    plt.title("Equalized R channel")
    plt.imshow(r_eq, cmap="gray")
    plt.axis("off")

    plt.subplot(3, 3, 6)
    plt.title("Equalized G channel")
    plt.imshow(g_eq, cmap="gray")
    plt.axis("off")

    plt.subplot(3, 3, 7)
    plt.title("Equalized B channel")
    plt.imshow(b_eq, cmap="gray")
    plt.axis("off")

    plt.subplot(3, 3, 8)
    plt.title("Merged Equalized RGB Image")
    plt.imshow(img_eq_rgb)
    plt.axis("off")

    plt.tight_layout()
    plt.show()
