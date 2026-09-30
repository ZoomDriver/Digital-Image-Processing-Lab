import cv2
import numpy as np
from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ========= 1.读取本地医学图像 =========
img_path = "medical.jpg"
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError("图片读取失败，请检查路径是否正确")

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. 预处理：高斯模糊降噪
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# 3. 二值化：OTSU自动阈值
ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# 4. 形态学操作：去除小噪点，开运算
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# 确定背景区域：膨胀
sure_bg = cv2.dilate(opening, kernel, iterations=1)

# 5. 距离变换，获取确定前景
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.25 * dist_transform.max(), 255, 0)
sure_fg = np.uint8(sure_fg)

# =========修复subtract类型错误=========
sure_bg = np.uint8(sure_bg)
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# 7. 标记连通组件，生成marker
ret, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0

# 8. 执行分水岭算法
markers = cv2.watershed(img_rgb, markers)
img_rgb[markers == -1] = [255, 0, 0]

# ========= 绘图展示结果 =========
plt.figure(figsize=(14, 8))
plt.subplot(231), plt.imshow(img_rgb), plt.title("原图+分割红线边界"), plt.xticks([]), plt.yticks([])
plt.subplot(232), plt.imshow(thresh, cmap="gray"), plt.title("二值图"), plt.xticks([]), plt.yticks([])
plt.subplot(233), plt.imshow(sure_bg, cmap="gray"), plt.title("确定背景"), plt.xticks([]), plt.yticks([])
plt.subplot(234), plt.imshow(sure_fg, cmap="gray"), plt.title("确定前景"), plt.xticks([]), plt.yticks([])
plt.subplot(235), plt.imshow(unknown, cmap="gray"), plt.title("未知边界区"), plt.xticks([]), plt.yticks([])
plt.subplot(236), plt.imshow(markers, cmap="jet"), plt.title("Marker标记图"), plt.xticks([]), plt.yticks([])
plt.tight_layout()
plt.show()

cv2.imwrite("watershed_result.png", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
print("分割结果已保存为 watershed_result.png")
