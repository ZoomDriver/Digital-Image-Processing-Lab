import cv2
import numpy as np

# 读取本地图片（把你的图命名为 shape.png）
img_path = "shape.png"
img = cv2.imread(img_path)
if img is None:
    print("图片读取失败，请检查图片路径！")
    exit()

img_result = img.copy()

# 图像预处理：灰度化、高斯降噪、Canny边缘检测
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
# 调整Canny阈值，减少细碎边缘
edges = cv2.Canny(blur, 80, 200)

# ========== 霍夫概率直线检测（绿色线段）==========
# 调高 threshold，增大minLineLength，过滤短线噪声
lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180,
                        threshold=80,
                        minLineLength=60,
                        maxLineGap=8)
if lines is not None:
    for line in lines:
        coords = line.reshape(4)
        x1, y1, x2, y2 = coords
        cv2.line(img_result, (x1, y1), (x2, y2), (0, 255, 0), 2)

# ========== 霍夫圆检测（红色圆+圆心）==========
# param2调高，减少误检；半径范围适配图中大圆小圆
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT,
                           dp=1.2,
                           minDist=25,
                           param1=80,
                           param2=45,
                           minRadius=8,
                           maxRadius=300)
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        cx, cy, r = i[0], i[1], i[2]
        cv2.circle(img_result, (cx, cy), r, (0, 0, 255), 2)
        cv2.circle(img_result, (cx, cy), 2, (0, 0, 255), -1)

# ====================== 保存图片 ======================
cv2.imwrite("original.png", img)
cv2.imwrite("canny_edge.png", edges)
cv2.imwrite("hough_result.png", img_result)
print("图片已保存到当前文件夹！")

# 显示窗口
cv2.imshow("Original Image", img)
cv2.imshow("Canny Edge", edges)
cv2.imshow("Hough Line & Circle Result", img_result)

cv2.waitKey(0)
cv2.destroyAllWindows()
