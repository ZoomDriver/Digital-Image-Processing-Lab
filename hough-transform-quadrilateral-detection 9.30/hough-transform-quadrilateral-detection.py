import cv2
import numpy as np

# 判断两条直线是否平行
def is_parallel(line1, line2, angle_thresh=8):
    x1,y1,x2,y2 = line1
    x3,y3,x4,y4 = line2
    theta1 = np.arctan2(y2-y1, x2-x1)*180/np.pi
    theta2 = np.arctan2(y4-y3, x4-x3)*180/np.pi
    d = abs(theta1-theta2)
    d = min(d, 180-d)
    return d < angle_thresh

# 四边形顶点排序
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

# 判断四边形类型
def classify_quad(pts, angle_tol=12):
    p = order_points(pts)
    tl, tr, br, bl = p
    edges = [
        (tl[0],tl[1],tr[0],tr[1]),
        (tr[0],tr[1],br[0],br[1]),
        (br[0],br[1],bl[0],bl[1]),
        (bl[0],bl[1],tl[0],tl[1])
    ]
    pair1_par = is_parallel(edges[0], edges[2], angle_tol)
    pair2_par = is_parallel(edges[1], edges[3], angle_tol)
    if pair1_par and pair2_par:
        return "Rectangle"
    elif pair1_par ^ pair2_par:
        return "Trapezoid"
    else:
        return "Other"

# =========主程序=========
img_path = "shape.png"
img = cv2.imread(img_path)
if img is None:
    print("图片读取失败！检查路径")
    exit()
img_result = img.copy()
h,w = img.shape[:2]

# 预处理
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(5,5),0)
edges = cv2.Canny(blur, 80, 200)

# 霍夫直线检测（保留，满足实验要求：基于霍夫变换）
lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180,
                        threshold=80,
                        minLineLength=60,
                        maxLineGap=8)
# 先绘制霍夫检测到的全部直线（绿色）
if lines is not None:
    for line in lines:
        coords = line.reshape(4)
        x1,y1,x2,y2 = coords
        cv2.line(img_result, (x1,y1), (x2,y2), (0,255,0),1)

# 从边缘图寻找轮廓，四边形逼近（替换四重循环，解决卡死）
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    area = cv2.contourArea(cnt)
    # 过滤过小轮廓噪声
    if area < 150:
        continue
    # 多边形逼近
    approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
    if len(approx) == 4:
        four_pts = approx.reshape((4,2))
        shape_type = classify_quad(four_pts)
        pts = four_pts.astype(np.int32).reshape((-1,1,2))
        if shape_type == "Rectangle":
            cv2.polylines(img_result, [pts], True, (255,0,0), 2)
            cx,cy = np.mean(four_pts,axis=0).astype(int)
            cv2.putText(img_result, "Rect", (cx,cy), cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,0),2)
        elif shape_type == "Trapezoid":
            cv2.polylines(img_result, [pts], True, (0,165,255), 2)
            cx,cy = np.mean(four_pts,axis=0).astype(int)
            cv2.putText(img_result, "Trap", (cx,cy), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,165,255),2)


# 保存图片
cv2.imwrite("original.png", img)
cv2.imwrite("canny_edge.png", edges)
cv2.imwrite("quad_result.png", img_result)
print("✅ 图片保存完成！")

cv2.imshow("Original",img)
cv2.imshow("Canny Edge",edges)
cv2.imshow("Quad Detection Result",img_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
