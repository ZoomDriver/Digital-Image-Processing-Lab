import cv2
import numpy as np

# -------------------------- 配置参数 --------------------------
VIDEO_PATH = "test.mp4"       # 替换成你的本地视频路径
THRESHOLD = 0.8               # 置信度阈值 >0.8
METHOD = cv2.TM_CCOEFF_NORMED # 归一化相关系数，输出0~1，越接近1匹配越好
# TM_CCOEFF_NORMED / TM_CCORR_NORMED 这两个适合置信度判断
# -------------------------------------------------------------

# 打开视频
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("无法打开视频，请检查路径！")
    exit()

# 读取第一帧，用来框选模板
ret, frame = cap.read()
if not ret:
    print("读取第一帧失败")
    cap.release()
    exit()

# 手动框选ROI模板，框选完按回车确认
print("请框选需要追踪的模板目标，框选后按回车确认")
roi = cv2.selectROI("select template", frame, False, False)
cv2.destroyWindow("select template")
x, y, w, h = roi
if w == 0 or h == 0:
    print("模板框选无效！")
    cap.release()
    exit()

# 截取模板图像
template = frame[y:y+h, x:x+w]
t_h, t_w = template.shape[:2]

# 初始模板保留一份原始模板兜底
origin_template = template.copy()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_temp = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(gray_frame, gray_temp, METHOD)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    print(f"conf: {max_val:.3f}")

    if max_val > 0.8:
        top_left = max_loc
        bottom_right = (top_left[0] + t_w, top_left[1] + t_h)
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
        cv2.putText(frame, f"conf:{max_val:.3f}",
                    (top_left[0], top_left[1]-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        
        # 高分才更新模板，慢慢跟着目标外观变化
        if max_val > 0.85:
            template = frame[top_left[1]:top_left[1]+t_h, top_left[0]:top_left[0]+t_w]
    else:
        # 分数太低，切回原始模板重新搜索
        template = origin_template.copy()

    cv2.imshow("template match track", frame)
    key = cv2.waitKey(25) & 0xFF
    if key == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
