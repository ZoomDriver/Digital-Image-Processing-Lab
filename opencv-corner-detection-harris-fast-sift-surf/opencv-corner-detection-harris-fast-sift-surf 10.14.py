import cv2
import numpy as np

# ===================== 视频：Harris角点检测 =====================
def harris_corner_video(video_path, out_video_path):
    cap = cv2.VideoCapture(video_path)
    # 获取视频参数
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_video_path, fourcc, fps, (w, h))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        # Harris角点检测
        dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
        # 膨胀，方便显示角点
        dst = cv2.dilate(dst, None)
        # 阈值筛选，标记角点为红色
        frame[dst > 0.01 * dst.max()] = [0, 0, 255]
        out.write(frame)
        cv2.imshow("Harris Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Harris检测视频已保存：{out_video_path}")

# ===================== 视频：FAST角点检测 =====================
def fast_corner_video(video_path, out_video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_video_path, fourcc, fps, (w, h))

    fast = cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp = fast.detect(gray, None)
        # 在帧上绘制FAST角点
        frame = cv2.drawKeypoints(frame, kp, None, color=(0,255,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        out.write(frame)
        cv2.imshow("FAST Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"FAST检测视频已保存：{out_video_path}")

# ===================== 图像：SIFT特征点检测 =====================
def sift_image(img_path, out_img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None)
    img_sift = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite(out_img_path, img_sift)
    cv2.imshow("SIFT Image", img_sift)
    cv2.imshow("Original Image", img)
    print(f"SIFT特征图保存：{out_img_path}")
    return img, img_sift

# ===================== 图像：SURF特征点检测 =====================
def surf_image(img_path, out_img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Hessian阈值，值越大检测点越少
    surf = cv2.xfeatures2d.SURF_create(hessianThreshold=400)
    kp, des = surf.detectAndCompute(gray, None)
    img_surf = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite(out_img_path, img_surf)
    cv2.imshow("SURF Image", img_surf)
    print(f"SURF特征图保存：{out_img_path}")
    return img, img_surf

if __name__ == "__main__":
    # ========== 在这里修改你的本地文件路径 ==========
    video_file = "test.mp4"        # 你的本地视频
    image_file = "test.jpg"        # 你的本地图片

    # 运行视频角点检测
    harris_corner_video(video_file, "harris_output.mp4")
    fast_corner_video(video_file, "fast_output.mp4")

    # 运行图像SIFT/SURF检测
    sift_image(image_file, "sift_output.jpg")
    surf_image(image_file, "surf_output.jpg")

    cv2.waitKey(0)
    cv2.destroyAllWindows()
