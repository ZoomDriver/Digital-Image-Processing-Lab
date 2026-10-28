# -*- coding: utf-8 -*-
"""
灰度图模板匹配实验
==================
从一张灰度图中截取一个小区域作为模板，然后分别用：
  1) SSDA 算法（序贯相似性检测算法）
  2) 特征匹配算法（ORB）
  3) 二阶段模板匹配算法（粗搜 + 精搜）
找出模板在原图中的位置，并可视化对比。

用法：
  python template_matching.py <原图路径> [模板宽] [模板高] [左上角x] [左上角y]

  - 不带参数时，自动生成一张带纹理的测试图；
  - 只传原图路径时，自动在原图中心截取一个 80x80 的小块作为模板；
  - 传 6 个参数时，按指定位置截取模板。
"""

import sys
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use("Agg")
# 依次尝试 Windows 常见中文字体
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "KaiTi", "FangSong"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示为方块

# ============================================================
# 0. 读入 / 生成灰度图
# ============================================================
def load_or_create_image():
    """读取本地图片并转灰度；若未提供路径则生成一张测试图。"""
    if len(sys.argv) >= 2:
        img_path = sys.argv[1]
        bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"无法读取图片: {img_path}")
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        print(f"[INFO] 已读取本地图片: {img_path}, 尺寸={gray.shape}")
        return gray
    else:
        # 生成一张带纹理 + 几何图形的测试图
        h, w = 480, 640
        gray = np.zeros((h, w), dtype=np.uint8)
        # 背景纹理：正弦条纹 + 随机噪声
        yy, xx = np.mgrid[0:h, 0:w]
        texture = (60 + 40 * np.sin(xx / 15.0) * np.cos(yy / 20.0)).astype(np.uint8)
        gray[:] = texture
        # 画一些几何块，便于模板定位
        cv2.rectangle(gray, (150, 120), (260, 220), 200, -1)
        cv2.circle(gray, (420, 300), 55, 180, -1)
        cv2.putText(gray, "TEST", (320, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, 230, 3)
        rng = np.random.default_rng(0)
        gray = np.clip(gray.astype(np.int32) + rng.integers(-15, 15, (h, w)), 0, 255).astype(np.uint8)
        print("[INFO] 未提供图片路径，已生成测试图 640x480")
        return gray


def extract_template(gray):
    """从原图截取一个小灰度块作为模板。"""
    h, w = gray.shape
    if len(sys.argv) >= 6:
        tw, th, tx, ty = map(int, sys.argv[2:6])
    else:
        tw, th = 80, 80
        tx, ty = w // 2 - tw // 2, h // 2 - th // 2
    # 边界保护
    tx = max(0, min(tx, w - tw))
    ty = max(0, min(ty, h - th))
    templ = gray[ty:ty + th, tx:tx + tw].copy()
    print(f"[INFO] 截取模板: 位置=({tx},{ty}), 尺寸={tw}x{th}")
    return templ, (tx, ty, tw, th)


# ============================================================
# 1. SSDA 算法（序贯相似性检测）
# ============================================================
def ssda_match(gray, templ, threshold=200, sample_step=4):
    """
    SSDA (Sequential Similarity Detection Algorithm)
    ------------------------------------------------
    思路：
      对每个候选位置，随机（或按步长）取若干像素点累加绝对误差 |I-T|，
      一旦累计误差超过阈值 T 就立即跳到下一个位置（早退），
      只有少数"真匹配"位置能撑到采样结束，从而大幅减少计算量。

    参数:
      threshold   累计误差阈值（早退线）
      sample_step 采样步长，越小越精确但越慢

    返回:
      (top_left_x, top_left_y, elapsed_ms)
    """
    H, W = gray.shape
    h, w = templ.shape
    t0 = time.time()

    # 预生成采样点（在模板内按步长取点）
    ys, xs = np.mgrid[0:h:sample_step, 0:w:sample_step]
    pts = list(zip(ys.ravel(), xs.ravel()))
    n_pts = len(pts)

    best_pos = (0, 0)
    best_score = -1e9  # 能撑到最后的点越多 / 最终误差越小越好

    for y in range(0, H - h + 1):
        for x in range(0, W - w + 1):
            acc = 0
            quit_flag = False
            for (dy, dx) in pts:
                acc += abs(int(gray[y + dy, x + dx]) - int(templ[dy, dx]))
                if acc > threshold:
                    quit_flag = True
                    break
            if quit_flag:
                continue
            # 没被早退：计算全图绝对误差作为评分（越小越好）
            roi = gray[y:y + h, x:x + w].astype(np.int32)
            diff = np.abs(roi - templ.astype(np.int32)).sum()
            score = -diff
            if score > best_score:
                best_score = score
                best_pos = (x, y)

    elapsed = (time.time() - t0) * 1000
    print(f"[SSDA] 耗时 {elapsed:.1f} ms, 匹配位置={best_pos}, 评分={-best_score:.0f}")
    return best_pos[0], best_pos[1], elapsed


# ============================================================
# 2. 特征匹配算法（ORB + 单应性 / 仿射估计）
# ============================================================
def feature_match_orb(gray, templ):
    """
    用 ORB 特征点 + 描述子匹配，再通过 RANSAC 估计单应矩阵，
    求出模板在原图中的四角位置，进而得到外接矩形。
    对旋转 / 尺度变化有一定鲁棒性。
    """
    t0 = time.time()

    orb = cv2.ORB_create(nfeatures=500)
    kp1, des1 = orb.detectAndCompute(templ, None)
    kp2, des2 = orb.detectAndCompute(gray, None)

    if des1 is None or des2 is None or len(kp1) < 5 or len(kp2) < 5:
        print("[ORB] 特征点不足，返回 (0,0)")
        return 0, 0, (time.time() - t0) * 1000, None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda m: m.distance)

    good = matches[:max(10, len(matches) // 4)]
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)

    h, w = templ.shape
    if M is None:
        x, y = 0, 0
        corners = None
    else:
        pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        corners = cv2.transform(pts, M).reshape(-1, 2)
        x, y = int(corners[:, 0].min()), int(corners[:, 1].min())

    elapsed = (time.time() - t0) * 1000
    print(f"[ORB] 耗时 {elapsed:.1f} ms, 好匹配点={len(good)}, 估计位置=({x},{y})")
    return x, y, elapsed, corners


# ============================================================
# 3. 二阶段模板匹配（粗搜 + 精搜）
# ============================================================
def two_stage_match(gray, templ, coarse_step=8):
    """
    二阶段模板匹配：
      阶段一（粗搜）：以 coarse_step 为步长滑动，用归一化相关系数 TM_CCOEFF_NORMED 打分，
                     只在稀疏网格上找 Top-K 候选位置；
      阶段二（精搜）：在每个候选位置周围 coarse_step 邻域内做逐像素精确匹配，
                     取得分最高者。
    相比逐像素全图匹配，速度通常快 5~10 倍。
    """
    t0 = time.time()
    H, W = gray.shape
    h, w = templ.shape

    # ---- 阶段一：粗搜 ----
    coarse_scores = []
    for y in range(0, H - h + 1, coarse_step):
        for x in range(0, W - w + 1, coarse_step):
            roi = gray[y:y + h, x:x + w]
            res = cv2.matchTemplate(roi, templ, cv2.TM_CCOEFF_NORMED)
            coarse_scores.append((float(res[0, 0]), x, y))
    coarse_scores.sort(reverse=True, key=lambda t: t[0])
    top_candidates = coarse_scores[:10]

    # ---- 阶段二：精搜 ----
    best_val, best_loc = -1e9, (0, 0)
    for _, cx, cy in top_candidates:
        y0 = max(0, cy - coarse_step)
        y1 = min(H - h, cy + coarse_step)
        x0 = max(0, cx - coarse_step)
        x1 = min(W - w, cx + coarse_step)
        # 在小窗口内逐像素
        sub = gray[y0:y1 + h, x0:x1 + w]
        if sub.shape[0] < h or sub.shape[1] < w:
            continue
        res = cv2.matchTemplate(sub, templ, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val > best_val:
            best_val = max_val
            best_loc = (x0 + max_loc[0], y0 + max_loc[1])

    elapsed = (time.time() - t0) * 1000
    print(f"[二阶段] 耗时 {elapsed:.1f} ms, 匹配位置={best_loc}, 相关系数={best_val:.4f}")
    return best_loc[0], best_loc[1], elapsed


# ============================================================
# 4. 可视化
# ============================================================
def draw_rect(ax, img, x, y, w, h, color, label):
    ax.imshow(img, cmap="gray")
    if x is not None and y is not None:
        rect = plt.Rectangle((x, y), w, h, edgecolor=color, facecolor="none", linewidth=2)
        ax.add_patch(rect)
    ax.set_title(label, fontsize=12)
    ax.axis("off")


def main():
    gray = load_or_create_image()
    templ, (gt_x, gt_y, tw, th) = extract_template(gray)

    # 运行三种算法
    sx, sy, s_t = ssda_match(gray, templ)
    ox, oy, o_t, orb_corners = feature_match_orb(gray, templ)
    tx, ty, t_t = two_stage_match(gray, templ, coarse_step=8)

    # 画图
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    # 真值
    draw_rect(axes[0], gray, gt_x, gt_y, tw, th, "lime",
              f"GT 真实位置 ({gt_x},{gt_y})")

    # SSDA
    draw_rect(axes[1], gray, sx, sy, tw, th, "red",
              f"SSDA\n({sx},{sy})  {s_t:.0f} ms")

    # ORB
    draw_rect(axes[2], gray, ox, oy, tw, th, "cyan",
              f"ORB 特征匹配\n({ox},{oy})  {o_t:.0f} ms")
    if orb_corners is not None:
        poly = plt.Polygon(orb_corners, edgecolor="cyan", facecolor="none", linewidth=2)
        axes[2].add_patch(poly)

    # 二阶段
    draw_rect(axes[3], gray, tx, ty, tw, th, "yellow",
              f"二阶段模板匹配\n({tx},{ty})  {t_t:.0f} ms")

    plt.suptitle("三种模板匹配算法对比", fontsize=14)
    plt.tight_layout()

    out_path = "match_result.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"\n[INFO] 结果图已保存: {out_path}")

    # 同时把模板单独存一张
    cv2.imwrite("template.png", templ)
    print("[INFO] 模板图已保存: template.jpg")

    # 打印误差对比
    print("\n========== 结果汇总 ==========")
    print(f"真实位置:        ({gt_x}, {gt_y})")
    print(f"SSDA 位置:       ({sx}, {sy})  误差=({sx-gt_x}, {sy-gt_y})")
    print(f"ORB 位置:        ({ox}, {oy})  误差=({ox-gt_x}, {oy-gt_y})")
    print(f"二阶段位置:      ({tx}, {ty})  误差=({tx-gt_x}, {ty-gt_y})")


if __name__ == "__main__":
    main()
