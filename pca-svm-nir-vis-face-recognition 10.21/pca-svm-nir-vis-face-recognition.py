import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.metrics import accuracy_score

# ====================== 配置区 ======================
root = r"C:\Users\stu\Documents\pca-svm-face-recognition 10.21"
vis_dir = os.path.join(root, "VIS-cropped")
nir_dir = os.path.join(root, "NIR-cropped")
img_size = (128,128)
n_components = 80
test_ratio = 0.3
# ====================================================

def load_images(folder_path):
    data = []
    labels = []
    files = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg",".png",".bmp"))]
    print(f"\n文件夹 {folder_path} 找到图片数量：{len(files)}")
    for fname in files:
        print("当前文件名：", fname)
        # 适配 0001-1.bmp，按短横线分割取前面ID
        id_str = fname.split("-")[0]
        try:
            label = int(id_str)
        except Exception as e:
            print(f"!!! 解析ID失败，跳过文件 {fname}, err: {e}")
            continue

        img = Image.open(os.path.join(folder_path, fname)).convert("L")
        img = img.resize(img_size)
        arr = np.array(img) / 255.0
        flat = arr.reshape(-1)
        data.append(flat)
        labels.append(label)
    return np.array(data), np.array(labels)

# 加载VIS、NIR
X_vis, y_vis = load_images(vis_dir)
X_nir, y_nir = load_images(nir_dir)

print(f"\nVIS加载成功：X_vis={X_vis.shape}, y_vis={y_vis.shape}")
print(f"NIR加载成功：X_nir={X_nir.shape}, y_nir={y_nir.shape}")

# 合并
X = np.vstack([X_vis, X_nir])
y = np.hstack([y_vis, y_nir])
print(f"\n总样本数：{X.shape[0]}, 单张图片原始特征维度：{X.shape[1]}")
print(f"总类别数：{len(np.unique(y))}")

if X.shape[0] == 0 or len(np.unique(y)) < 2:
    raise Exception("没有加载到有效样本！检查图片命名和路径！")

# 划分训练测试
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_ratio, random_state=42, stratify=y
)

# PCA特征提取
pca = PCA(n_components=n_components, whiten=True, random_state=42)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
print(f"PCA降维后特征维度：{X_train_pca.shape[1]}")

base_svm = SVC(kernel="rbf", C=10, gamma=0.001, random_state=42)

# OVR
ovr = OneVsRestClassifier(base_svm)
ovr.fit(X_train_pca, y_train)
pred_ovr = ovr.predict(X_test_pca)
acc_ovr = accuracy_score(y_test, pred_ovr)

# OVO
ovo = OneVsOneClassifier(base_svm)
ovo.fit(X_train_pca, y_train)
pred_ovo = ovo.predict(X_test_pca)
acc_ovo = accuracy_score(y_test, pred_ovo)

print("="*40)
print(f"OVR(一对多) 识别精度：{acc_ovr:.4f}")
print(f"OVO(一对一) 识别精度：{acc_ovo:.4f}")
print("="*40)

# 绘图保存
plt.figure(figsize=(6,4))
bars = plt.bar(["OVR One-Versus-Rest","OVO One-Versus-One"], [acc_ovr, acc_ovo], color=["#4472C4","#ED7D31"])
plt.ylim(0,1)
plt.ylabel("Accuracy")
plt.title("PCA+SVM VIS-NIR人脸识别精度对比")
for bar,val in zip(bars, [acc_ovr, acc_ovo]):
    plt.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f"{val:.4f}", ha="center")

os.makedirs("results", exist_ok=True)
plt.savefig("./results/acc_compare.png", dpi=300, bbox_inches="tight")
plt.show()

# 单张bmp图片预测函数
def predict_single(img_path):
    img = Image.open(img_path).convert("L").resize(img_size)
    arr = np.array(img)/255.0
    flat = arr.reshape(1,-1)
    feat = pca.transform(flat)
    p1 = ovr.predict(feat)[0]
    p2 = ovo.predict(feat)[0]
    print(f"单张图片预测结果：OVR预测ID={p1}, OVO预测ID={p2}")
    plt.figure()
    plt.imshow(arr, cmap="gray")
    plt.title(f"Pred: OVR={p1}, OVO={p2}")
    plt.axis("off")
    plt.show()

# 测试单张，示例：predict_single(r"C:\xxx\0001-1.bmp")
# predict_single(r"test.bmp")
