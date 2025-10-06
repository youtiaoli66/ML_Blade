import numpy as np
import cv2
import csv
import matplotlib.pyplot as plt
from scipy import interpolate

def load_mask_csv(path):
    with open(path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        mask = np.array([[int(val) for val in row] for row in reader])
    return mask.astype(np.uint8)

def save_dense_boundary_csv(dense_points, save_path, x_min, y_min, x_scale, y_scale):
    with open(save_path, 'w', encoding='utf-8', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["X (m)", "Y (m)"])
        for pt in dense_points:
            x_pix, y_pix = pt[0], pt[1]
            x_real = x_pix / x_scale + x_min
            y_real = y_pix / y_scale + y_min
            writer.writerow([x_real, y_real])

def densify_contour(contour, num_points=10000):
    contour = contour[:, 0, :]  # shape (N, 2)
    x, y = contour[:, 0], contour[:, 1]
    dist = np.cumsum(np.sqrt(np.diff(x, prepend=x[0])**2 + np.diff(y, prepend=y[0])**2))
    dist = dist / dist[-1]  # Normalize to [0, 1]

    interp_dist = np.linspace(0, 1, num=num_points)
    fx = interpolate.interp1d(dist, x, kind='linear')
    fy = interpolate.interp1d(dist, y, kind='linear')

    x_new = fx(interp_dist)
    y_new = fy(interp_dist)
    return np.stack([x_new, y_new], axis=1).astype(np.float32)

def extract_inner_boundary(mask_path, scale_csv_path, save_csv_path, num_points=5000):
    # 1. 读取 mask
    mask = load_mask_csv(mask_path)

    # 2. 提取所有轮廓（含层级信息）
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        raise ValueError("❌ 没有找到任何轮廓")
    hierarchy = hierarchy[0]

    # 3. 筛选所有有父轮廓的内轮廓
    inner_contours = [cnt for i, cnt in enumerate(contours) if hierarchy[i][3] != -1]
    if not inner_contours:
        raise ValueError("❌ 没有找到任何内部轮廓")

    # 4. 选择最大内部轮廓
    target_contour = max(inner_contours, key=cv2.contourArea)

    # 5. 加载缩放信息
    with open(scale_csv_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        x_min, x_max = map(float, next(reader))
        y_min, y_max = map(float, next(reader))
    scale = mask.shape[0]
    x_scale = (scale - 1) / (x_max - x_min)
    y_scale = (scale - 1) / (y_max - y_min)

    # 6. 插值加密轮廓
    dense_points = densify_contour(target_contour, num_points=num_points)

    # 7. 保存为 CSV
    save_dense_boundary_csv(dense_points, save_csv_path, x_min, y_min, x_scale, y_scale)

    # 8. 可视化
    plt.imshow(mask, cmap='gray')
    for cnt in inner_contours:
        cnt_np = cnt[:, 0, :]
        plt.plot(cnt_np[:, 0], cnt_np[:, 1], linewidth=0.5, linestyle='--', color='blue')
    plt.plot(dense_points[:, 0], dense_points[:, 1], color='red', linewidth=2.0, label='Dense Inner Boundary')
    plt.title("Extracted Dense Inner Boundary")
    plt.gca().invert_yaxis()
    plt.axis('equal')
    plt.legend()
    plt.show()

# 使用路径（你可以改成自己的）
mask_path = "C:/Users/86176/Desktop/python/AICFD/heeds/heeds_Study_1/HEEDS_0/Design42/STARCCM_3/mask.csv"
scale_csv_path = "C:/Users/86176/Desktop/python/AICFD/mask/data_scale.csv"
save_csv_path = "C:/Users/86176/Desktop/python/AICFD/heeds/heeds_Study_1/HEEDS_0/Design42/STARCCM_3/XYZ blade boundarydenorm.csv"

# 提取并保存加密边界（默认 1000 点）
extract_inner_boundary(mask_path, scale_csv_path, save_csv_path, num_points=5000)
