
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.spatial.distance import cdist
import alphashape
from shapely.geometry import Point


def load_pressure_points(path):
    data = np.genfromtxt(path, delimiter=',', skip_header=1)
    pressure = data[:, 0]
    x = data[:, 1]
    y = data[:, 2]
    return x, y, pressure

def load_boundary_points(boundary_path):
    data = np.genfromtxt(boundary_path, delimiter=',', skip_header=1)
    x = data[:, 0]
    y = data[:, 1]
    return np.column_stack((x, y))

def sort_boundary_points(boundary):
    sorted_pts = [boundary[0]]
    used = set([0])
    for _ in range(1, len(boundary)):
        last = sorted_pts[-1]
        dists = cdist([last], boundary)[0]
        dists[list(used)] = np.inf
        next_idx = np.argmin(dists)
        sorted_pts.append(boundary[next_idx])
        used.add(next_idx)
    return np.array(sorted_pts)

def estimate_normals_from_points(boundary):
    normals = []
    midpoints = []
    for i in range(len(boundary)):
        p1 = boundary[i]
        p2 = boundary[(i + 1) % len(boundary)]
        edge = p2 - p1
        midpoint = (p1 + p2) / 2.0
        normal = np.array([-edge[1], edge[0]], dtype=float)
        normal /= np.linalg.norm(normal) + 1e-8
        midpoints.append(midpoint)
        normals.append(normal)
    return np.array(midpoints), np.array(normals)

def compute_force_segmentwise(x, y, p, boundary_path, p_ref=101325):
    boundary = load_boundary_points(boundary_path)
    boundary = sort_boundary_points(boundary)
    midpoints, normals = estimate_normals_from_points(boundary)

    # 插值压力（中点）
    interp_p = griddata(points=(x, y), values=p, xi=midpoints, method='linear')
    interp_p = np.nan_to_num(interp_p)

    total_force = np.zeros(2)
    for i in range(len(midpoints)):
        p_local = interp_p[i] - p_ref
        # 每段长度作为局部 ds（注意闭合曲线末尾连首段）
        p1 = boundary[i]
        p2 = boundary[(i + 1) % len(boundary)]
        ds = np.linalg.norm(p2 - p1)
        F = -p_local * normals[i] * ds
        total_force += F

    # 可视化
    plt.figure(figsize=(6, 6))
    plt.scatter(x, y, s=1, alpha=0.3, label='All Points')
    plt.plot(boundary[:, 0], boundary[:, 1], 'r-', lw=2, label='Sorted Boundary')
    plt.quiver(midpoints[:, 0], midpoints[:, 1], normals[:, 0], normals[:, 1],
               angles='xy', scale_units='xy', scale=1, color='blue', width=0.002, label='Normals')
    plt.title(f"Segmentwise: Drag = {total_force[0]:.2f} N, Lift = {total_force[1]:.2f} N")
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.legend()
    plt.axis('equal')
    plt.grid(True)
    plt.show()

    return total_force[0], total_force[1]


# ✅ 在这里替换为你自己电脑上的实际文件路径
pressure_csv = "C:/Users/86176/Desktop/python/AICFD/heeds/heeds_Study_1/HEEDS_0/Design92/STARCCM_3/pressuredenorm.csv"
boundary_csv = "C:/Users/86176/Desktop/python/AICFD/heeds/heeds_Study_1/HEEDS_0/Design92/STARCCM_3/XYZ blade boundarydenorm.csv"

x, y, p = load_pressure_points(pressure_csv)
Fx, Fy = compute_force_segmentwise(x, y, p, boundary_csv)

print(f"压力造成的 Drag (X方向): {Fx:.3f} N")
print(f"压力造成的 Lift (Y方向): {Fy:.3f} N")
