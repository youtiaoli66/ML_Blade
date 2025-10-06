import csv
import os

import cv2
from sympy.integrals.intpoly import point_sort
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import math
from collections import deque


# 读取 CSV 文件
def load_csv(path):
    with open(path, 'r', encoding='utf-8', newline='') as file:
        data = []
        csv_reader = csv.reader(file)
        header = next(csv_reader)  # 读取表头
        for row in csv_reader:
            row_list = []
            for i in row:
                row_list.append(float(i))  # 转换数据为浮点数
            data.append(np.array(row_list))  # 转换为 numpy 数组并添加到数据列表中
    return header, data


# 写入 CSV 文件
def write_csv(data, filename):
    with open(filename, 'w', encoding='utf-8', newline='') as file1:
        csv_writer = csv.writer(file1)
        for row in data:
            csv_writer.writerow(row)  # 将数据逐行写入 CSV


# 读取数据并进行处理
def read_data(path, name, filename, scale):
    mask = np.zeros((scale, scale))  # 创建全零矩阵
    total_pressure = [[] for _ in range(scale)]  # 创建嵌套列表

    processed_total_pressure = np.zeros((scale, scale))  # 处理后的压力矩阵
    processed_total_temperature = np.zeros((scale, scale))  # 处理后的温度矩阵
    processed_pressure = np.zeros((scale, scale))  # 处理后的压力数据

    _,processed_data = load_csv(path+'/'+name+'/STARCCM_3/'+filename)  # 加载 CSV 数据
    x_point_list = []
    y_point_list = []

    for i in processed_data:
        for j in range(len(i)):
            i[j] = float(i[j])
            if j == 0:
                x_point_list.append(i[j])
            elif j == 1:
                y_point_list.append(i[j])

    x_min, x_max = min(x_point_list), max(x_point_list)
    y_min, y_max = min(y_point_list), max(y_point_list)

    x_scale = (scale - 1) / (x_max - x_min)
    y_scale = (scale - 1) / (y_max - y_min)


    with open('mask/data_scale.csv', 'w', encoding='utf-8', newline='') as file2:
        min_max = csv.writer(file2)#创建文件写入器
        min_max.writerow([x_min, x_max])
        min_max.writerow([y_min, y_max])

    return x_min, y_min, x_scale, y_scale
# 计算裁剪区域
def get_clip_region(path, name, x_length, y_length):
    mask = []
    title, xy_data_blade = load_csv(path + '/' + name + '/STARCCM_3/' + blade_boundary_file)
    for i in range(len(title)):
        if title[i] == "X (m)":
            xy_data_blade = point_sort(xy_data_blade, int(i), int(i) + 1)
            blade_x_index = int(i)
            blade_y_index = int(i) + 1

    title, xy_data_side = load_csv(path + '/' + name + '/STARCCM_3/' + side_boundary_file)
    for i in range(len(title)):
        if title[i] == "X (m)":
            xy_data_side = point_sort(xy_data_side, int(i), int(i) + 1)
            side_x_index = int(i)
            side_y_index = int(i) + 1
    xy_map = np.zeros((x_length, y_length))

    x_min, y_min, x_scale, y_scale = read_data(heeds_path, name, filename, scale)
    blade_scaled_region = []
    side_scaled_region = []

    for i in xy_data_blade:
        scaled_x = (i[blade_x_index] - x_min) * x_scale
        scaled_y = (i[blade_y_index] - y_min) * y_scale
        blade_scaled_region.append([int(scaled_x), int(scaled_y)])

    for i in xy_data_side:
        scaled_x = (i[side_x_index] - x_min) * x_scale
        scaled_y = (i[side_y_index] - y_min) * y_scale
        side_scaled_region.append([int(scaled_x), int(scaled_y)])

    blade_scaled_region = np.array(blade_scaled_region)
    side_scaled_region = np.array(side_scaled_region)

    mask = cv2.polylines(
        xy_map,
        pts=[blade_scaled_region.reshape(-1, 1, 2), side_scaled_region.reshape(-1, 1, 2)],
        isClosed=True,
        color=[255, 255, 255],
        thickness=1
    )

    mask = poly(mask, point=[128, 0])
    mask = binary(mask)
    plt.imshow(mask)
    plt.savefig(path + '/' + name + '/STARCCM_3/mask.png')


    return mask


# 二值化处理
def binary(array):
    threshold = 128
    binary_array = np.where(array > threshold, 1, 0)  # 将大于 128 的像素设为 1，否则设为 0
    return binary_array


# 多边形填充
def poly(mat, point):
    h, w = mat.shape[:2]
    inq = [[False] * w for _ in range(h)]
    mat,inq = bfs(point[0], point[1], mat, inq)
    return mat


def bfs(x, y, mat, inq):
    X = [0, 0, 1, -1]
    Y = [1, -1, 0, 0]
    dq = deque()
    mat[x][y] = 255
    dq.append([x, y])
    inq[x][y] = True

    while len(dq) != 0:
        top = dq.popleft()
        for i in range(4):
            new_x = top[0] + X[i]
            new_y = top[1] + Y[i]
            if judge(new_x, new_y, mat, inq):
                dq.append([new_x, new_y])
                inq[new_x][new_y] = True
                mat[new_x][new_y] = 255

    return mat, inq

def judge(x, y, mat, inq):
    if x >= mat.shape[0] or x < 0 or y >= mat.shape[1] or y < 0:
        return False
    if mat[x][y] == 255 or inq[x][y] == True:
        return False
    return True

def calculate_distance(point1, point2, x, y):
    x1, y1 = point2[x], point1[y]
    x2, y2 = point1[x], point2[y]
    return (x1 - x2)**2 + (y1 - y2)**2

def point_sort(points, x, y):
    points = [tuple(point) for point in points]
    sorted_points = []
    current_point = points.pop(0)
    sorted_points.append(current_point)

    while points:
        closest_point = min(points, key=lambda p: calculate_distance(current_point, p, x, y))
        points.remove(closest_point)
        sorted_points.append(closest_point)
        current_point = closest_point

    return sorted_points


if __name__ == '__main__':
    scale = 256
    heeds_path = 'C:/Users/86176/Desktop/python/AICFD/heeds/heeds_Study_1/HEEDS_0'
    filename = 'XYZ Internal Table input data.csv'

    blade_boundary_file = 'XYZ blade boundary.csv'
    side_boundary_file = 'XYZ side boundary.csv'

    filenames = os.listdir(heeds_path)  # 获取路径下所有文件名

    for i, name in tqdm(enumerate(filenames), desc='generate mask...'):
        print(i, name)
        if i >= 0:

            x_min, y_min, x_scale, y_scale = read_data(heeds_path, name, filename, scale)
            mask = get_clip_region(heeds_path, name,scale,scale)
            write_csv(mask, heeds_path + '/'+ name + '/STARCCM_3/mask.csv')
            if i == 29:
                print(name)