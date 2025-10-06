import numpy as np
import csv

def masked_pressure_to_points(masked_csv_path, scale_csv_path, mask_path, save_csv_path):
    # 读取 masked 压力图
    with open(masked_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        pressure = np.array([[float(val) for val in row] for row in reader])

    # 读取 mask
    with open(mask_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        mask = np.array([[float(val) for val in row] for row in reader])

    # 读取 scale
    with open(scale_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        scale_vals = [[float(val) for val in row] for row in reader]
        x_min, x_max = scale_vals[0]
        y_min, y_max = scale_vals[1]

    scale = pressure.shape[0]
    x_scale = (x_max - x_min) / (scale - 1)
    y_scale = (y_max - y_min) / (scale - 1)

    # 收集非零值点
    result = []
    for i in range(scale):
        for j in range(scale):
            if mask[i][j] != 0:
                x_real = j * x_scale + x_min
                y_real = i * y_scale + y_min
                value = pressure[i][j]
                result.append([value, x_real, y_real])

    # 保存为 CSV
    with open(save_csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Pressure (Pa)', 'X (m)', 'Y (m)'])
        for row in result:
            writer.writerow(row)

if __name__ == '__main__':
    masked_pressure_to_points(
        masked_csv_path='C:/Users/86176/Desktop/python/AICFD/heeds/heeds_Study_1/HEEDS_0/Design42/STARCCM_3/pressure_targets.csv',
        scale_csv_path='C:/Users/86176/Desktop/python/AICFD/mask/data_scale.csv',
        mask_path='C:/Users/86176/Desktop/python/AICFD/heeds/heeds_Study_1/HEEDS_0/Design42/STARCCM_3/mask.csv',
        save_csv_path='C:/Users/86176/Desktop/python/AICFD/heeds/heeds_Study_1/HEEDS_0/Design42/STARCCM_3/pressuredenorm.csv'
    )
