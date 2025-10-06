import csv
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from scipy.interpolate import griddata

def load_data(path):
    with open(path, 'r', encoding='utf-8', newline='') as file:
        data = []
        csv_reader = csv.reader(file)
        header = next(csv_reader)
        for row in csv_reader:
            # 检查行是否为空，若为空则跳过
            if not any(row):  # 如果整行为空
                continue
            row_list = []
            for i in row:
                row_list.append(float(i))
            data.append(np.array(row_list))
    return data

def load_mask(path):
    with open(path, 'r', encoding='utf-8', newline='') as file1:
        data = []
        csv_reader = csv.reader(file1)
        #header = next(csv_reader)
        for row in csv_reader:
            row_list = []
            for i in row:
                row_list.append(float(i))
            data.append(np.array(row_list))
    return data

def get_data_scale(path):
    with open(path, 'r', encoding='utf-8', newline='') as file2:
        data = []
        csv_reader = csv.reader(file2)
        #header = next(csv_reader)
        for row in csv_reader:
            row_list = []
            for i in row:
                row_list.append(float(i))
            data.append(np.array(row_list))
            data_scale=data
    return data_scale[0][0],data_scale[0][1],data_scale[1][0],data_scale[1][1]


def generate_picture(data, path, scale):
    temp_pic = np.zeros((scale, scale))
    pressure = [[[] for _ in range(scale)] for _ in range(scale)]
    processed_pressure = np.zeros((scale, scale))
    total_pressure = [[[] for _ in range(scale)] for _ in range(scale)]
    processed_total_pressure = np.zeros((scale, scale))
    total_temperature = [[[] for _ in range(scale)] for _ in range(scale)]
    processed_total_temperature = np.zeros((scale, scale))
    mach = [[[] for _ in range(scale)] for _ in range(scale)]
    processed_mach = np.zeros((scale, scale))

    mask = load_mask(path + 'mask.csv')
    x_min, x_max, y_min, y_max = get_data_scale('C:/Users\86176\Desktop\python\AICFD\mask/data_scale.csv')
    x_scale = (scale - 1) / (x_max - x_min)
    y_scale = (scale - 1) / (y_max - y_min)

    for i in data:
        i[0] = round((i[0] - x_min) * x_scale)
        i[1] = round((i[1] - y_min) * y_scale)
        pressure[int(i[0])][int(i[1])].append(i[5])
        total_pressure[int(i[0])][int(i[1])].append(i[3])
        total_temperature[int(i[0])][int(i[1])].append(i[4])

    for i in range(scale):
        for j in range(scale):
            if not pressure[i][j]:
                continue
            else:
                processed_pressure[i][j] = sum(pressure[i][j]) / len(pressure[i][j])
                processed_total_pressure[i][j] = sum(total_pressure[i][j]) / len(total_pressure[i][j])
                processed_total_temperature[i][j] = sum(total_temperature[i][j]) / len(total_temperature[i][j])
                temp_pic[i][j] = 1

    nonzero_position = np.argwhere(processed_pressure != 0)
    nonzero_pressure = processed_pressure[nonzero_position[:, 0], nonzero_position[:, 1]]
    nonzero_total_pressure = processed_total_pressure[nonzero_position[:, 0], nonzero_position[:, 1]]
    nonzero_total_temperature = processed_total_temperature[nonzero_position[:, 0], nonzero_position[:, 1]]
#坐标位置转变行列序号
    x, y = np.meshgrid(np.arange(scale), np.arange(scale))
    pressure_points = np.column_stack([y.ravel(), x.ravel()])
    total_pressure_points = np.column_stack([y.ravel(), x.ravel()])
    total_temperature_points = np.column_stack([y.ravel(), x.ravel()])

    filled_pressure = griddata(nonzero_position, nonzero_pressure, pressure_points, method='linear')
    filled_pressure = filled_pressure.reshape(processed_pressure.shape)

    filled_total_pressure = griddata(nonzero_position, nonzero_total_pressure, total_pressure_points, method='linear')
    filled_total_pressure = filled_total_pressure.reshape(processed_total_pressure.shape)

    filled_total_temperature = griddata(nonzero_position, nonzero_total_temperature, total_temperature_points, method='linear')
    filled_total_temperature = filled_total_temperature.reshape(processed_total_temperature.shape)

    filled_pressure = np.nan_to_num(filled_pressure)
    filled_total_pressure = np.nan_to_num(filled_total_pressure)
    filled_total_temperature = np.nan_to_num(filled_total_temperature)

    filled_pressure = filled_pressure.T * np.array(mask)
    filled_total_pressure = filled_total_pressure.T * np.array(mask)
    filled_total_temperature = filled_total_temperature.T * np.array(mask)

    plt.imshow(filled_pressure)
    plt.savefig(path + '/filled_input_pressure.png')

    plt.close()

    return filled_pressure,filled_total_pressure,filled_total_temperature


def generate_picture_target(data, path, scale):
    temp_pic = np.zeros((scale, scale))
    pressure = [[[] for _ in range(scale)] for _ in range(scale)]
    processed_pressure = np.zeros((scale, scale))
    total_pressure = [[[] for _ in range(scale)] for _ in range(scale)]
    processed_total_pressure = np.zeros((scale, scale))
    total_temperature = [[[] for _ in range(scale)] for _ in range(scale)]
    processed_total_temperature = np.zeros((scale, scale))
    mach = [[[] for _ in range(scale)] for _ in range(scale)]
    processed_mach = np.zeros((scale, scale))

    mask = load_mask(path + 'mask.csv')
    x_min, x_max, y_min, y_max = get_data_scale('C:/Users\86176\Desktop\python\AICFD\mask/data_scale.csv')
    x_scale = (scale - 1) / (x_max - x_min)
    y_scale = (scale - 1) / (y_max - y_min)

    for i in data:
        i[5] = round((i[5] - x_min) * x_scale)
        i[6] = round((i[6] - y_min) * y_scale)
        pressure[int(i[5])][int(i[6])].append(i[1])
        mach[int(i[5])][int(i[6])].append(i[0])


    for i in range(scale):
        for j in range(scale):
            if not pressure[i][j]:
                continue
            else:
                processed_pressure[i][j] = sum(pressure[i][j]) / len(pressure[i][j])
                processed_mach[i][j] = sum(mach[i][j]) / len(mach[i][j])

                temp_pic[i][j] = 1

    nonzero_position = np.argwhere(processed_pressure != 0)
    nonzero_pressure = processed_pressure[nonzero_position[:, 0], nonzero_position[:, 1]]
    nonzero_mach = processed_mach[nonzero_position[:, 0], nonzero_position[:, 1]]


    x, y = np.meshgrid(np.arange(scale), np.arange(scale))
    pressure_points = np.column_stack([y.ravel(), x.ravel()])
    mach_points = np.column_stack([y.ravel(), x.ravel()])


    filled_pressure = griddata(nonzero_position, nonzero_pressure, pressure_points, method='linear')
    filled_pressure = filled_pressure.reshape(processed_pressure.shape)

    filled_mach = griddata(nonzero_position, nonzero_mach, mach_points, method='linear')
    filled_mach = filled_mach.reshape(processed_total_pressure.shape)

    filled_pressure = np.nan_to_num(filled_pressure)
    filled_mach = np.nan_to_num(filled_mach)

    filled_pressure = filled_pressure.T * np.array(mask)
    filled_mach = filled_mach.T * np.array(mask)

    plt.imshow(filled_pressure)
    plt.savefig(path+'/filled_pressure.png')

    plt.close()

    return filled_pressure,filled_mach


def batch_process(path, scale):
    filenames = os.listdir(path)
    for i, name in tqdm(enumerate(filenames), desc='generate picture...'):
        if i > -1:
            print(i, name)
            data_input = load_data(path + '/' + name + '/STARCCM_3/' + filename_input)
            data_target = load_data(path + '/' + name + '/STARCCM_3/' + filename_target)
            pressure_inputs, total_pressure_inputs, total_temperature_inputs = generate_picture(data_input,
                                                                                                path + '/' + name + '/STARCCM_3/', scale=256)
            pressure_targets, mach_targets = generate_picture_target(data_target,
                                                                      path + '/' + name + '/STARCCM_3/', scale=256)
            with open(path + '/' + name + '/STARCCM_3/pressure_inputs.csv', 'w', encoding='utf-8', newline='') as pressure_inputs_file:
                pressure_inputs_writer = csv.writer(pressure_inputs_file)
                for row in pressure_inputs:
                    pressure_inputs_writer.writerow(row)

            with open(path + '/' + name + '/STARCCM_3/total_pressure_inputs.csv', 'w', encoding='utf-8', newline='') as total_pressure_inputs_file:
                total_pressure_inputs_writer = csv.writer(total_pressure_inputs_file)
                for row in total_pressure_inputs:
                    total_pressure_inputs_writer.writerow(row)

            with open(path + '/' + name + '/STARCCM_3/total_temperature_inputs.csv', 'w', encoding='utf-8', newline='') as total_temperature_inputs_file:
                total_temperature_inputs_writer = csv.writer(total_temperature_inputs_file)
                for row in total_temperature_inputs:
                    total_temperature_inputs_writer.writerow(row)

            with open(path + '/' + name + '/STARCCM_3/pressure_targets.csv', 'w', encoding='utf-8', newline='') as pressure_targets_file:
                pressure_targets_writer = csv.writer(pressure_targets_file)
                for row in pressure_targets:
                    pressure_targets_writer.writerow(row)

            with open(path + '/' + name + '/STARCCM_3/mach_targets.csv', 'w', encoding='utf-8', newline='') as mach_targets_file:
                mach_targets_writer = csv.writer(mach_targets_file)
                for row in mach_targets:
                    mach_targets_writer.writerow(row)



if __name__ == '__main__':
    heeds_path = 'C:/Users/86176/Desktop/python/AICFD/heeds/heeds_Study_1/HEEDS_0'
    filename_input = 'XYZ Internal Table input data.csv'
    filename_target = 'XYZ Internal Table.csv'
    batch_process(heeds_path, 256)