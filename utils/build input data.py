import os
import csv
from tqdm import tqdm

#读取csv文件
def load_data(path):
    data=[]
    with open(path,'r',encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        header=next(csv_reader)
        for row in csv_reader:
    
            data.append(list(row))
    return data

#构建训练输入数据，根据结果文件创建输入文件
def build_input_file(path,filename,input_filename,inlet_total_pressure,inlet_total_temperature,outlet_pressure):
    filenames = os.listdir(path)
    for name in tqdm(filenames,'read data...'):
        path_full = path +'/'+name + '/STARCCM_3/' + filename
        data = []
        result_data = load_data(path_full)
        for row in result_data:
            input_data = row[-3:]
            dt = [str(inlet_total_pressure),str(inlet_total_temperature),str(outlet_pressure)]
            input_data = input_data + dt
            data.append(input_data)
        header = ['x','y','z','inlet_total_pressure','inlet_total_temperature','outlet_pressure']
        #在首行插入标题
        data.insert(0,header)
        with open(path+'/'+name+'/STARCCM_3/'+input_filename,'w',encoding='utf-8') as file_result:
            csv_writer = csv.writer(file_result)
            for row in data:
                csv_writer.writerow(row)




if __name__ == '__main__':
    inlet_total_pressure = 122000
    inlet_total_temperature = 2
    outlet_pressure = 103400
    result_filename = 'XYZ Internal Table.csv'
    input_filename = 'XYZ Internal Table input data.csv'
    heeds_path = 'C:/Users/86176/Desktop/python/AICFD/heeds/heeds_Study_1/HEEDS_0'

    build_input_file(heeds_path, result_filename, input_filename, inlet_total_pressure, inlet_total_temperature, outlet_pressure)

