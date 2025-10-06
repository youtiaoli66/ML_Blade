import csv

# 读取原始文件，处理数据并写入新的 CSV 文件
def add_commas_to_data(input_file, output_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    # 处理每一行数据
    processed_lines = []
    for line in lines:
        # 删除两端空白字符，并以空格分割数据
        numbers = line.strip().split()
        # 将每行的数据作为一行列表添加到 processed_lines
        processed_lines.append(numbers)

    # 将处理后的数据写入新的 CSV 文件
    with open(output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(processed_lines)

# 输入原始文件和输出文件路径
input_file = r'C:/Users/86176/Desktop/Xfoil/241.txt'  # 请替换为您的输入文件路径
output_file = r'C:/Users/86176/Desktop/Xfoil/241.csv'  # 请替换为您的输出文件路径，输出为CSV文件

# 执行处理
add_commas_to_data(input_file, output_file)
