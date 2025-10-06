import os
import subprocess
import time
import sys
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor


def check_files(path):
    """
    检查文件夹中是否已经生成所有必需的 CSV 文件。
    """
    filenames_to_check = [
        'XYZ blade boundary.csv',
        'XYZ Internal Table.csv',
        'XYZ side boundary.csv'
    ]
    # 检查文件是否存在
    for filename in filenames_to_check:
        if not os.path.isfile(os.path.join(path, filename)):
            return False
    return True


def run_starccm_macro(work_file, macro_name):
    """
    执行 StarCCM+ 宏文件以生成 CSV 文件，并确保进程退出。
    """
    # 检查宏文件是否存在
    macro_file = os.path.join(work_file, macro_name)
    if not os.path.isfile(macro_file):
        print(f"❌ 错误：找不到文件 {macro_file}")
        sys.exit(1)

    # 构建命令，指定完整的 starccm+ 路径
    command = [
        r"C:\Program Files\Siemens\19.02.009\STAR-CCM+19.02.009\star\lib\win64\clang15.0vc14.2\lib\starccm+",
        "-load", os.path.join(work_file, "blade.sim"),  # 加载 sim 文件
        "-m", macro_file  # 执行宏文件
    ]

    # 输出命令并执行
    print(f"正在运行 star-ccm+ 命令: {' '.join(command)}")

    try:
        # 启动 starccm+ 进程
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # 设置超时（例如 30 分钟），避免卡住
        stdout, stderr = process.communicate(timeout=100)  # 30 s超时
        print(stdout)
        if stderr:
            print(f"错误输出: {stderr}")

        # 如果进程正常退出，返回码为 0
        if process.returncode != 0:
            print(f"StarCCM+ 执行失败，退出码: {process.returncode}")
        else:
            print("StarCCM+ 执行成功!")

    except subprocess.TimeoutExpired:
        # 如果超时，终止进程
        print("StarCCM+ 执行超时，正在终止进程...")
        process.kill()
        stdout, stderr = process.communicate()
        print(stdout)
        print(stderr)

    except Exception as e:
        print(f"执行过程中出现错误: {str(e)}")
        process.kill()


def process_design(name, path):
    """
    处理每个设计文件夹，执行宏并检查文件是否生成。
    """
    work_file = os.path.join(path, name, 'STARCCM_3')  # 进入 STAR-CCM+ 计算目录

    # 检查是否已经生成 CSV 文件
    if check_files(work_file):
        print(f"文件夹 {name} 已经生成所需的 CSV 文件，跳过此文件夹。")
        return

    # 运行三个宏命令
    for macro_name in ['Output_data.java', 'output_blade_boundary.java', 'output_side_boundary.java']:
        run_starccm_macro(work_file, macro_name)

    # 检查文件是否存在，如果不存在则等待
    while not check_files(work_file):
        print(f"文件未完全准备，等待10秒后继续检查：{name}")
        time.sleep(10)  # 每 10 秒检查一次文件是否准备好

    # 文件准备好后结束进程
    print(f"所有必要的文件已存在，结束进程：{name}")

    # 检查是否有 starccm+ 进程在运行，如果有则终止它
    task_check_command = 'tasklist /FI "IMAGENAME eq starccm+.exe"'
    task_check_process = subprocess.Popen(task_check_command, shell=True, stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE, text=True)
    stdout, stderr = task_check_process.communicate()
    if "starccm+.exe" in stdout:
        print(f"starccm+ 正在运行，准备终止进程：{name}")
        subprocess.run(['taskkill', '/F', '/IM', 'starccm+.exe'], check=True, capture_output=True, text=True)
    else:
        print(f"没有找到 starccm+ 进程，无需终止：{name}")


def run_starccm(path):
    filenames = os.listdir(path)  # 获取 path 目录下的所有文件和文件夹
    # 使用 ProcessPoolExecutor 来并行处理多个文件夹
    with ProcessPoolExecutor(max_workers=10) as executor:
        # 将每个设计文件夹的处理任务提交给进程池
        futures = [executor.submit(process_design, name, path) for name in filenames]

        # 等待所有任务完成
        for future in tqdm(futures, desc='Processing data...'):
            future.result()  # 获取每个任务的执行结果，若出现异常会抛出


if __name__ == '__main__':
    # 设置你的路径
    heeds_path = r'C:/Users/86176/Desktop/python/AICFD/heeds/heeds_Study_1/HEEDS_0'

    # 开始处理数据
    run_starccm(heeds_path)
