import argparse

# 创建解析器对象
parser = argparse.ArgumentParser(description='简单的演示程序')

# 添加参数
parser.add_argument('-n', '--name', type=str, help='你的名字')
parser.add_argument('-a', '--age', type=int, help='你的年龄')
parser.add_argument('-v', '--verbose', action='store_true', help='是否显示更多信息')

# 解析命令行参数
args = parser.parse_args()

# 使用参数
print(f"你好，{args.name}！你今年 {args.age} 岁了。")

if args.verbose:
    print("你使用了 verbose 模式，会显示更多信息。")
