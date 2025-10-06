import argparse
import logging
import os
import torch
import yaml
import warnings



from util import seed_setup, count_params
from dataset import get_dataloader
from model import get_model
from trainer import train, test, verify

# 忽略 PyTorch 中的 UserWarning
warnings.filterwarnings(action='ignore', category=UserWarning)

# 命令行参数定义
parser = argparse.ArgumentParser(description='blade AI CFD')
parser.add_argument('-c', '--config',
                    type=str,
                    default='C:/Users/86176/Desktop/python/AICFD/config/Blade_Unet.yaml',
                    help='模型配置文件路径')
parser.add_argument('-r', '--resume',
                    action='store_true',
                    help='加载已训练模型')
parser.add_argument('-t', '--test',
                    action='store_true',
                    help='测试模式')
parser.add_argument('-l', '--log',
                    action='store_true',
                    help='记录模式')
parser.add_argument('-v', '--verify',
                    action='store_true',
                    help='验证模式')

if __name__ == '__main__':
    args = parser.parse_args()
    # 解析命令行参数

    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.load(f, yaml.FullLoader)
        # 将 YAML 转为 Python 字典

    # 随机种子 & 设备初始化
    seed_setup(config['data']['seed'])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    # 构建数据加载器
    train_loader, test_loader, verify_loader = get_dataloader(
        config['data']['heeds_path'],
        config['data']['train_size'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers']
    )

    # 构建模型并移动到设备
    model = get_model(config['model']['model_type'], config)
    model = model.to(device)
    print(f'model is of {count_params(model)/1e6:.2f} million parameters')

    # 检查点路径
    ckpt_dir = os.path.join('C:/Users/86176/Desktop/python/AICFD/model', 'ckpts')
    os.makedirs(ckpt_dir, exist_ok=True)
    # 如果开启日志模式，设置 logging
    if args.log:
        os.makedirs(config['log']['base_dir'], exist_ok=True)
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(message)s',
            handlers=[

                logging.FileHandler(os.path.join(config['log']['base_dir'], 'train.log')),
                logging.StreamHandler()
            ]
        )

    # 如果指定加载已有模型
    if args.resume:
        ckpt = os.path.join(ckpt_dir, 'model-best.pt')
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(ckpt)['model'])
        else:
            model.load_state_dict(
                torch.load(ckpt, map_location=torch.device('cpu'))['model']
            )

    # 根据模式选择：测试 / 验证 / 训练
    if args.test:
        test_err, predictions = test(model, test_loader, device)
        logging.info(f'precision:{test_err}')
    elif args.verify:
        ckpt = os.path.join(ckpt_dir, 'model-best.pt')
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(ckpt)['model'])
        else:
            model.load_state_dict(
                torch.load(ckpt, map_location=torch.device('cpu'))['model']
            )
        prec_verify, _ = test(model, verify_loader, device)
        print(f'verify_precision:{prec_verify}')
        verify(model, verify_loader, device, config['data']['heeds_path'])
    else:
        # 默认进入训练流程
        train(model, train_loader, test_loader, args,config, device)
        print('done')