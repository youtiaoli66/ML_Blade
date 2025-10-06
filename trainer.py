import os
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.optim import AdamW

from loss import precision, data_loss
from util import save_checkpoint


def train(model, train_loader, test_loader, args, config, device):
    best_prec = np.array([-np.inf, -np.inf])
    eval_prec = np.array([-np.inf, -np.inf])

    # AdamW() 优化器；lr 学习率；weight_decay 权重衰减系数
    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.0005)
    # scheduler 学习策略；gamma 是学习衰减因子
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    pbar = tqdm(range(100), desc='training')
    for epoch in pbar:
        model.train()
        for inputs, targets, mask in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = data_loss(outputs, targets, mask=mask)
            loss.backward()
            optimizer.step()


        # 评估
        eval_prec, predictions = test(model, test_loader, device)
        pbar.set_description(
            f'train_loss:{loss.item():.2e} | val_prec:{eval_prec}% | best_prec:{best_prec}%'
        )

        # 保存最佳模型
        if eval_prec.mean() > best_prec.mean():
            best_prec = eval_prec
            ckpt_path = os.path.join(
                'C:/Users/86176/Desktop/python/AICFD/model', 'ckpts', 'model-best.pt'
            )
            save_checkpoint(ckpt_path, model, optimizer, scheduler)

        # 日志
        if args.log:
            logging.info(
                f'Epoch {epoch + 1} | val_prec%{eval_prec}% | best_prec%{best_prec}%'
            )

        # 每 10 epoch 额外保存一次
        if (epoch + 1) % 10 == 0 and epoch > 0:
            ckpt_path = os.path.join(
                'C:/Users/86176/Desktop/python/AICFD/model', 'ckpts', f'model-{epoch + 1}.pt'
            )
            save_checkpoint(ckpt_path, model, optimizer, scheduler)
        scheduler.step()
    pbar.close()


@torch.no_grad()  # 不计算梯度
def test(model, test_loader, device):
    model.eval()
    precisions, field_precs = [], []

    for inputs, targets, mask in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        mask = mask.to(device)

        outputs = model(inputs)
        precisions.append(outputs.cpu().numpy())

        # 求每个通道的精度
        mask = mask.unsqueeze(1).repeat(1, 2, 1, 1).permute(1,0,2,3)
        field_outputs = (
            outputs.permute(1,0,2,3).flatten(-2)[mask.flatten(-2).bool()].reshape(2, -1).T
        )
        field_targets = (
            targets.permute(1,0,2,3).flatten(-2)[mask.flatten(-2).bool()].reshape(2, -1).T
        )

        denorm = (
            test_loader.dataset.dataset.transform_targets.denorm \
            if isinstance(test_loader.dataset, torch.utils.data.Subset)
            else test_loader.dataset.transform_targets.denorm
        )

        field_prec = precision(field_outputs, field_targets, denorm)

        field_precs.append(field_prec.cpu().numpy())

    avg_prec = np.mean(field_precs, 0)          # 计算列平均
    precisions = np.concatenate(precisions, 0)  # 拼接 batch 维
    return np.round(avg_prec, 2), precisions          # 保留两位小数


@torch.no_grad()
def verify(model, verify_loader, device, path):
    """
    推理并将结果、目标以及相对/绝对误差分别保存为图片：
        mach_output.png / pressure_output.png
        mach_targets.png / pressure_targets.png
        mach_relative_error.png / pressure_relative_error.png
        mach_absolute_error.png / pressure_absolute_error.png
    """
    model.eval()

    for idx in range(len(verify_loader.dataset)):
        inputs, targets, mask = verify_loader.dataset.dataset[
            verify_loader.dataset.indices[idx]
        ]
        filename = verify_loader.dataset.dataset.filenames[
            verify_loader.dataset.indices[idx]
        ]

        inputs = inputs.unsqueeze(0).to(device)
        targets = targets.unsqueeze(0).to(device)
        mask = mask.unsqueeze(0).to(device)

        outputs = model(inputs)

        # ---------- 后处理 ----------


        mask_exp = mask.unsqueeze(1).repeat(1, 2, 1, 1).permute(1,0,2,3)
        field_outputs = (
            outputs.permute(1,0,2,3).flatten(-2)[mask_exp.flatten(-2).bool()].reshape(2, -1).T
        )
        field_targets = (
            targets.permute(1,0,2,3).flatten(-2)[mask_exp.flatten(-2).bool()].reshape(2, -1).T
        )

        denorm = (
            verify_loader.dataset.dataset.transform_targets.denorm\
            if isinstance(verify_loader.dataset, torch.utils.data.Subset)
            else verify_loader.dataset.transform_targets.denorm

        )
        field_outputs = denorm(field_outputs)
        field_targets = denorm(field_targets)


        field_error = abs(field_outputs - field_targets)  # 绝对误差

        field_prec = field_error / (field_targets)
        # # 计算相对误差（逐像素）
        # img_data = []
        # for image, ftarget in zip(field_error, field_targets):
        #     for img, ttarget in zip(image, ftarget):
        #         sample_mask = (img != 0).bool().to(device)
        #         field_prec1 = torch.zeros_like(img).to(device)
        #         field_prec1[sample_mask] = abs(img[sample_mask] / ttarget[sample_mask])
        #         img_data.append(field_prec1.cpu().numpy())
        # field_prec = torch.from_numpy(np.array(img_data)).unsqueeze(0)  # 相对误差
#####################################################################################################
        # 设定 H, W
        _, _, H, W = outputs.shape

        # 初始化填充图：全 0 或 NaN
        field_outputs_full = torch.zeros(2, H, W, device=field_outputs.device)
        field_targets_full = torch.zeros(2, H, W, device=field_targets.device)
        field_error_full = torch.zeros(2, H, W, device=field_targets.device)
        field_prec_full = torch.zeros(2, H, W, device=field_targets.device)

        # 展开 mask 取 index
        mask_2d = mask[0]  # [H, W]
        mask_flat = mask_2d.flatten()
        idx = torch.nonzero(mask_flat, as_tuple=False).squeeze()  # [N]

        # 把 denorm 后的值分别填回两个通道
        field_outputs_full = field_outputs_full.view(2, -1)  # [2, H*W]
        field_targets_full = field_targets_full.view(2, -1)  # [2, H*W]
        field_error_full = field_error_full.view(2, -1)
        field_prec_full = field_prec_full.view(2, -1)




        field_outputs_full[:, idx] = field_outputs.T  # T 是因为 field_outputs 是 [N, 2]
        field_targets_full[:, idx] = field_targets.T
        field_error_full[:, idx] = field_error.T
        field_prec_full[:, idx] = field_prec.T



        # reshape 回 [2, H, W]
        field_outputs = field_outputs_full.view(2, H, W)
        field_targets = field_targets_full.view(2, H, W)
        field_error = field_error_full.view(2, H, W)
        field_prec = field_prec_full.view(2, H, W)


######################################################################################################


        # 把四组数据整理成方便遍历的列表
        images1 = field_outputs.cpu().numpy()
        images2 = field_targets.cpu().numpy()
        images3 = field_prec.cpu().numpy()
        images4 = field_error.cpu().numpy()

        # ----------- 逐通道保存图片 -------------
        # 1) 预测值
        for i, image in enumerate(images1):
            plt.figure()
            plt.imshow(image)
            if i == 0:
                plt.pcolormesh(image, vmin=0, vmax=2)
                plt.colorbar()
                plt.savefig(
                    os.path.join(path, filename, 'STARCCM_3', 'mach_output.png')
                )
            else:
                plt.pcolormesh(image, vmin=0, vmax=122000)
                plt.colorbar()
                plt.savefig(
                    os.path.join(
                        path, filename, 'STARCCM_3', 'pressure_output.png'
                    )
                )
            plt.close()

        # 2) 目标值
        for i, image in enumerate(images2):
            plt.figure()
            plt.imshow(image)
            if i == 0:
                plt.pcolormesh(image, vmin=0, vmax=2)
                plt.colorbar()
                plt.savefig(
                    os.path.join(path, filename, 'STARCCM_3', 'mach_targets.png')
                )
            else:
                plt.pcolormesh(image, vmin=0, vmax=122000)
                plt.colorbar()
                plt.savefig(
                    os.path.join(
                        path, filename, 'STARCCM_3', 'pressure_targets.png'
                    )
                )
            plt.close()
        # 3) 相对误差
        for i, image in enumerate(images3):
            plt.figure()
            plt.imshow(image)
            if i == 0:
                plt.pcolormesh(image, vmin=0, vmax=1)
                plt.colorbar()
                plt.savefig(
                    os.path.join(
                        path, filename, 'STARCCM_3', 'mach_relative_error.png'
                    )
                )
                # ✅ 保存 Mach 相对误差为 CSV
                np.savetxt(
                    os.path.join(path, filename, 'STARCCM_3', 'mach_relative_error.csv'),
                    image, delimiter=',', fmt='%.6f'
                )
            else:
                plt.pcolormesh(image, vmin=0, vmax=1)
                plt.colorbar()
                plt.savefig(
                    os.path.join(
                        path,
                        filename,
                        'STARCCM_3',
                        'pressure_relative_error.png',
                    )
                )
                # ✅ 保存 Pressure 相对误差为 CSV
                np.savetxt(
                    os.path.join(path, filename, 'STARCCM_3', 'pressure_relative_error.csv'),
                    image, delimiter=',', fmt='%.6f'
                )
            plt.close()


        # 4) 绝对误差
        for i, image in enumerate(images4):
            plt.figure()
            plt.imshow(image)
            if i == 0:
                plt.pcolormesh(image, vmin=0, vmax=2)
                plt.colorbar()
                plt.savefig(
                    os.path.join(
                        path, filename, 'STARCCM_3', 'mach_absolute_error.png'
                    )
                )
            else:
                plt.pcolormesh(image, vmin=0, vmax=122000)
                plt.colorbar()
                plt.savefig(
                    os.path.join(
                        path,
                        filename,
                        'STARCCM_3',
                        'pressure_absolute_error.png',
                    )
                )
            plt.close()
