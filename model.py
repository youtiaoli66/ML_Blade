import torch
import torch.nn as nn


def blockUNet(in_c, out_c, name, transposed=False, bn=True, relu=True, size=4, pad=1, dropout=0.):
    block = nn.Sequential()
    if relu:
        block.add_module('%s_relu' % name, nn.ReLU(inplace=True))
    else:
        block.add_module('%s_Leakyrelu' % name, nn.LeakyReLU(negative_slope=0.2, inplace=True))

    if not transposed:
        block.add_module('%s_conv' % name, nn.Conv2d(in_c, out_c, kernel_size=size, stride=2, padding=pad, bias=True))#bias偏置项帮助学习
    else:
        block.add_module('%s_upsample' % name, nn.Upsample(scale_factor=2, mode='bilinear'))
        block.add_module('%s_tconv' % name, nn.Conv2d(in_c, out_c, kernel_size=size-1, stride=1, padding=pad, bias=True))
    if bn:
        block.add_module('%s_bn' % name, nn.BatchNorm2d(out_c))
    if dropout:
        block.add_module('%s_dropout' % name, nn.Dropout2d(p=dropout))

    return block


class UNet(nn.Module):
    def __init__(self, ChannelExponent=6, dropout=0.):
        super(UNet, self).__init__()
        channels = int(2 ** ChannelExponent + 0.5)

        self.layer1 = nn.Sequential()
        self.layer1.add_module('layer1_relu', nn.ReLU(inplace=True))
        self.layer1.add_module('layer1_tconv',nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=4, stride=2,padding=1, bias=True))


        self.layer2 = blockUNet(channels , channels * 2, name='layer2', transposed=False, bn=True, relu=False,
                                dropout=dropout)
        self.layer3 = blockUNet(channels * 2, channels * 4, name='layer3', transposed=False, bn=True, relu=False,
                                dropout=dropout)
        self.layer4 = blockUNet(channels * 4, channels * 8, name='layer4', transposed=False, bn=True, relu=False,
                                dropout=dropout,size=4)
        self.layer5 = blockUNet(channels * 8, channels * 8, name='layer5', transposed=False, bn=True, relu=False,
                                dropout=dropout,size=2,pad=0)
        self.layer6 = blockUNet(channels * 8, channels * 8, name='layer6', transposed=False, bn=True, relu=False,
                                dropout=dropout,size=2,pad=0)

        self.dlayer6 = blockUNet(channels * 8, channels * 8, name='dlayer6', transposed=True, bn=True, relu=True,
                                dropout=dropout,size=2,pad=0)
        self.dlayer5 = blockUNet(channels * 16, channels * 8, name='dlayer6', transposed=True, bn=True, relu=True,
                                 dropout=dropout, size=2, pad=0)
        self.dlayer4 = blockUNet(channels * 16, channels * 4, name='dlayer6', transposed=True, bn=True, relu=True,
                                 dropout=dropout, size=2, pad=0)
        self.dlayer3 = blockUNet(channels * 8, channels * 2, name='dlayer6', transposed=True, bn=True, relu=True,
                                 dropout=dropout, size=2, pad=0)
        self.dlayer2 = blockUNet(channels * 4, channels , name='dlayer6', transposed=True, bn=True, relu=True,
                                 dropout=dropout, size=2, pad=0)

        self.dlayer1 = nn.Sequential()
        self.dlayer1.add_module('dlayer1_relu',nn.ReLU(inplace=True))
        self.dlayer1.add_module('dlayer1_tconv',nn.ConvTranspose2d(in_channels=channels*2, out_channels=2, kernel_size=4, stride=2, padding=1, bias=True))



    def forward(self, x):

        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)

        dout6 = self.dlayer6(out6)
        dout6_out5 = torch.cat([dout6,out5],1)
        dout5 = self.dlayer5(dout6_out5)
        dout5_out4 = torch.cat([dout5,out4],1)
        dout4 = self.dlayer4(dout5_out4)
        dout4_out3 = torch.cat([dout4,out3],1)
        dout3 = self.dlayer3(dout4_out3)
        dout3_out2 = torch.cat([dout3,out2],1)
        dout2 = self.dlayer2(dout3_out2)
        dout2_out1 = torch.cat([dout2,out1],1)
        dout1 = self.dlayer1(dout2_out1)

        return dout1


def get_model(model_type, config):
    if model_type == 'UNet':
        model = UNet(config['model']['expo'], config['model']['dropout'])
    else:
        raise NotImplementedError
    return model

