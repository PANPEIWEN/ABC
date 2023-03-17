# @Time    : 2023/3/17 15:56
# @Author  : PEIWEN PAN
# @Email   : 121106022690@njust.edu.cn
# @File    : ABCNet.py
# @Software: PyCharm
import torch
import torch.nn as nn
from model.ABC.Module import conv_block, up_conv, _upsample_like, conv_relu_bn, dconv_block
from einops import rearrange


class Attention(nn.Module):
    def __init__(self, in_dim, in_feature, out_feature):
        super(Attention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=1, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=1, kernel_size=1)
        self.query_line = nn.Linear(in_features=in_feature, out_features=out_feature)
        self.key_line = nn.Linear(in_features=in_feature, out_features=out_feature)
        self.s_conv = nn.Conv2d(in_channels=1, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        q = rearrange(self.query_line(rearrange(self.query_conv(x), 'b 1 h w -> b (h w)')), 'b h -> b h 1')
        k = rearrange(self.key_line(rearrange(self.key_conv(x), 'b 1 h w -> b (h w)')), 'b h -> b 1 h')
        att = rearrange(torch.matmul(q, k), 'b h w -> b 1 h w')
        att = self.softmax(self.s_conv(att))
        return att


class Conv(nn.Module):
    def __init__(self, in_dim):
        super(Conv, self).__init__()
        self.convs = nn.ModuleList([conv_relu_bn(in_dim, in_dim, 1) for _ in range(3)])

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        return x


class DConv(nn.Module):
    def __init__(self, in_dim):
        super(DConv, self).__init__()
        dilation = [2, 4, 2]
        self.dconvs = nn.ModuleList([conv_relu_bn(in_dim, in_dim, dirate) for dirate in dilation])

    def forward(self, x):
        for dconv in self.dconvs:
            x = dconv(x)
        return x


class ConvAttention(nn.Module):
    def __init__(self, in_dim, in_feature, out_feature):
        super(ConvAttention, self).__init__()
        self.conv = Conv(in_dim)
        self.dconv = DConv(in_dim)
        self.att = Attention(in_dim, in_feature, out_feature)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        q = self.conv(x)
        k = self.dconv(x)
        v = q + k
        att = self.att(x)
        out = torch.matmul(att, v)
        return self.gamma * out + v + x


class FeedForward(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FeedForward, self).__init__()
        self.conv = conv_relu_bn(in_dim, out_dim, 1)
        # self.x_conv = nn.Conv2d(in_dim, out_dim, kernel_size=1)
        self.x_conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.conv(x)
        x = self.x_conv(x)
        return x + out


class ConvTransformer(nn.Module):
    def __init__(self, in_dim, out_dim, in_feature, out_feature):
        super(ConvTransformer, self).__init__()
        self.attention = ConvAttention(in_dim, in_feature, out_feature)
        self.feedforward = FeedForward(in_dim, out_dim)

    def forward(self, x):
        x = self.attention(x)
        out = self.feedforward(x)
        return out


class ABCNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, dim=64, ori_h=256, deep_supervision=True, **kwargs):
        super(ABCNet, self).__init__()
        self.deep_supervision = deep_supervision
        filters = [dim, dim * 2, dim * 4, dim * 8, dim * 16]
        features = [ori_h // 2, ori_h // 4, ori_h // 8, ori_h // 16]
        self.maxpools = nn.ModuleList([nn.MaxPool2d(kernel_size=2, stride=2) for _ in range(4)])
        self.Conv1 = conv_block(in_ch=in_ch, out_ch=filters[0])
        # self.Conv1 = ConvTransformer(in_ch, filters[0], pow(ori_h, 2), ori_h)
        self.Convtans2 = ConvTransformer(filters[0], filters[1], pow(features[0], 2), features[0])
        self.Convtans3 = ConvTransformer(filters[1], filters[2], pow(features[1], 2), features[1])
        self.Convtans4 = ConvTransformer(filters[2], filters[3], pow(features[2], 2), features[2])
        self.Conv5 = dconv_block(in_ch=filters[3], out_ch=filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = dconv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

        # --------------------------------------------------------------------------------------------------------------
        self.conv5 = nn.Conv2d(filters[4], out_ch, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(filters[3], out_ch, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(filters[2], out_ch, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(filters[1], out_ch, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(filters[0], out_ch, kernel_size=3, stride=1, padding=1)
        # --------------------------------------------------------------------------------------------------------------

    def forward(self, x):
        e1 = self.Conv1(x)

        e2 = self.maxpools[0](e1)
        e2 = self.Convtans2(e2)

        e3 = self.maxpools[1](e2)
        e3 = self.Convtans3(e3)

        e4 = self.maxpools[2](e3)
        e4 = self.Convtans4(e4)

        e5 = self.maxpools[3](e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        d_s1 = self.conv1(d2)
        d_s2 = self.conv2(d3)
        d_s2 = _upsample_like(d_s2, d_s1)
        d_s3 = self.conv3(d4)
        d_s3 = _upsample_like(d_s3, d_s1)
        d_s4 = self.conv4(d5)
        d_s4 = _upsample_like(d_s4, d_s1)
        d_s5 = self.conv5(e5)
        d_s5 = _upsample_like(d_s5, d_s1)
        if self.deep_supervision:
            outs = [d_s1, d_s2, d_s3, d_s4, d_s5, out]
        else:
            outs = out
        # d1 = self.active(out)

        return outs


if __name__ == '__main__':
    x = torch.randn(8, 3, 256, 256)
    model = ABCNet(ori_h=256)
    print(model(x))

