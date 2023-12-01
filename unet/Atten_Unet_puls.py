import torch
from torch import nn


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class input_conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(input_conv_block, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.input_layer(x) + self.input_skip(x)
        return x


class down_conv(nn.Module):
    def __init__(self, ch_in):
        super(down_conv, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down = nn.Sequential(
            nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(ch_in),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.Maxpool(x)
        x2 = self.down(x)
        return x1 + x2


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class AttU_Net_plus(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, filters=[32, 64, 128, 256, 512, 1024]):
        super(AttU_Net_plus, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.down_conv1 = down_conv(filters[1])
        self.down_conv2 = down_conv(filters[2])
        self.down_conv3 = down_conv(filters[3])
        self.down_conv4 = down_conv(filters[4])

        self.Conv1 = input_conv_block(ch_in=n_channels, ch_out=filters[1])
        self.Conv2 = conv_block(ch_in=filters[1], ch_out=filters[2])
        self.Conv3 = conv_block(ch_in=filters[2], ch_out=filters[3])
        self.Conv4 = conv_block(ch_in=filters[3], ch_out=filters[4])
        self.Conv5 = conv_block(ch_in=filters[4], ch_out=filters[5])

        self.Up5 = up_conv(ch_in=filters[5], ch_out=filters[4])
        self.Att5 = Attention_block(F_g=filters[4], F_l=filters[4], F_int=filters[3])
        self.Up_conv5 = conv_block(ch_in=filters[5], ch_out=filters[4])

        self.Up4 = up_conv(ch_in=filters[4], ch_out=filters[3])
        self.Att4 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_conv4 = conv_block(ch_in=filters[4], ch_out=filters[3])

        self.Up3 = up_conv(ch_in=filters[3], ch_out=filters[2])
        self.Att3 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_conv3 = conv_block(ch_in=320, ch_out=filters[2])

        self.Up2 = up_conv(ch_in=filters[2], ch_out=filters[1])
        self.Att2 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_conv2 = conv_block(ch_in=filters[3], ch_out=filters[1])

        self.Up1 = up_conv(ch_in=filters[1], ch_out=filters[0])
        self.Conv_1x1 = nn.Conv2d(filters[1], n_classes, kernel_size=1, stride=1, padding=0)

        # 深度级连
        self.d5_cat_d3 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear'),
            nn.Conv2d(filters[5], filters[1], 3, padding=1),
            nn.BatchNorm2d(filters[1]),
            nn.ReLU(inplace=True)
        )

        self.d5_cat_d2 = nn.Sequential(
            nn.Upsample(scale_factor=8, mode='bilinear'),
            nn.Conv2d(filters[5], filters[1], 3, padding=1),
            nn.BatchNorm2d(filters[1]),
            nn.ReLU(inplace=True)
        )

        # self.d5_cat_d1 = nn.Sequential(
        #     nn.Upsample(scale_factor=16, mode='bilinear'),
        #     nn.Conv2d(filters[5], filters[1], 3, padding=1),
        #     nn.BatchNorm2d(filters[1]),
        #     nn.ReLU(inplace=True)
        # )

        self.d4_cat_d2 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear'),
            nn.Conv2d(filters[4], filters[1], 3, padding=1),
            nn.BatchNorm2d(filters[1]),
            nn.ReLU(inplace=True)
        )

        # self.d4_cat_d1 = nn.Sequential(
        #     nn.Upsample(scale_factor=8, mode='bilinear'),
        #     nn.Conv2d(filters[4], filters[1], 3, padding=1),
        #     nn.BatchNorm2d(filters[1]),
        #     nn.ReLU(inplace=True)
        # )

        self.d3_cat_d1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(filters[3], filters[1], 3, padding=1),
            nn.BatchNorm2d(filters[1]),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.down_conv1(x1)
        x2 = self.Conv2(x2)

        x3 = self.down_conv2(x2)
        x3 = self.Conv3(x3)

        x4 = self.down_conv3(x3)
        x4 = self.Conv4(x4)

        x5 = self.down_conv4(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)

        d5_cat_d3 = self.d5_cat_d3(d5)
        d5_cat_d2 = self.d5_cat_d2(d5)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)

        d4_cat_d2 = self.d4_cat_d2(d4)

        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)

        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)

        # d5 cat d3
        d3 = torch.cat((d3, d5_cat_d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)

        # d5 cat d2 && d4 cat d2
        d2 = torch.cat((d2, d4_cat_d2, d5_cat_d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


if __name__ == '__main__':
    model = AttU_Net_plus()
    input = torch.randn(1, 3, 320, 320)
    out = model(input)
    print(out.size())
