import torch
import torch.nn as nn
import numpy as np
import argparse
import math
import torch.nn.functional as F
from torch.autograd import Variable
from unet.loss import GANLoss

class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample=False):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance


class Discriminator(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.max_conv_dim = 512
        self.num_domains = 1
        self.dim_in = 64

        blocks = []
        blocks += [nn.Conv2d(3, self.dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(args.trainsize)) - 2
        fin_ker = int(args.trainsize / (2 ** repeat_num))
        repeat_num = 6
        for _ in range(repeat_num):
            self.dim_out = min(self.dim_in*2, self.max_conv_dim)
            blocks += [ResBlk(self.dim_in, self.dim_out, downsample=True)]
            self.dim_in = self.dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(self.dim_out, self.dim_out, fin_ker, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv2d(self.dim_out, self.num_domains, 1, 1, 0)]
        self.main = nn.Sequential(*blocks)

    def forward(self, x):
        out = self.main(x)
        out = out.view(out.size(0), -1)  # (batch, num_domains)
        # idx = torch.LongTensor(range(y.size(0)))
        # out = out[idx, y]  # (batch)
        return out

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--trainsize', metavar='E', type=int, default=384)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    net = Discriminator(args).cuda()
    x = torch.randn(6,3,384,384).cuda()
    y = Variable(torch.zeros(x.size(0))).to(dtype=torch.int64).cuda()
    print(y)
    loss = GANLoss("vanilla").cuda()
    out = net(x)
    print(out)
    loss_fake = loss(out, True)

    print(loss_fake)