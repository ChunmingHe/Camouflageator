import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.Res2Net_v1b import res2net50_v1b_26w_4s
from lib.resnet import resnet50_v1
from lib.Modules import *
import time
import timm
from thop import profile


class Network(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channels=32, imagenet_pretrained=True):
        super(Network, self).__init__()
        # ---- ResNet Backbone ----
        self.shared_encoder = timm.create_model(model_name="resnet50", pretrained=True, in_chans=3, features_only=True)
        self.GCM = GCM(256, channels)
        self.GPM = GPM()
        self.REM = REM(channels, channels)  # HH+HL+LH+LL
        self.REM2 = REM2(channels, channels)  # HH+HLLH+LL
        self.REM3 = REM3(channels, channels)  # HH+HLLH+LL+aspp
        self.REM4 = REM4(channels, channels)  # HH+HLLH+HLLH+LL
        self.REM5 = REM5(channels, channels)  # HH+HLLH+LL+LL 
        self.REM6 = REM6(channels, channels)  # HH+HH+HLLH+LL
        self.REM7 = REM7(channels, channels)  # HH+HH+HLLH 多尺度

    def forward(self, x):
        image = x
        # Feature Extraction
        en_feats = self.shared_encoder(x)
        x0, x1, x2, x3, x4 = en_feats
        LL, LH, HL, HH = self.GCM(x1, x2, x3, x4)
        prior_cam = self.GPM(x4)  # bs, 1, 12, 12
        pred_0 = F.upsample(prior_cam, size=image.size()[2:], mode='bilinear', align_corners=False)
        # pred_hh_1,pred_hh_2, pred_hllh, pred_ll, pred_b_hh_1,pred_b_hh_2, pred_b_hllh, pred_b_ll= self.REM6([LL,LH,HL,HH], prior_cam, image)
        # return pred_0, pred_hh_1,pred_hh_2, pred_hllh, pred_ll, pred_b_hh_1,pred_b_hh_2, pred_b_hllh, pred_b_ll  #hh*2
        # pred_hh, pred_hllh, pred_ll_1,pred_ll_2, pred_b_hh, pred_b_hllh, pred_b_ll_1, pred_b_ll_2= self.REM5([LL,LH,HL,HH], prior_cam, image)
        # return pred_0, pred_hh, pred_hllh, pred_ll_1,pred_ll_2, pred_b_hh, pred_b_hllh, pred_b_ll_1, pred_b_ll_2  # ll*2
        # pred_hh, pred_hllh_1,pred_hllh_2, pred_ll, pred_b_hh, pred_b_hllh_1,pred_b_hllh_2, pred_b_ll = self.REM4([LL,LH,HL,HH], prior_cam, image)
        # return pred_0, pred_hh, pred_hllh_1,pred_hllh_2, pred_ll, pred_b_hh, pred_b_hllh_1,pred_b_hllh_2, pred_b_ll # hllh*2
        pred_hh, pred_hllh, pred_ll, pred_b_hh, pred_b_hllh, pred_b_ll = self.REM7([LL, LH, HL, HH], prior_cam, image)
        return pred_0, pred_hh, pred_hllh, pred_ll, pred_b_hh, pred_b_hllh, pred_b_ll


'''
res2net
'''


class Network2(nn.Module):
    # res2net based encoder decoder
    def __init__(self, channels=32, imagenet_pretrained=True):
        super(Network2, self).__init__()
        # ---- ResNet Backbone ----
        self.resnet = res2net50_v1b_26w_4s(pretrained=imagenet_pretrained)
        self.GCM = GCM(256, channels)
        self.GPM = GPM()
        self.REM = REM(channels, channels)
        self.REM2 = REM2(channels, channels)

    def forward(self, x):
        image = x
        # Feature Extraction
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)  # bs, 64, 96 ,96
        x1 = self.resnet.layer1(x)  # bs, 256, 96 ,96
        x2 = self.resnet.layer2(x1)  # bs, 512, 48 ,48
        x3 = self.resnet.layer3(x2)  # bs, 1024, 24 ,24
        x4 = self.resnet.layer4(x3)  # bs, 2048, 12 ,12
        LL, LH, HL, HH = self.GCM(x1, x2, x3, x4)
        prior_cam = self.GPM(x4)  # bs, 1, 12, 12
        pred_0 = F.upsample(prior_cam, size=image.size()[2:], mode='bilinear', align_corners=False)
        pred_hh, pred_hllh, pred_ll, pred_b_hh, pred_b_hllh, pred_b_ll = self.REM2([LL, LH, HL, HH], prior_cam, image)
        return pred_0, pred_hh, pred_hllh, pred_ll, pred_b_hh, pred_b_hllh, pred_b_ll


'''
resnet50+sub-pixel
'''


class Network3(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channels=32, imagenet_pretrained=True):
        super(Network3, self).__init__()
        # ---- ResNet Backbone ----
        self.shared_encoder = timm.create_model(model_name="resnet50", pretrained=True, in_chans=3, features_only=True)
        self.GCM = GCM(256, channels)
        self.GPM = GPM()
        self.REM8 = REM8(channels, channels)  # sub-pixel

    def forward(self, x):
        image = x
        # Feature Extraction
        en_feats = self.shared_encoder(x)
        x0, x1, x2, x3, x4 = en_feats
        LL, LH, HL, HH = self.GCM(x1, x2, x3, x4)
        # print(LL.shape)
        prior_cam = self.GPM(x4)
        # print(prior_cam.shape)              # bs, 1, 12, 12
        pred_0 = F.upsample(prior_cam, size=image.size()[2:], mode='bilinear', align_corners=False)
        pred_hh, pred_hllh, pred_ll, pred_b_hh, pred_b_hllh, pred_b_ll = self.REM8([LL, LH, HL, HH], prior_cam, image)
        return pred_0, pred_hh, pred_hllh, pred_ll, pred_b_hh, pred_b_hllh, pred_b_ll


'''
resnet50-GCM
'''


class Network4(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channels=32, imagenet_pretrained=True):
        super(Network4, self).__init__()
        # ---- ResNet Backbone ----
        self.shared_encoder = timm.create_model(model_name="resnet50", pretrained=True, in_chans=3, features_only=True)
        self.GCM2 = GCM2(256, channels)
        self.GPM = GPM()
        self.REM9 = REM9(channels, channels)  # HH+HL+LH+LL 多尺度

    def forward(self, x):
        image = x
        # Feature Extraction
        en_feats = self.shared_encoder(x)
        x0, x1, x2, x3, x4 = en_feats
        f1, f2, f3, f4 = self.GCM2(x1, x2, x3, x4)
        prior_cam = self.GPM(x4)  # bs, 1, 12, 12
        pred_0 = F.upsample(prior_cam, size=image.size()[2:], mode='bilinear', align_corners=False)
        f4, f3, f2, f1, bound_f4, bound_f3, bound_f2, bound_f1 = self.REM9([f1, f2, f3, f4], prior_cam, image)
        return pred_0, f4, f3, f2, f1, bound_f4, bound_f3, bound_f2, bound_f1


"""
    f1+HH,f4+LL
"""


class Network5(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channels=32, imagenet_pretrained=True):
        super(Network5, self).__init__()
        # ---- ResNet Backbone ----
        self.shared_encoder = timm.create_model(model_name="resnet50", pretrained=True, in_chans=3, features_only=True)
        self.GCM3 = GCM3(256, channels)
        self.GPM = GPM()
        self.REM9 = REM9(channels, channels)  # HH+HL+LH+LL 多尺度

        self.LL_down = nn.Sequential(
            BasicConv2d(channels, channels, stride=2, kernel_size=3, padding=1),
            BasicConv2d(channels, channels, stride=2, kernel_size=3, padding=1)
        )
        self.dePixelShuffle = torch.nn.PixelShuffle(2)
        # self.one_conv_f4_ll = nn.Conv2d(in_channels=channels*2, out_channels=channels, kernel_size=1)
        self.one_conv_f1_hh = nn.Conv2d(in_channels=channels + channels // 4, out_channels=channels, kernel_size=1)

    def forward(self, x):
        image = x
        # Feature Extraction
        en_feats = self.shared_encoder(x)
        x0, x1, x2, x3, x4 = en_feats
        LL, LH, HL, HH, f1, f2, f3, f4 = self.GCM3(x1, x2, x3, x4)

        HH_up = self.dePixelShuffle(HH)  # 64*48*48 ->16*96*96
        f1_HH = torch.cat([HH_up, f1], dim=1)  # 16*96*96+64*96*96->80*96*96
        f1_HH = self.one_conv_f1_hh(f1_HH)  # 80*96*96 -> 64*96*96

        # LL_down =self.LL_down(LL)  # 64*48*48 ->64*12*12
        # f4_LL = torch.cat([LL_down, f4], dim=1) # 64*12*12+64*12*12->128*12*12
        # f4_LL = self.one_conv_f4_ll(f4_LL) # 128*12*12-> 64*12*12

        prior_cam = self.GPM(x4)  # bs, 1, 12, 12
        pred_0 = F.upsample(prior_cam, size=image.size()[2:], mode='bilinear', align_corners=False)
        f4, f3, f2, f1, bound_f4, bound_f3, bound_f2, bound_f1 = self.REM9([f1_HH, f2, f3, f4], prior_cam, image)
        return pred_0, f4, f3, f2, f1, bound_f4, bound_f3, bound_f2, bound_f1


'''
resnet50-noGCM-TFD
'''


class Network6(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channels=32, imagenet_pretrained=True):
        super(Network6, self).__init__()
        # ---- ResNet Backbone ----
        self.shared_encoder = timm.create_model(model_name="resnet50", pretrained=True, in_chans=3, features_only=True)
        self.GCM2 = GCM2(256, channels)
        self.GPM = GPM()
        self.REM10 = REM10(channels, channels)  # HH+HL+LH+LL 多尺度

    def forward(self, x):
        image = x
        # Feature Extraction
        en_feats = self.shared_encoder(x)
        x0, x1, x2, x3, x4 = en_feats
        f1, f2, f3, f4 = self.GCM2(x1, x2, x3, x4)
        prior_cam = self.GPM(x4)  # bs, 1, 12, 12
        pred_0 = F.upsample(prior_cam, size=image.size()[2:], mode='bilinear', align_corners=False)
        f4, f3, f2, f1, bound_f4, bound_f3, bound_f2, bound_f1 = self.REM10([f1, f2, f3, f4], prior_cam, image)
        return pred_0, f4, f3, f2, f1, bound_f4, bound_f3, bound_f2, bound_f1


'''
resnet50- LL, LH, HL, HH, f1, f2, f3, f4-TFD
'''


class Network7(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channels=32, imagenet_pretrained=True):
        super(Network7, self).__init__()
        self.shared_encoder = timm.create_model(model_name="resnet50", pretrained=True, in_chans=3, features_only=True)
        self.GCM3 = GCM3(256, channels)
        self.GPM = GPM()
        self.REM10 = REM10(channels, channels)

        self.LL_down = nn.Sequential(
            BasicConv2d(channels, channels, stride=2, kernel_size=3, padding=1),
            BasicConv2d(channels, channels, stride=2, kernel_size=3, padding=1)
        )
        # self.dePixelShuffle = torch.nn.PixelShuffle(2)
        self.one_conv_f4_ll = nn.Conv2d(in_channels=channels * 2, out_channels=channels, kernel_size=1)
        # self.one_conv_f1_hh = nn.Conv2d(in_channels=channels+channels//4, out_channels=channels, kernel_size=1)

    def forward(self, x):
        image = x
        # Feature Extraction
        en_feats = self.shared_encoder(x)
        x0, x1, x2, x3, x4 = en_feats
        LL, LH, HL, HH, f1, f2, f3, f4 = self.GCM3(x1, x2, x3, x4)

        # HH_up = self.dePixelShuffle(HH)  # 64*48*48 ->16*96*96
        # f1_HH = torch.cat([HH_up, f1], dim=1)  # 16*96*96+64*96*96->80*96*96
        # f1_HH = self.one_conv_f1_hh(f1_HH) # 80*96*96 -> 64*96*96

        LL_down = self.LL_down(LL)  # 64*48*48 ->64*12*12
        f4_LL = torch.cat([LL_down, f4], dim=1)  # 64*12*12+64*12*12->128*12*12
        f4_LL = self.one_conv_f4_ll(f4_LL)  # 128*12*12-> 64*12*12

        prior_cam = self.GPM(x4)  # bs, 1, 12, 12
        pred_0 = F.upsample(prior_cam, size=image.size()[2:], mode='bilinear', align_corners=False)
        f4, f3, f2, f1, bound_f4, bound_f3, bound_f2, bound_f1 = self.REM10([f1, f2, f3, f4_LL], prior_cam, image)
        return pred_0, f4, f3, f2, f1, bound_f4, bound_f3, bound_f2, bound_f1


'''
resnet50- LL, LH, HL, HH, f1, f2, f3, f4-ODE
'''


class Network8(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channels=32, imagenet_pretrained=True):
        super(Network8, self).__init__()
        self.shared_encoder = timm.create_model(model_name="resnet50", pretrained=True, in_chans=3, features_only=True)
        self.GCM3 = GCM3(256, channels)
        self.GPM = GPM()
        self.REM11 = REM11(channels, channels)

        # self.LL_down = nn.Sequential(
        #     BasicConv2d(channels, channels,stride=2, kernel_size=3, padding=1),
        #     BasicConv2d(channels, channels,stride=2, kernel_size=3, padding=1)
        # )
        self.dePixelShuffle = torch.nn.PixelShuffle(2)
        # self.one_conv_f4_ll = nn.Conv2d(in_channels=channels*2, out_channels=channels, kernel_size=1)
        self.one_conv_f1_hh = nn.Conv2d(in_channels=channels + channels // 4, out_channels=channels, kernel_size=1)

    def forward(self, x):
        image = x
        # Feature Extraction
        en_feats = self.shared_encoder(x)
        x0, x1, x2, x3, x4 = en_feats
        LL, LH, HL, HH, f1, f2, f3, f4 = self.GCM3(x1, x2, x3, x4)

        HH_up = self.dePixelShuffle(HH)  # 64*48*48 ->16*96*96
        f1_HH = torch.cat([HH_up, f1], dim=1)  # 16*96*96+64*96*96->80*96*96
        f1_HH = self.one_conv_f1_hh(f1_HH)  # 80*96*96 -> 64*96*96

        # LL_down =self.LL_down(LL)  # 64*48*48 ->64*12*12
        # f4_LL = torch.cat([LL_down, f4], dim=1) # 64*12*12+64*12*12->128*12*12
        # f4_LL = self.one_conv_f4_ll(f4_LL) # 128*12*12-> 64*12*12

        prior_cam = self.GPM(x4)  # bs, 1, 12, 12
        pred_0 = F.upsample(prior_cam, size=image.size()[2:], mode='bilinear', align_corners=False)
        f4, f3, f2, f1, bound_f4, bound_f3, bound_f2, bound_f1 = self.REM11([f1_HH, f2, f3, f4], prior_cam, image)
        return pred_0, f4, f3, f2, f1, bound_f4, bound_f3, bound_f2, bound_f1


'''
resnet50- LL, LH, HL, HH, f1, f2, f3, f4-ODE-Hamiltonian Reversible
'''


class Network9(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channels=32, imagenet_pretrained=True):
        super(Network9, self).__init__()
        self.shared_encoder = timm.create_model(model_name="resnet50", pretrained=True, in_chans=3, features_only=True)
        self.GCM3 = GCM3(256, channels)
        self.GPM = GPM()
        self.REM12 = REM12(channels, channels)

        # self.LL_down = nn.Sequential(
        #     BasicConv2d(channels, channels,stride=2, kernel_size=3, padding=1),
        #     BasicConv2d(channels, channels,stride=2, kernel_size=3, padding=1)
        # )
        # self.dePixelShuffle = torch.nn.PixelShuffle(2)
        # self.one_conv_f4_ll = nn.Conv2d(in_channels=channels*2, out_channels=channels, kernel_size=1)
        # self.one_conv_f1_hh = nn.Conv2d(in_channels=channels+channels//4, out_channels=channels, kernel_size=1)

    def forward(self, x):
        image = x
        # Feature Extraction
        en_feats = self.shared_encoder(x)
        x0, x1, x2, x3, x4 = en_feats
        LL, LH, HL, HH, f1, f2, f3, f4 = self.GCM3(x1, x2, x3, x4)

        # HH_up = self.dePixelShuffle(HH)  # 64*48*48 ->16*96*96
        # f1_HH = torch.cat([HH_up, f1], dim=1)  # 16*96*96+64*96*96->80*96*96
        # f1_HH = self.one_conv_f1_hh(f1_HH) # 80*96*96 -> 64*96*96

        # LL_down =self.LL_down(LL)  # 64*48*48 ->64*12*12
        # f4_LL = torch.cat([LL_down, f4], dim=1) # 64*12*12+64*12*12->128*12*12
        # f4_LL = self.one_conv_f4_ll(f4_LL) # 128*12*12-> 64*12*12

        prior_cam = self.GPM(x4)  # bs, 1, 12, 12
        pred_0 = F.upsample(prior_cam, size=image.size()[2:], mode='bilinear', align_corners=False)
        f4, f3, f2, f1, bound_f4, bound_f3, bound_f2, bound_f1 = self.REM12([f1, f2, f3, f4], prior_cam, image)
        return pred_0, f4, f3, f2, f1, bound_f4, bound_f3, bound_f2, bound_f1


'''
noGCM
'''


class Network10(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channels=32, imagenet_pretrained=True):
        super(Network10, self).__init__()
        # ---- ResNet Backbone ----
        self.shared_encoder = timm.create_model(model_name="resnet50", pretrained=True, in_chans=3, features_only=True)
        self.GCM2 = GCM2(256, channels)
        self.GPM = GPM()
        self.REM9 = REM9(channels, channels)  # HH+HL+LH+LL 多尺度

    def forward(self, x):
        image = x
        # Feature Extraction
        en_feats = self.shared_encoder(x)
        x0, x1, x2, x3, x4 = en_feats
        f1, f2, f3, f4 = self.GCM2(x1, x2, x3, x4)
        prior_cam = self.GPM(x4)  # bs, 1, 12, 12
        pred_0 = F.upsample(prior_cam, size=image.size()[2:], mode='bilinear', align_corners=False)
        f4, f3, f2, f1, bound_f4, bound_f3, bound_f2, bound_f1 = self.REM9([f1, f2, f3, f4], prior_cam, image)
        return pred_0, f4, f3, f2, f1, bound_f4, bound_f3, bound_f2, bound_f1


'''
res2net- LL, LH, HL, HH, f1, f2, f3, f4-ODE
'''


class Network11(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channels=32, imagenet_pretrained=True):
        super(Network11, self).__init__()
        self.resnet = res2net50_v1b_26w_4s(pretrained=imagenet_pretrained)
        self.GCM3 = GCM3(256, channels)
        self.GPM = GPM()
        self.REM11 = REM11(channels, channels)

        # self.LL_down = nn.Sequential(
        #     BasicConv2d(channels, channels,stride=2, kernel_size=3, padding=1),
        #     BasicConv2d(channels, channels,stride=2, kernel_size=3, padding=1)
        # )
        self.dePixelShuffle = torch.nn.PixelShuffle(2)
        # self.one_conv_f4_ll = nn.Conv2d(in_channels=channels*2, out_channels=channels, kernel_size=1)
        self.one_conv_f1_hh = nn.Conv2d(in_channels=channels + channels // 4, out_channels=channels, kernel_size=1)

    def forward(self, x):
        image = x
        # Feature Extraction
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)  # bs, 64, 96 ,96
        x1 = self.resnet.layer1(x)  # bs, 256, 96 ,96
        x2 = self.resnet.layer2(x1)  # bs, 512, 48 ,48
        x3 = self.resnet.layer3(x2)  # bs, 1024, 24 ,24
        x4 = self.resnet.layer4(x3)  # bs, 2048, 12 ,12
        LL, LH, HL, HH, f1, f2, f3, f4 = self.GCM3(x1, x2, x3, x4)

        HH_up = self.dePixelShuffle(HH)  # 64*48*48 ->16*96*96
        f1_HH = torch.cat([HH_up, f1], dim=1)  # 16*96*96+64*96*96->80*96*96
        f1_HH = self.one_conv_f1_hh(f1_HH)  # 80*96*96 -> 64*96*96

        # LL_down =self.LL_down(LL)  # 64*48*48 ->64*12*12
        # f4_LL = torch.cat([LL_down, f4], dim=1) # 64*12*12+64*12*12->128*12*12
        # f4_LL = self.one_conv_f4_ll(f4_LL) # 128*12*12-> 64*12*12

        prior_cam = self.GPM(x4)  # bs, 1, 12, 12
        pred_0 = F.upsample(prior_cam, size=image.size()[2:], mode='bilinear', align_corners=False)
        f4, f3, f2, f1, bound_f4, bound_f3, bound_f2, bound_f1 = self.REM11([f1_HH, f2, f3, f4], prior_cam, image)
        return pred_0, f4, f3, f2, f1, bound_f4, bound_f3, bound_f2, bound_f1


'''
res2net- LL, LH, HL, HH, f1, f2, f3, f4-ODE-Hamiltonian Reversible
'''


class Network12(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channels=32, imagenet_pretrained=True):
        super(Network12, self).__init__()
        self.resnet = res2net50_v1b_26w_4s(pretrained=imagenet_pretrained)
        self.GCM3 = GCM3(256, channels)
        self.GPM = GPM()
        self.REM12 = REM12(channels, channels)

        # self.LL_down = nn.Sequential(
        #     BasicConv2d(channels, channels,stride=2, kernel_size=3, padding=1),
        #     BasicConv2d(channels, channels,stride=2, kernel_size=3, padding=1)
        # )
        self.dePixelShuffle = torch.nn.PixelShuffle(2)
        # self.one_conv_f4_ll = nn.Conv2d(in_channels=channels*2, out_channels=channels, kernel_size=1)
        self.one_conv_f1_hh = nn.Conv2d(in_channels=channels + channels // 4, out_channels=channels, kernel_size=1)

    def forward(self, x):
        image = x
        # Feature Extraction
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)  # bs, 64, 96 ,96
        x1 = self.resnet.layer1(x)  # bs, 256, 96 ,96
        x2 = self.resnet.layer2(x1)  # bs, 512, 48 ,48
        x3 = self.resnet.layer3(x2)  # bs, 1024, 24 ,24
        x4 = self.resnet.layer4(x3)  # bs, 2048, 12 ,12
        LL, LH, HL, HH, f1, f2, f3, f4 = self.GCM3(x1, x2, x3, x4)

        HH_up = self.dePixelShuffle(HH)  # 64*48*48 ->16*96*96
        f1_HH = torch.cat([HH_up, f1], dim=1)  # 16*96*96+64*96*96->80*96*96
        f1_HH = self.one_conv_f1_hh(f1_HH)  # 80*96*96 -> 64*96*96

        # LL_down =self.LL_down(LL)  # 64*48*48 ->64*12*12
        # f4_LL = torch.cat([LL_down, f4], dim=1) # 64*12*12+64*12*12->128*12*12
        # f4_LL = self.one_conv_f4_ll(f4_LL) # 128*12*12-> 64*12*12

        prior_cam = self.GPM(x4)  # bs, 1, 12, 12
        pred_0 = F.upsample(prior_cam, size=image.size()[2:], mode='bilinear', align_corners=False)
        f4, f3, f2, f1, bound_f4, bound_f3, bound_f2, bound_f1 = self.REM12([f1_HH, f2, f3, f4], prior_cam, image)
        return pred_0, f4, f3, f2, f1, bound_f4, bound_f3, bound_f2, bound_f1


if __name__ == '__main__':
    # image = torch.randn(1, 3, 384, 384).cuda()
    # x1= torch.randn(1, 256, 96, 96).cuda()
    # x2= torch.randn(1, 512, 48, 48).cuda()
    # x3= torch.randn(1, 1024, 24, 24).cuda()
    # x4= torch.randn(1, 2048, 12, 12).cuda()
    # gpm = GPM().cuda()
    # gcm = GCM3(256, 64).cuda()
    # shared_encoder = timm.create_model(model_name="resnet50", pretrained=True, in_chans=3, features_only=True).cuda()
    # model = Network9(channels=64).cuda()
    # model1 = Network8(channels=64).cuda()
    # model2 = Network5(channels=64).cuda()
    image = torch.rand(2, 3, 384, 384).cuda()
    model = Network10(96).cuda()
    pred_0, f4, f3, f2, f1, bound_f4, bound_f3, bound_f2, bound_f1 = model(image)
    print(pred_0.shape)
    print(f4.shape)
    print(f3.shape)
    print(f2.shape)
    print(f1.shape)
    print(bound_f4.shape)
    print(bound_f3.shape)
    print(bound_f2.shape)
    print(bound_f1.shape)
    # feature = torch.randn(1, 2048, 12, 12).cuda()
    # macs, params = profile(gpm, (feature,))
    # print('GPM                            ','flops: ', 2*macs, 'params: ', params)

    # macs, params = profile(gcm, (x1,x2,x3,x4))
    # print('GCM                            ','flops: ', 2*macs, 'params: ', params)

    # macs, params = profile(shared_encoder, (image,))
    # print('ResNet50                       ','flops: ', 2*macs, 'params: ', params)

    # macs, params = profile(model, (image,))
    # print('64channel f1hh ODE-Hamiltonian ','flops: ', 2*macs, 'params: ', params)

    # macs, params = profile(model1, (image,))
    # print('64channel f1hh ODE             ','flops: ', 2*macs, 'params: ', params)

    # macs, params = profile(model1, (image,))
    # print('64channel f1hh                 ','flops: ', 2*macs, 'params: ', params)

    # rem9 = REM9(64, 64).cuda()
    # rem11 = REM11(64, 64).cuda()
    # rem12 = REM12(64, 64).cuda()
    # f1= torch.randn(1, 64, 96, 96).cuda()
    # f2= torch.randn(1, 64, 48, 48).cuda()
    # f3= torch.randn(1, 64, 24, 24).cuda()
    # f4= torch.randn(1, 64, 12, 12).cuda()
    # prior_cam = torch.randn(1, 1, 12, 12).cuda()

    # macs, params = profile(rem9, ([f1,f2,f3,f4],prior_cam,image))
    # print('64channel rem ','flops: ', 2*macs, 'params: ', params)

    # macs, params = profile(rem11, ([f1,f2,f3,f4],prior_cam,image))
    # print('64channel rem ODE             ','flops: ', 2*macs, 'params: ', params)

    # macs, params = profile(rem12, ([f1,f2,f3,f4],prior_cam,image))
    # print('64channel rem ODE-Hamiltonian                ','flops: ', 2*macs, 'params: ', params)
