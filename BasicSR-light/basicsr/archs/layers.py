import torch
import torch.nn as nn
import torch.nn.functional as F

class CALayer(nn.Module):
    """
    Channel Attention Layer
    parameter: in_channel
    More detail refer to:
    """
    def __init__(self, channel, reduction=4):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid())

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class CALayerV2(nn.Module):
    """
    Channel Attention Layer
    parameter: in_channel
    More detail refer to:
    """
    def __init__(self, channel, reduction=4):
        super(CALayerV2, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid())

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x + x * y


class SHALayer(nn.Module):
    def __init__(self, in_channel, reduction=4):
        super(SHALayer, self).__init__()
        self.conv_ks1 = nn.Conv2d(in_channel, in_channel // reduction, 1, 1, 0)
        self.conv_ks3 = nn.Conv2d(in_channel // reduction, in_channel, 3, 1, 1)
        self.relu6 = nn.ReLU6()
        self.sigmoid = nn.Sigmoid()

    def forward(self, fea):
        b, c, h, w = fea.shape

        H_avg = torch.mean(fea, dim=3, keepdim=True)
        H_max, _ = torch.max(fea, dim=3, keepdim=True)
        H_enc = H_avg + H_max
        H_enc = H_enc.view(b, c, h, 1)

        V_avg = torch.mean(fea, dim=2, keepdim=True)
        V_max, _ = torch.max(fea, dim=2, keepdim=True)
        V_enc = V_avg + V_max
        V_enc = V_enc.view(b, c, w, 1)

        enc = torch.cat([H_enc, V_enc], dim=2)
        enc = self.conv_ks1(enc)
        enc = self.relu6(enc)

        reduce_c = enc.shape[1]

        H, V = torch.split(enc, [h, w], dim=2)
        H = H.view(b, reduce_c, h, 1)
        V = V.view(b, reduce_c, 1, w)
        
        H = self.conv_ks3(H)
        V = self.conv_ks3(V)

        attn_mask = H * V
        attn_mask = self.sigmoid(attn_mask)
        return attn_mask * fea


class SHALayerV2(nn.Module):
    def __init__(self, in_channel, reduction=4):
        super(SHALayerV2, self).__init__()
        self.conv_ks1 = nn.Conv2d(in_channel, in_channel // reduction, 1, 1, 0)
        self.conv_ks3 = nn.Conv2d(in_channel // reduction, in_channel, 3, 1, 1)
        self.relu6 = nn.ReLU6()
        self.sigmoid = nn.Sigmoid()

    def forward(self, fea):
        b, c, h, w = fea.shape

        H_avg = torch.mean(fea, dim=3, keepdim=True)
        H_max, _ = torch.max(fea, dim=3, keepdim=True)
        H_enc = H_avg + H_max
        H_enc = H_enc.view(b, c, h, 1)

        V_avg = torch.mean(fea, dim=2, keepdim=True)
        V_max, _ = torch.max(fea, dim=2, keepdim=True)
        V_enc = V_avg + V_max
        V_enc = V_enc.view(b, c, w, 1)

        enc = torch.cat([H_enc, V_enc], dim=2)
        enc = self.conv_ks1(enc)
        enc = self.relu6(enc)

        reduce_c = enc.shape[1]

        H, V = torch.split(enc, [h, w], dim=2)
        H = H.view(b, reduce_c, h, 1)
        V = V.view(b, reduce_c, 1, w)
        
        H = self.conv_ks3(H)
        V = self.conv_ks3(V)

        attn_mask = H * V
        attn_mask = self.sigmoid(attn_mask)
        return attn_mask * fea + fea