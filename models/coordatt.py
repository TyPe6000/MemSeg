# https://github.com/houqb/CoordAttention/blob/main/coordatt.py

import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        # self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        # self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h, x_w = None, None

        # ✅ 축 평균으로 명확히 표현 (항상 NCHW)
        if torch.onnx.is_in_onnx_export():
            # ONNX export 시엔 ReduceMean이 생성됨
            x_h = x.mean(dim=3, keepdim=True)        # (N,C,H,1)  ← W축 평균
            x_w = x.mean(dim=2, keepdim=True)        # (N,C,1,W)  ← H축 평균
            # 2) concat 없이 각 분기에 동일한 1x1 conv → BN → act 적용
            y_h = self.act(self.bn1(self.conv1(x_h)))  # (N,mip,H,1)
            y_w = self.act(self.bn1(self.conv1(x_w)))  # (N,mip,1,W)

            # 3) 축별 주의맵 산출
            a_h = torch.sigmoid(self.conv_h(y_h))      # (N,oup,H,1)
            a_w = torch.sigmoid(self.conv_w(y_w))      # (N,oup,1,W)

            out = x * a_h * a_w                        # 브로드캐스트 곱 (N,C,H,W)
            return out
        else:
            x_h = self.pool_h(x)
            x_w = self.pool_w(x).permute(0, 1, 3, 2)

            y = torch.cat([x_h, x_w], dim=2)
            y = self.conv1(y)
            y = self.bn1(y)
            y = self.act(y) 
            
            x_h, x_w = torch.split(y, [h, w], dim=2)
            x_w = x_w.permute(0, 1, 3, 2)

            a_h = self.conv_h(x_h).sigmoid()
            a_w = self.conv_w(x_w).sigmoid()

            out = identity * a_w * a_h

            return out

    # def forward(self, x):
    #     identity = x
        
    #     n,c,h,w = x.size()
    #     x_h = self.pool_h(x)
    #     x_w = self.pool_w(x).permute(0, 1, 3, 2)

    #     y = torch.cat([x_h, x_w], dim=2)
    #     y = self.conv1(y)
    #     y = self.bn1(y)
    #     y = self.act(y) 
        
    #     x_h, x_w = torch.split(y, [h, w], dim=2)
    #     x_w = x_w.permute(0, 1, 3, 2)

    #     a_h = self.conv_h(x_h).sigmoid()
    #     a_w = self.conv_w(x_w).sigmoid()

    #     out = identity * a_w * a_h

    #     return out