
from lib.pvtv2 import pvt_v2_b2
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class conv2d(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, act=True):
        super().__init__()
        self.act = act

        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.act == True:
            x = self.relu(x)
        return x

# Channel Attention Block
class CAB(nn.Module):
    def __init__(self, in_planes, ratio=16):  
        super(CAB, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1) 
        self.max_pool = nn.AdaptiveMaxPool2d(1)  

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)  
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False) 
        self.sigmoid = nn.Sigmoid() 

    def forward(self, x):  
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x)))) 
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x)))) 

        out = avg_out + max_out  
        return self.sigmoid(out)  
# Spatial Attention Block
class SAB(nn.Module):
    def __init__(self, kernel_size=7):  
        super(SAB, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)  
        self.sigmoid = nn.Sigmoid()  

    def forward(self, x):  
        avg_out = torch.mean(x, dim=1, keepdim=True)  
        max_out, _ = torch.max(x, dim=1, keepdim=True)  

        x = torch.cat([avg_out, max_out], dim=1) 
        x = self.conv1(x)  
        return self.sigmoid(x) 

# ============================
#   FEPM: Feature Enhancement and Propagation Module
# ============================
class FEPM(nn.Module):
    def __init__(self, in_channels, ratio=16): 
        super().__init__()
        self.in_channels = in_channels 
       
        self.dwconvs = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, 3, 1, 3 // 2, groups=self.in_channels, bias=False),
            nn.BatchNorm2d(self.in_channels)
            nn.ReLU6(inplace=True))  

      
        self.conv1X1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.dwconv3X3 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, 3, 1, 3 // 2, groups=self.in_channels, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU6(inplace=True))
        self.dwconv5X5 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, 5, 1, 5 // 2, groups=self.in_channels, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU6(inplace=True))
        self.dwconv7X7 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, 7, 1, 7 // 2, groups=self.in_channels, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU6(inplace=True))
        self.dwconv9X9 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, 9, 1, 9 // 2, groups=self.in_channels, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU6(inplace=True))

        self.avg_pool = nn.AdaptiveAvgPool2d(1)  
        self.max_pool = nn.AdaptiveMaxPool2d(1)  
        self.fc1 = nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False)  
        self.relu1 = nn.ReLU()  
        self.fc2 = nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)  
        self.sigmoid = nn.Sigmoid()  

      
        self.conv1 = nn.Conv2d(2, 1, 7, padding=7 // 2, bias=False)  
        self.sigmoid = nn.Sigmoid()  

    def forward(self, x):  
        x = self.dwconv3X3(x)  
        x = x + self.dwconv5X5(x) + self.dwconv7X7(x) + self.dwconv9X9(x) 
        x = self.conv1X1(x) 

        avg_out1 = self.fc2(self.relu1(self.fc1(self.avg_pool(x)))) 
        max_out1 = self.fc2(self.relu1(self.fc1(self.max_pool(x))))  
        out1 = avg_out1 + max_out1 
        c_attention = self.sigmoid(out1)  

       
        avg_out2 = torch.mean(x, dim=1, keepdim=True)  
        max_out2, _ = torch.max(x, dim=1, keepdim=True) 
        out2 = torch.cat([avg_out2, max_out2], dim=1)  
        out2 = self.conv1(out2)  
        s_attention = self.sigmoid(out2)  

      
        output = x * c_attention + x * s_attention
        return output  

class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.dw(x)
        x = x.reshape(B, C, H*W).transpose(1, 2)
        return x


class GCLM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        hidden = dim * 2
        self.fc1 = nn.Linear(dim, hidden * 2)
        self.dw = DWConv(hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.act = nn.GELU()

    def forward(self, x):
        B, C, H, W = x.size()
        x = x.view(B, C, H*W).transpose(1, 2)  # B (H*W) C
        x1, x2 = self.fc1(x).chunk(2, dim=-1)
        x = self.act(self.dw(x1, H, W)) * self.dw(x2, H, W)
        x = self.fc2(x)
        return x.transpose(1, 2).reshape(B, C, H, W)


# -------------------------------
# Cross Attention Supporting Q/KV with Different Sizes & Channels
# -------------------------------
class ConvCrossAttention(nn.Module):
    def __init__(self, q_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads

        # q → q
        self.q_proj = nn.Conv2d(q_dim, q_dim, 1)
        self.q_dw = nn.Conv2d(q_dim, q_dim, 3, 1, 1, groups=q_dim)

        # kv → k, v (mapped to q_dim)
        self.kv_proj = nn.Conv2d(q_dim, q_dim*2, 1)
        self.kv_dw = nn.Conv2d(q_dim*2, q_dim*2, 3, 1, 1, groups=q_dim*2)

        self.out = nn.Conv2d(q_dim, q_dim, 1)
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

    def forward(self, q, kv):
        B, C, H, W = q.size()

        # Q branch
        q = self.q_dw(self.q_proj(q))

        # KV branch → produce K, V
        kv = self.kv_dw(self.kv_proj(kv))
        k, v = kv.chunk(2, dim=1)

        # reshape to multi-head
        q = rearrange(q, "b (h d) h1 w1 -> b h d (h1 w1)", h=self.num_heads)
        k = rearrange(k, "b (h d) h1 w1 -> b h d (h1 w1)", h=self.num_heads)
        v = rearrange(v, "b (h d) h1 w1 -> b h d (h1 w1)", h=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = rearrange(out, "b h d (h1 w1) -> b (h d) h1 w1",
                        h=self.num_heads, h1=H, w1=W)

        return self.out(out)


# ============================
#  DSSM：Dual-Stream Semantic Mixture 
# ============================
class DFSM(nn.Module):
    """
    Q:  (B, Cq, Hq, Wq)
    KV: (B, Ckv, Hkv, Wkv)

   (B, Cq, Hq, Wq)
    """
    def __init__(self, q_channels, kv_channels, num_heads=4):
        super().__init__()

      
        self.kv_align = nn.Conv2d(kv_channels, q_channels, 1)

        self.norm_q = nn.LayerNorm(q_channels)
        self.norm_kv = nn.LayerNorm(q_channels)

        self.att = ConvCrossAttention(q_channels, num_heads)
        self.gclm = GCLM(q_channels)

    def forward(self, q, kv):
        B, Cq, Hq, Wq = q.shape

        # STEP 1: resize KV → Q 
        kv = F.interpolate(kv, size=(Hq, Wq),
                           mode="bilinear", align_corners=False)

        # STEP 2: KV →Cq
        kv = self.kv_align(kv)

        # STEP 3: LayerNorm
        qn = self.norm_q(q.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        kvn = self.norm_kv(kv.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # STEP 4: Cross Attention (Q ← KV)
        att = self.att(qn, kvn)

        # STEP 5: GCLM
        gclm_out = self.gclm(qn)

        # STEP 6: output
        return q + att + gclm_out

# ============================
#   HMAP：Hierarchical Multi-scale Aggregation and Prediction
# ============================
class HMAP(nn.Module):
    def __init__(self, in_c, out_c):
        super(HMAP, self).__init__()
     
        self.c3 = conv2d(in_c[0], out_c, kernel_size=1, padding=0)
        self.c2 = conv2d(in_c[1], out_c, kernel_size=1, padding=0)
        self.c1 = conv2d(in_c[2], out_c, kernel_size=1, padding=0)

  
        self.branch1_1 = conv2d(out_c * 2, out_c)
        self.branch1_2 = conv2d(out_c, out_c, kernel_size=1, padding=0)

        self.branch2_1 = conv2d(out_c * 2, out_c)
        self.branch2_2 = conv2d(out_c, out_c, kernel_size=1, padding=0)

       
        self.fuse1 = conv2d(out_c * 2, out_c)
        self.fuse2 = conv2d(out_c, out_c)

        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.dc1 = nn.Conv2d(64, 64, 1)
        self.dc2 = nn.Conv2d(128, 64, 1)
        self.dc3 = nn.Conv2d(320, 64, 1)
        self.dc4 = nn.Conv2d(512, 64, 1)
        self.bn_dc1 = nn.BatchNorm2d(64)
        self.bn_dc2 = nn.BatchNorm2d(64)
        self.bn_dc3 = nn.BatchNorm2d(64)
        self.bn_dc4 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 4)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

        self.out1 = nn.Conv2d(64, 1, 1)
        self.out2 = nn.Conv2d(128, 1, 1)
        self.out3 = nn.Conv2d(320, 1, 1)
        self.out4 = nn.Conv2d(512, 1, 1)

        self.out5 = nn.Conv2d(64, 1, 1)

    def forward(self, x1, x2, x3, x4):

        target_h = x1.size(2) * 2
        target_w = x1.size(3) * 2

       
        x3_up = self.c3(x3)
        x3_up = F.interpolate(x3_up, size=(target_h, target_w), mode="bilinear", align_corners=True)

        x2_up = self.c2(x2)
        x2_up = F.interpolate(x2_up, size=(target_h, target_w), mode="bilinear", align_corners=True)

       
        x12 = torch.cat([x3_up, x2_up], dim=1)


        b1 = self.branch1_2(self.branch1_1(x12))
        b2 = self.branch2_2(self.branch2_1(x12))

      
        x1_up = self.c1(x1)
        x1_up = F.interpolate(x1_up, size=(target_h, target_w), mode="bilinear", align_corners=True)

    
        feat = x1_up * b1 + b2

       
        out = torch.cat([feat, x1_up], dim=1)
        out = self.fuse2(self.fuse1(out))
        out = F.interpolate(self.out5(out), scale_factor=2, mode='bilinear')


        B = x1.shape[0]
        y1 = self.pooling(self.bn_dc1(self.dc1(x1)))
        y2 = self.pooling(self.bn_dc2(self.dc2(x2)))
        y3 = self.pooling(self.bn_dc3(self.dc3(x3)))
        y4 = self.pooling(self.bn_dc4(self.dc4(x4)))
        y = y1 + y2 + y3 + y4
        coeff = self.sigmoid(self.fc2(self.relu(self.fc1(y.reshape(B, -1)))))
        prediction1 = self.out1(x1) * coeff[:, 0].reshape(B, 1, 1, 1)
        prediction2 = self.out2(x2) * coeff[:, 1].reshape(B, 1, 1, 1)
        prediction3 = self.out3(x3) * coeff[:, 2].reshape(B, 1, 1, 1)
        prediction4 = self.out4(x4) * coeff[:, 3].reshape(B, 1, 1, 1)

        prediction1 = F.interpolate(prediction1, scale_factor=4, mode='bilinear')
        prediction2 = F.interpolate(prediction2, scale_factor=8, mode='bilinear')
        prediction3 = F.interpolate(prediction3, scale_factor=16, mode='bilinear')
        prediction4 = F.interpolate(prediction4, scale_factor=32, mode='bilinear')

        return out, prediction1, prediction2, prediction3, prediction4


class OURSNet(nn.Module):
    def __init__(self, channel=32):
        super(OURSNet, self).__init__()

        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = './pretrained_pth/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        self.fepm1 = FEPM(64)
        self.fepm2 = FEPM(128)
        self.fepm3 = FEPM(320)
        self.Translayer4 = BasicConv2d(512, 512, 1)

        self.dfsm34 = DFSM(320,512)
        self.dfsm23 = DFSM(128,320)
        self.dfsm12 = DFSM(64,128)

        self.hmap = HMAP([320, 128, 64], 64)

    def forward(self, x):
        # backbone
        pvt = self.backbone(x)
        x1 = pvt[0]
        x2 = pvt[1]
        x3 = pvt[2]
        x4 = pvt[3]

        #FEPM
        x1_fepm = self.fepm1(x1)
        x2_fepm = self.fepm2(x2)
        x3_fepm = self.fepm3(x3)
        x4_Tran = self.Translayer4(x4)

        #dssm
        x34_g = self.dfsm34(x3_fepm,x4_Tran)
        x23_g = self.dfsm23(x2_fepm,x34_g)
        x12_g = self.dfsm12(x1_fepm,x23_g)

        #hmap
        out, p1, p2, p3, p4 = self.hmap(x12_g, x23_g, x34_g, x4_Tran)
        return out, p1, p2, p3, p4


if __name__ == '__main__':
    model = OURSNet().cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()

    x_out, prediction1,prediction2 ,prediction3,prediction4 = model(input_tensor)
    print(x_out.size(),prediction1.size(), prediction2.size(), prediction3.size(),prediction4.size())