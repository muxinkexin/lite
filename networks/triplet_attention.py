import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.cuda


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)
    def forward(self, x):
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

    
class XCA(nn.Module):
    def __init__(self, dim=128, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'temperature'}

class XCA2(nn.Module):
    def __init__(self, dim=128, layer_scale_init_value=1e-6,num_heads=8, qkv_bias=True, attn_drop=0., drop=0.):
        super().__init__()

        self.dim = dim
        self.norm_xca = LayerNorm(self.dim, eps=1e-6)
        self.gamma_xca = nn.Parameter(layer_scale_init_value * torch.ones(self.dim),
                                      requires_grad=True) if layer_scale_init_value > 0 else None
        self.xca = XCA(self.dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

    def forward(self, x):

        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).permute(0, 2, 1)
        x = x + self.gamma_xca * self.xca(self.norm_xca(x))
        x = x.reshape(B, H, W, C)
        x = x.permute(0, 3, 1, 2) 
        return x

    
class GParam(nn.Module):
    def __init__(self):
        super(GParam, self).__init__()
        self.weight = nn.Parameter(torch.ones(3),requires_grad=True)
    def forward(self, x):
        weight = F.softmax(self.weight, 0)  
        s = 0
        for i in range(len(weight)):
            s += x[i] * weight[i]
        return s  

class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        self.compress = ZPool()
        self.conv1 = torch.nn.Sequential(
                nn.ReflectionPad2d(3),
                torch.nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=0,bias=True)
                )
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv1(x_compress)
        scale = torch.sigmoid_(x_out) 
        return scale

class TripletAttention(nn.Module):
    def __init__(self, no_spatial=False):
        super(TripletAttention, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.no_spatial=no_spatial
        if not no_spatial:
            self.hw = AttentionGate()
    def forward(self, x):
        x_perm1 = x.permute(0,2,1,3).contiguous()
        hGate = self.cw(x_perm1).permute(0,2,1,3)
        x_perm2 = x.permute(0,3,2,1).contiguous()
        wGate = self.hc(x_perm2).permute(0,3,2,1)
        if not self.no_spatial:
            cGate = self.hw(x)
        return [hGate, wGate, cGate]

class TripletAttention2(nn.Module):
    def __init__(self):
        super(TripletAttention2, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.hw = AttentionGate()
        self.ma = GParam()
    def forward(self, x):
        x_perm1 = x.permute(0,2,1,3).contiguous()
        hGate = self.cw(x_perm1).permute(0,2,1,3)
        x_perm2 = x.permute(0,3,2,1).contiguous()
        wGate = self.hc(x_perm2).permute(0,3,2,1)
        cGate = self.hw(x)
        x = self.ma([hGate*x,wGate*x,cGate*x])
        return x

class attentionUpdate1(nn.Module):
    def __init__(self):
        super(attentionUpdate1, self).__init__()
        inD = 128
        self.attention1 = TripletAttention()
        self.attention2 = TripletAttention()
        self.conv31 = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=2*inD, out_channels=inD, kernel_size=1, stride=1, padding=0,bias=True),
                torch.nn.Tanh())
        self.conv32 = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=2*inD, out_channels=inD, kernel_size=1, stride=1, padding=0,bias=True),
                torch.nn.Tanh())
        self.convq =  torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=2*inD, out_channels=inD, kernel_size=1, stride=1, padding=0,bias=True),
                torch.nn.Tanh())
        self.rh = GParam()
        self.zh = GParam()
        self.zx = GParam()
    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        hx1 = self.conv31(hx)
        hx2 = self.conv32(hx)
        z = self.attention1(hx1)
        r = self.attention2(hx2)
        rh = self.rh([(r[0]*h) , (r[1]*h) ,r[2]*h])
        q = self.convq(torch.cat([rh, x], dim=1))
        
        zh =  self.zh([((1-z[0])*h),((1-z[1])*h),(1-z[2])*h])
        zx =  self.zx([(z[0]*q), (z[1]*q), z[2]*q])
        return zh + zx  
    
class attentionUpdate2(nn.Module):
    def __init__(self):
        super(attentionUpdate2, self).__init__()
        inD = 128
        self.attention1 = TripletAttention()
        self.attention2 = XCA2()
        self.conv31 = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=2*inD, out_channels=inD, kernel_size=1, stride=1, padding=0,bias=True),
                torch.nn.Tanh())
        self.conv32 = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=2*inD, out_channels=inD, kernel_size=1, stride=1, padding=0,bias=True),
                torch.nn.Tanh())
        self.convq =  torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=2*inD, out_channels=inD, kernel_size=1, stride=1, padding=0,bias=True),
                torch.nn.Tanh())
        self.zh = GParam()
        self.zx = GParam()
    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        hx1 = self.conv31(hx)
        hx2 = self.conv32(hx)
        z = self.attention1(hx1)
        r = self.attention2(hx2)
        q = self.convq(torch.cat([r * h, x], dim=1))
        
        zh =  self.zh([((1-z[0])*h),((1-z[1])*h),(1-z[2])*h])
        zx =  self.zx([(z[0]*q), (z[1]*q), z[2]*q])
        return zh + zx  

class SepConvGRU(nn.Module):
    def __init__(self):
        super(SepConvGRU, self).__init__()
        hidden_dim = 128
        catt = 256

        self.convz1 = nn.Conv2d(catt, hidden_dim, (1, 3), padding=(0, 1))
        self.convr1 = nn.Conv2d(catt, hidden_dim, (1, 3), padding=(0, 1))
        self.convq1 = nn.Conv2d(catt, hidden_dim, (1, 3), padding=(0, 1))

        self.convz2 = nn.Conv2d(catt, hidden_dim, (3, 1), padding=(1, 0))
        self.convr2 = nn.Conv2d(catt, hidden_dim, (3, 1), padding=(1, 0))
        self.convq2 = nn.Conv2d(catt, hidden_dim, (3, 1), padding=(1, 0))
    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        return h