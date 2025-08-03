import torch
import torch.nn.functional as F
from .utils import *
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
from .kan import KANLinear, KAN

from torch import nn
from torch.autograd import Variable

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            # nn.ReLU6()
            nn.SiLU(inplace=True)
        )

class Conv(nn.Sequential):
            def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
                super().__init__(
                    nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                              dilation=dilation, stride=stride,
                              padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
                )


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )

class BaseConv(nn.Sequential):
    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize,
                 stride,
                 groups=1,
                 bias=False,
                 act="silu"):
        super(BaseConv, self).__init__()


        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=(ksize - 1) // 2,
            groups=groups,
            bias=bias)
        self.bn = nn.BatchNorm2d(
            out_channels)

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.conv.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        # use 'x * F.sigmoid(x)' replace 'silu'
        x = self.bn(self.conv(x))
        y = x * F.sigmoid(x)
        return y

def get_act_fn(act=None, trt=False):
    assert act is None or isinstance(act, (
        str, dict)), 'name of activation should be str, dict or None'
    if not act:
        return nn.Identity()

    if isinstance(act, dict):
        name = act['name']
        act.pop('name')
        kwargs = act
    else:
        name = act
        kwargs = dict()

    fn = getattr(F, name)

    return lambda x: fn(x, **kwargs)

class RepVggBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu', alpha=False):
        super(RepVggBlock, self).__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvBN(
            ch_in, ch_out, 3, stride=1)
        self.conv2 = ConvBN(
            ch_in, ch_out, 1, stride=1)
        self.act = get_act_fn(act) if act is None or isinstance(act, (
            str, dict)) else act
        if alpha:
            self.alpha = nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, x):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            if self.alpha:
                y = self.conv1(x) + self.alpha * self.conv2(x)
            else:
                y = self.conv1(x) + self.conv2(x)
        y = self.act(y)
        return y

    def convert_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2D(
                in_channels=self.ch_in,
                out_channels=self.ch_out,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=1)
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.set_value(kernel)
        self.conv.bias.set_value(bias)
        self.__delattr__('conv1')
        self.__delattr__('conv2')

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        if self.alpha:
            return kernel3x3 + self.alpha * self._pad_1x1_to_3x3_tensor(
                kernel1x1), bias3x3 + self.alpha * bias1x1
        else:
            return kernel3x3 + self._pad_1x1_to_3x3_tensor(
                kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.bn._mean
        running_var = branch.bn._variance
        gamma = branch.bn.weight
        beta = branch.bn.bias
        eps = branch.bn._epsilon
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape((-1, 1, 1, 1))
        return kernel * t, beta - running_mean * gamma / std

class CSPRepLayer(nn.Module):
    def __init__(self,
                     in_channels,
                     out_channels,
                     num_blocks=3,
                     expansion=1.0,
                     bias=False,
                     act="silu"):
            super(CSPRepLayer, self).__init__()
            # hidden_channels = int(out_channels * expansion)
            hidden_channels =out_channels

            self.conv1 = BaseConv(
                in_channels, hidden_channels, ksize=1, stride=1, bias=bias, act=act)
            self.conv2 = BaseConv(
                in_channels, hidden_channels, ksize=1, stride=1, bias=bias, act=act)
            self.bottlenecks = nn.Sequential(*[RepVggBlock(
                    hidden_channels, hidden_channels, act=act)
                for _ in range(num_blocks)
            ])
            if hidden_channels != out_channels:
                self.conv3 = BaseConv(
                    hidden_channels,
                    out_channels,
                    ksize=1,
                    stride=1,
                    bias=bias,
                    act=act)
            else:
                self.conv3 = nn.Identity()

    def forward(self, x):
            x_1 = self.conv1(x)
            x_1 = self.bottlenecks(x_1)
            x_2 = self.conv2(x)
            return self.conv3(x_1 + x_2)


class CBAM(nn.Module):
    def __init__(self, in_channels):
        super(CBAM, self).__init__()

        # 通道注意力模块
        self.channel_attention = ChannelAttention(in_channels)

        # 空间注意力模块，使用空洞卷积
        self.spatial_attention = SpatialAttention(in_channels)

    def forward(self, x):
        # 先通过通道注意力
        a = self.channel_attention(x)  # Apply channel attention

        b = self.spatial_attention(x)  # Apply spatial attention

        c = self.spatial_attention(a)

        sig = nn.Sigmoid()
        d = sig(a) * a + sig(b) * b + sig(c) * c
        return d



class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()

        # 全局平均池化和最大池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 1x1卷积来生成通道注意力权重
        self.fc1 = nn.Conv2d(in_channels, in_channels // 16, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // 16, in_channels, kernel_size=1)

        # 引入局部池化
        self.local_pool = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=2, dilation=2)

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))

        # 局部池化
        local_out = self.local_pool(x)

        # 加权求和
        out = avg_out + max_out + local_out
        return torch.sigmoid(out)* x


class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()

        # 空洞卷积用于计算空间注意力
        self.dilated_conv = nn.Conv2d(in_channels, 1, kernel_size=3, padding=2, dilation=2)

    def forward(self, x):
        # 使用空洞卷积进行空间注意力计算
        attention_map = torch.sigmoid(self.dilated_conv(x))
        return attention_map * x




class KANLayer(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., shift_size=5):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features

        grid_size = 5
        spline_order = 3
        scale_noise = 0.1
        scale_base = 1.0
        scale_spline = 1.0
        base_activation = torch.nn.SiLU
        grid_eps = 0.02
        grid_range = [-1, 1]

        self.fc1 = KANLinear(
            in_features,
            hidden_features,
            grid_size=grid_size,
            spline_order=spline_order,
            scale_noise=scale_noise,
            scale_base=scale_base,
            scale_spline=scale_spline,
            base_activation=base_activation,
            grid_eps=grid_eps,
            grid_range=grid_range,
        )
        self.fc2 = KANLinear(
            hidden_features,
            out_features,
            grid_size=grid_size,
            spline_order=spline_order,
            scale_noise=scale_noise,
            scale_base=scale_base,
            scale_spline=scale_spline,
            base_activation=base_activation,
            grid_eps=grid_eps,
            grid_range=grid_range,
        )
        self.fc3 = KANLinear(
            hidden_features,
            out_features,
            grid_size=grid_size,
            spline_order=spline_order,
            scale_noise=scale_noise,
            scale_base=scale_base,
            scale_spline=scale_spline,
            base_activation=base_activation,
            grid_eps=grid_eps,
            grid_range=grid_range,
        )

        self.dwconv_1 = DW_bn_relu(hidden_features)
        self.dwconv_2 = DW_bn_relu(hidden_features)
        self.dwconv_3 = DW_bn_relu(hidden_features)

        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)#截断正态分布，初始化线性层的权重，均值通常设为0，偏离0的程度标准差0.02
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)#直接修改传入的张量，而不是返回一个新的张量，constant后有下划线（_）
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):#Kaiming初始化
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        # pdb.set_trace()
        B, N, C = x.shape

        x = self.fc1(x.reshape(B * N, C))
        x = x.reshape(B, N, C).contiguous()
        x = self.dwconv_1(x, H, W)
        x = self.fc2(x.reshape(B * N, C))
        x = x.reshape(B, N, C).contiguous()
        x = self.dwconv_2(x, H, W)
        x = self.fc3(x.reshape(B * N, C))
        x = x.reshape(B, N, C).contiguous()
        x = self.dwconv_3(x, H, W)

        return x


class KANBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.layer = KANLayer(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.layer(self.norm2(x), H, W))

        return x




class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class DW_bn_relu(nn.Module):
    def __init__(self, dim=768):
        super(DW_bn_relu, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.bn = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU()

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class D_ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(D_ConvLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)



class kanCCFMabc(nn.Module):  #有注意力机制
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, img_size=224, patch_size=16, in_chans=3,
                 embed_dims=[256, 320, 512], num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False,
                 qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1], **kwargs):
        super().__init__()

        kan_input_dim = embed_dims[0]

        self.in_channels = input_channels
        self.encoder1 = ConvLayer(3, kan_input_dim // 8)
        self.encoder2 = ConvLayer(kan_input_dim // 8, kan_input_dim // 4)
        self.encoder3 = ConvLayer(kan_input_dim // 4, kan_input_dim)

        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])

        self.dnorm3 = norm_layer(embed_dims[1])
        self.dnorm4 = norm_layer(embed_dims[0])


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.block1 = nn.ModuleList([KANBlock(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])



        self.dblock1 = nn.ModuleList([KANBlock(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])])



        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])

        self.decoder1 = D_ConvLayer(embed_dims[2], embed_dims[1])
        self.DA1 = CBAM(32)
        self.DA2 = CBAM(64)
        self.DA3 = CBAM(256)
        self.DA4 = CBAM(320)


        self.segmentation_head = nn.Sequential(ConvBNReLU(32, embed_dims[0] // 8),
                                               nn.Dropout2d(p=0.1, inplace=True),
                                               Conv(embed_dims[0] // 8, num_classes, kernel_size=1))

        self.final = nn.Conv2d(embed_dims[0] // 8, num_classes, kernel_size=1)
        self.soft = nn.Softmax(dim=1)
        self.encoder_channels = [32, 64, 256, 320, 512]

        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        # for idx in range(self.in_channels-1, 0, -1):
        for idx in range(self.in_channels, 0, -1): #3，2，1同此处
            # 添加 lateral convolution
            self.lateral_convs.append(
                BaseConv(
                    # self.encoder_channels[idx], self.encoder_channels[idx], 1, 1)
                    self.encoder_channels[idx], self.encoder_channels[idx - 1], 1, 1)
            )
            self.fpn_blocks.append(
                CSPRepLayer(
                    self.encoder_channels[idx - 1],  # 假设这是根据上下文确定的输出通道数
                    self.encoder_channels[idx - 1],
                    # self.encoder_channels[idx] + self.encoder_channels[idx-1],  # 假设这是根据上下文确定的输出通道数
                    # self.encoder_channels[idx-1],
                )
            )


    def forward(self, x):
        B, _, h, w = x.shape
        # print(x.shape)
        out = F.relu(F.max_pool2d(self.encoder1(x), 2, 2))
        t1 = self.DA1(out)
        # t1 = out
        # print(t1.shape)
        out = F.relu(F.max_pool2d(self.encoder2(out), 2, 2))
        t2 = self.DA2(out)
        # t2 = out
        # print(t2.shape)
        out = F.relu(F.max_pool2d(self.encoder3(out), 2, 2))
        t3 = self.DA3(out)
        # t3 =out
        # print(t3.shape)
        out, H, W = self.patch_embed3(out)

        for i, blk in enumerate(self.block1):
            out = blk(out, H, W)
        out = self.norm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = self.DA4(out)
        # t4 = out

        res = [t1, t2, t3, t4]
        inner_outs = [res[-1]]  # inner_outs=[t5]
        for idx in range(self.in_channels, 0, -1):

            feat_heigh = inner_outs[0]  # 将inner_outs列表中的第一个元素t5,t5融合t4,（即当前处理的最精细尺度的特征图）赋值给feat_heigh
            feat_low = res[idx - 1]  # 当前要处理的较低分辨率的特征图赋值给feat_low=t4,t3

            # feat_heigh = self.lateral_convs[self.in_channels+1 - idx](feat_heigh) #0，1，2，3
            a = self.lateral_convs[self.in_channels - idx](feat_heigh)
            inner_outs[0] = a  #覆盖原来的0索引值
            a = F.interpolate(a, scale_factor=2., mode="nearest")
            sig = nn.Sigmoid()
            b = a + feat_low
            c = a - feat_low
            d = sig(c) * b + sig(b)
            inner_out = self.fpn_blocks[self.in_channels - idx](d)

            inner_outs.insert(0, inner_out)  #inner_outs=[t5融t4融t3融t2融t1，t5融t4融t3融t2，t5融t4融t3，t5融t4,t5]插入到列表的开头，原列表中的其他元素会依次向后移动一位


        out = F.interpolate(inner_outs[0], scale_factor=(2, 2), mode='bilinear')
        out = self.segmentation_head(out)  # 改变类别[8, 5, 512, 512]
        # print(out.shape)
        return {"out": out}


