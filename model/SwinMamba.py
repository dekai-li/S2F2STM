import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from mamba_ssm import Mamba

class MultiScaleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MultiScaleConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3)
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.final_conv = nn.Conv2d(out_channels * 4, out_channels, kernel_size=1)
        self.norm = nn.GroupNorm(1, out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)
        out4 = self.conv4(x)
        out = torch.cat((out1, out2, out3, out4), dim=1)
        out = self.final_conv(out)
        out = self.norm(out)
        out = self.activation(out)
        return out

class SwinTransformerBlock(nn.Module):
    def __init__(self, in_channels, embed_dim=96, num_heads=5, window_size=7, shift_size=0):
        super(SwinTransformerBlock, self).__init__()

        # Patch Embedding
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=4, stride=4, padding=0)

        # Multihead Attention
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

        # Feed Forward Networks (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape

        # Patch embedding
        x = self.patch_embed(x)  # Shape: (B, embed_dim, H', W')
        print(f"After patch_embed: {x.shape}")  # Debug print

        # Assuming the output of patch_embed has shape (B, embed_dim, H', W')
        _, embed_dim, H_, W_ = x.shape

        # Flatten spatial dimensions and permute to (H' * W', B, embed_dim)
        x = x.flatten(2).permute(2, 0, 1)  # Shape: (H' * W', B, embed_dim)

        # Apply MultiheadAttention (self-attention)
        x, _ = self.attn(x, x, x)

        x = x.permute(1, 2, 0).reshape(B, embed_dim, H_, W_)  # Reshape back to (B, embed_dim, H', W')

        # Apply LayerNorm and Feed-Forward Network
        x = x.flatten(2).transpose(1, 2)  # Flatten spatial dimensions and permute to (B, H' * W', embed_dim)
        x = self.ffn(self.norm1(x))  # Apply LayerNorm and Feed-Forward Network

        # Reshape back to (B, embed_dim, H', W')
        x = x.transpose(1, 2).reshape(B, embed_dim, H_, W_)

        return x


class ImprovedSpeMamba(nn.Module):
    def __init__(self, channels, token_num=8, use_residual=True, group_num=4):
        super(ImprovedSpeMamba, self).__init__()
        self.token_num = token_num
        self.use_residual = use_residual

        self.group_channel_num = math.ceil(channels / token_num)
        self.channel_num = self.token_num * self.group_channel_num

        self.mamba = Mamba(
            d_model=self.group_channel_num,
            d_state=16,
            d_conv=4,
            expand=2,
        )

        self.multi_scale_conv = MultiScaleConv(channels, self.channel_num)

        self.proj = nn.Sequential(
            nn.GroupNorm(group_num, self.channel_num),
            nn.SiLU()
        )

    def padding_feature(self, x):
        B, C, H, W = x.shape
        if C < self.channel_num:
            pad_c = self.channel_num - C
            pad_features = torch.zeros((B, pad_c, H, W)).to(x.device)
            cat_features = torch.cat([x, pad_features], dim=1)
            return cat_features
        else:
            return x

    def forward(self, x):
        x_pad = self.padding_feature(x)
        x_pad = self.multi_scale_conv(x_pad)  # Apply multi-scale convolution
        B, C, H, W = x_pad.shape
        x_pad = x_pad.view(B * H * W, self.token_num, self.group_channel_num)
        x_recon = self.mamba(x_pad)
        x_recon = x_recon.view(B, C, H, W)  # reshape to (batch_size, channels, height, width)
        x_proj = self.proj(x_recon)
        return x + x_proj if self.use_residual else x_proj


class SwinMambaHSI(nn.Module):
    def __init__(self, in_channels=128, hidden_dim=128, num_classes=10, token_num=4, group_num=4, use_residual=True, mamba_type='both'):
        super(SwinMambaHSI, self).__init__()
        self.token_num = token_num
        self.use_residual = use_residual

        # Patch Embedding Layer
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(group_num, hidden_dim),
            nn.SiLU()
        )

        #   Block
        self.swin_transformer = SwinTransformerBlock(hidden_dim)

        # 选择Mamba模块
        if mamba_type == 'spa':
            self.mamba = ImprovedSpeMamba(hidden_dim, token_num=token_num, use_residual=use_residual, group_num=group_num)
        elif mamba_type == 'spe':
            self.mamba = ImprovedSpeMamba(hidden_dim, token_num=token_num, use_residual=use_residual, group_num=group_num)
        elif mamba_type == 'both':
            self.mamba = ImprovedSpeMamba(hidden_dim, token_num=token_num, use_residual=use_residual, group_num=group_num)

        # 使用 1x1 卷积调整 swin_features 的通道数为 128
        self.swin_conv = nn.Conv2d(96, 128, kernel_size=1, stride=1)

        # 分类头
        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(group_num, 128),
            nn.SiLU(),
            nn.Conv2d(in_channels=128, out_channels=num_classes, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        # 通过Patch Embedding进行初步特征映射
        print(x.shape)
        x = self.patch_embedding(x)

        # 通过Swin Transformer提取空间特征
        swin_features = self.swin_transformer(x)

        # 通过Mamba模块提取光谱特征
        mamba_features = self.mamba(x)

        # 调整 swin_features 的通道数，使其与 mamba_features 的通道数匹配
        swin_features = self.swin_conv(swin_features)

        # 使用上采样确保两个特征图尺寸匹配
        mamba_features = F.interpolate(mamba_features, size=(swin_features.shape[2], swin_features.shape[3]), mode='bilinear', align_corners=False)

        # 融合Swin Transformer和Mamba的特征
        fusion_features = swin_features + mamba_features


        # 最终分类
        logits = self.cls_head(fusion_features)
        print('logits: ', logits.shape)

        return logits

class SwinMambaHSI_NoSwinTransformer(nn.Module):
    def __init__(self, in_channels=128, hidden_dim=64, num_classes=10, token_num=4, group_num=4, use_residual=True, mamba_type='both'):
        super(SwinMambaHSI_NoSwinTransformer, self).__init__()
        self.token_num = token_num
        self.use_residual = use_residual

        # Patch Embedding Layer
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(group_num, hidden_dim),
            nn.SiLU()
        )

        # 移除 Swin Transformer Block
        # self.swin_transformer = SwinTransformerBlock(hidden_dim)

        # 仅使用 Mamba 模块提取光谱特征
        if mamba_type == 'spa':
            self.mamba = ImprovedSpeMamba(hidden_dim, token_num=token_num, use_residual=use_residual, group_num=group_num)
        elif mamba_type == 'spe':
            self.mamba = ImprovedSpeMamba(hidden_dim, token_num=token_num, use_residual=use_residual, group_num=group_num)
        elif mamba_type == 'both':
            self.mamba = ImprovedSpeMamba(hidden_dim, token_num=token_num, use_residual=use_residual, group_num=group_num)

        # 使用 1x1 卷积调整 swin_features 的通道数为 128（此处已移除 swin_features，因此可省略）
        # self.swin_conv = nn.Conv2d(96, 128, kernel_size=1, stride=1)

        # 分类头
        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dim, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(group_num, 128),
            nn.SiLU(),
            nn.Conv2d(in_channels=128, out_channels=num_classes, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        # 通过Patch Embedding进行初步特征映射
        print(x.shape)
        x = self.patch_embedding(x)

        # 移除 Swin Transformer，直接通过 Mamba 模块提取特征
        mamba_features = self.mamba(x)

        # 分类头
        logits = self.cls_head(mamba_features)
        print('logits: ', logits.shape)

        return logits

class SwinMambaHSI_NoMamba(nn.Module):
    def __init__(self, in_channels=128, hidden_dim=64, num_classes=10, token_num=4, group_num=4, use_residual=True, mamba_type='both'):
        super(SwinMambaHSI_NoMamba, self).__init__()
        self.token_num = token_num
        self.use_residual = use_residual

        # Patch Embedding Layer
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(group_num, hidden_dim),
            nn.SiLU()
        )

        # Swin Transformer Block
        self.swin_transformer = SwinTransformerBlock(hidden_dim)

        # 移除 Mamba 模块
        # if mamba_type == 'spa':
        #     self.mamba = ImprovedSpeMamba(hidden_dim, token_num=token_num, use_residual=use_residual, group_num=group_num)
        # elif mamba_type == 'spe':
        #     self.mamba = ImprovedSpeMamba(hidden_dim, token_num=token_num, use_residual=use_residual, group_num=group_num)
        # elif mamba_type == 'both':
        #     self.mamba = ImprovedSpeMamba(hidden_dim, token_num=token_num, use_residual=use_residual, group_num=group_num)

        # 使用 1x1 卷积调整 swin_features 的通道数为 128
        self.swin_conv = nn.Conv2d(96, 128, kernel_size=1, stride=1)

        # 分类头
        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(group_num, 128),
            nn.SiLU(),
            nn.Conv2d(in_channels=128, out_channels=num_classes, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        # 通过Patch Embedding进行初步特征映射
        print(x.shape)
        x = self.patch_embedding(x)

        # 通过Swin Transformer提取空间特征
        swin_features = self.swin_transformer(x)

        # 移除 Mamba 模块，直接使用 swin_features
        # mamba_features = self.mamba(x)

        # 调整 swin_features 的通道数，使其与 mamba_features 的通道数匹配
        swin_features = self.swin_conv(swin_features)

        # 直接使用 swin_features 进行分类
        fusion_features = swin_features

        # 最终分类
        logits = self.cls_head(fusion_features)
        print('logits: ', logits.shape)

        return logits

class ImprovedSpeMamba_NoMultiScaleConv(nn.Module):
    def __init__(self, channels, token_num=8, use_residual=True, group_num=4):
        super(ImprovedSpeMamba_NoMultiScaleConv, self).__init__()
        self.token_num = token_num
        self.use_residual = use_residual

        self.group_channel_num = math.ceil(channels / token_num)
        self.channel_num = self.token_num * self.group_channel_num

        self.mamba = Mamba(
            d_model=self.group_channel_num,
            d_state=16,
            d_conv=4,
            expand=2,
        )

        # 使用单一卷积替代多尺度卷积
        self.single_scale_conv = nn.Conv2d(channels, self.channel_num, kernel_size=3, padding=1)

        self.proj = nn.Sequential(
            nn.GroupNorm(group_num, self.channel_num),
            nn.SiLU()
        )

    def padding_feature(self, x):
        B, C, H, W = x.shape
        if C < self.channel_num:
            pad_c = self.channel_num - C
            pad_features = torch.zeros((B, pad_c, H, W)).to(x.device)
            cat_features = torch.cat([x, pad_features], dim=1)
            return cat_features
        else:
            return x

    def forward(self, x):
        x_pad = self.padding_feature(x)
        x_pad = self.single_scale_conv(x_pad)  # 使用单一卷积
        B, C, H, W = x_pad.shape
        x_pad = x_pad.view(B * H * W, self.token_num, self.group_channel_num)
        x_recon = self.mamba(x_pad)
        x_recon = x_recon.view(B, C, H, W)  # reshape to (batch_size, channels, height, width)
        x_proj = self.proj(x_recon)
        return x + x_proj if self.use_residual else x_proj

class SwinMambaHSI_NoMultiScaleConv(nn.Module):
    def __init__(self, in_channels=128, hidden_dim=64, num_classes=10, token_num=4, group_num=4, use_residual=True, mamba_type='both'):
        super(SwinMambaHSI_NoMultiScaleConv, self).__init__()
        self.token_num = token_num
        self.use_residual = use_residual

        # Patch Embedding Layer
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(group_num, hidden_dim),
            nn.SiLU()
        )

        # Swin Transformer Block
        self.swin_transformer = SwinTransformerBlock(hidden_dim)

        # 选择Mamba模块
        if mamba_type == 'spa':
            self.mamba = ImprovedSpeMamba_NoMultiScaleConv(hidden_dim, token_num=token_num, use_residual=use_residual, group_num=group_num)
        elif mamba_type == 'spe':
            self.mamba = ImprovedSpeMamba_NoMultiScaleConv(hidden_dim, token_num=token_num, use_residual=use_residual, group_num=group_num)
        elif mamba_type == 'both':
            self.mamba = ImprovedSpeMamba_NoMultiScaleConv(hidden_dim, token_num=token_num, use_residual=use_residual, group_num=group_num)

        # 使用 1x1 卷积调整 swin_features 的通道数为 128
        self.swin_conv = nn.Conv2d(96, 128, kernel_size=1, stride=1)

        # 分类头
        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(group_num, 128),
            nn.SiLU(),
            nn.Conv2d(in_channels=128, out_channels=num_classes, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        # 通过Patch Embedding进行初步特征映射
        print(x.shape)
        x = self.patch_embedding(x)

        # 通过Swin Transformer提取空间特征
        swin_features = self.swin_transformer(x)

        # 通过Mamba模块提取光谱特征
        mamba_features = self.mamba(x)

        # 调整 swin_features 的通道数，使其与 mamba_features 的通道数匹配
        swin_features = self.swin_conv(swin_features)

        # 使用上采样确保两个特征图尺寸匹配
        mamba_features = F.interpolate(mamba_features, size=(swin_features.shape[2], swin_features.shape[3]), mode='bilinear', align_corners=False)

        # 融合Swin Transformer和Mamba的特征
        fusion_features = swin_features + mamba_features

        # 最终分类
        logits = self.cls_head(fusion_features)
        print('logits: ', logits.shape)

        return logits

class ImprovedSpeMamba_NoMultiScale(nn.Module):
    def __init__(self, channels, token_num=8, use_residual=True, group_num=4):
        super(ImprovedSpeMamba_NoMultiScale, self).__init__()
        self.token_num = token_num
        self.use_residual = use_residual

        self.group_channel_num = math.ceil(channels / token_num)
        self.channel_num = self.token_num * self.group_channel_num

        self.mamba = Mamba(
            d_model=self.group_channel_num,
            d_state=16,
            d_conv=4,
            expand=2,
        )

        self.proj = nn.Sequential(
            nn.GroupNorm(group_num, self.channel_num),
            nn.SiLU()
        )

    def padding_feature(self, x):
        B, C, H, W = x.shape
        if C < self.channel_num:
            pad_c = self.channel_num - C
            pad_features = torch.zeros((B, pad_c, H, W)).to(x.device)
            cat_features = torch.cat([x, pad_features], dim=1)
            return cat_features
        else:
            return x

    def forward(self, x):
        x_pad = self.padding_feature(x)  # 仅进行通道填充
        B, C, H, W = x_pad.shape
        x_pad = x_pad.view(B * H * W, self.token_num, self.group_channel_num)
        x_recon = self.mamba(x_pad)
        x_recon = x_recon.view(B, C, H, W)  # reshape to (batch_size, channels, height, width)
        x_proj = self.proj(x_recon)
        return x + x_proj if self.use_residual else x_proj


class SwinMambaHSI_NoMultiScale(nn.Module):
    def __init__(self, in_channels=128, hidden_dim=64, num_classes=10, token_num=4, group_num=4, use_residual=True, mamba_type='both'):
        super(SwinMambaHSI_NoMultiScale, self).__init__()
        self.token_num = token_num
        self.use_residual = use_residual

        # Patch Embedding Layer
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(group_num, hidden_dim),
            nn.SiLU()
        )

        # Swin Transformer Block
        self.swin_transformer = SwinTransformerBlock(hidden_dim)

        # Mamba 模块，没有多尺度卷积
        if mamba_type == 'spa':
            self.mamba = ImprovedSpeMamba_NoMultiScale(hidden_dim, token_num=token_num, use_residual=use_residual, group_num=group_num)
        elif mamba_type == 'spe':
            self.mamba = ImprovedSpeMamba_NoMultiScale(hidden_dim, token_num=token_num, use_residual=use_residual, group_num=group_num)
        elif mamba_type == 'both':
            self.mamba = ImprovedSpeMamba_NoMultiScale(hidden_dim, token_num=token_num, use_residual=use_residual, group_num=group_num)

        # 使用 1x1 卷积调整 swin_features 的通道数为 128
        self.swin_conv = nn.Conv2d(96, 128, kernel_size=1, stride=1)

        # 分类头
        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(group_num, 128),
            nn.SiLU(),
            nn.Conv2d(in_channels=128, out_channels=num_classes, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        # 通过Patch Embedding进行初步特征映射
        print(x.shape)
        x = self.patch_embedding(x)

        # 通过Swin Transformer提取空间特征
        swin_features = self.swin_transformer(x)

        # 通过Mamba模块提取光谱特征
        mamba_features = self.mamba(x)

        # 调整 swin_features 的通道数，使其与 mamba_features 的通道数匹配
        swin_features = self.swin_conv(swin_features)

        # 使用上采样确保两个特征图尺寸匹配
        mamba_features = F.interpolate(mamba_features, size=(swin_features.shape[2], swin_features.shape[3]), mode='bilinear', align_corners=False)

        # 融合Swin Transformer和Mamba的特征
        fusion_features = swin_features + mamba_features

        # 最终分类
        logits = self.cls_head(fusion_features)
        print('logits: ', logits.shape)

        return logits
