import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# 固定 GroupNorm 分组数为 8
def _group_norm(num_channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(8, num_channels)

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class TimeEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        half_dim = embed_dim // 8
        assert half_dim > 0
        self.register_buffer(
            "freqs",
            torch.exp(-math.log(10000.0) * torch.arange(half_dim, dtype=torch.float32) / (half_dim - 1)),
            persistent=False
        )
        self.fc1 = nn.Linear(embed_dim // 4, embed_dim)
        self.act = Swish()
        self.fc2 = nn.Linear(embed_dim, embed_dim)

    def forward(self, t):
        t = t.float()[:, None] * self.freqs[None]
        emb = torch.cat([t.sin(), t.cos()], dim=1)
        return self.fc2(self.act(self.fc1(emb)))

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_ch, dropout=0.1):
        super().__init__()
        self.norm1 = _group_norm(in_ch)
        self.act1 = Swish()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_ch, out_ch)
        self.norm2 = _group_norm(out_ch)
        self.act2 = Swish()
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(self.act1(self.norm1(x)))
        h = h + self.time_proj(t_emb)[:, :, None, None]
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))
        return h + self.shortcut(x)

class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=1, d_k=None):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_k or channels // num_heads
        self.scale = self.d_k ** -0.5
        self.norm = _group_norm(channels)
        self.qkv = nn.Linear(channels, self.num_heads * self.d_k * 3)
        self.proj = nn.Linear(self.num_heads * self.d_k, channels)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, -1).permute(0, 2, 1)
        qkv = self.qkv(h).view(B, -1, self.num_heads, 3 * self.d_k)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        attn = torch.einsum("bihd,bjhd->bijh", q, k) * self.scale
        attn = F.softmax(attn, dim=2)
        h = torch.einsum("bijh,bjhd->bihd", attn, v).contiguous().view(B, -1, self.num_heads * self.d_k)
        h = self.proj(h).permute(0, 2, 1).view(B, C, H, W)
        return x + h

class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1)

    def forward(self, x):
        return self.deconv(x)

class ResAttnBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_ch, use_attn=False, dropout=0.1):
        super().__init__()
        self.res = ResidualBlock(in_ch, out_ch, time_ch, dropout)
        self.attn = AttentionBlock(out_ch) if use_attn else nn.Identity()

    def forward(self, x, t_emb):
        return self.attn(self.res(x, t_emb))

class UNet(nn.Module):
    def __init__(self, image_channels=1, base_channels=32, channel_mults=(1, 2, 4), num_res_blocks=2, attn_resolutions=(2,), dropout=0.1):
        super().__init__()
        self.image_channels = image_channels
        self.base_channels = base_channels
        self.conv_in = nn.Conv2d(image_channels, base_channels, 3, padding=1)
        time_dim = base_channels * 4
        self.time_embed = TimeEmbedding(time_dim)

        self.down = nn.ModuleList()
        in_ch = base_channels
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            for j in range(num_res_blocks):
                self.down.append(
                    ResAttnBlock(in_ch, out_ch, time_dim, use_attn=False, dropout=dropout)
                )
                in_ch = out_ch


            if i == len(channel_mults) - 1:
                self.down.append(AttentionBlock(out_ch))


            if i < len(channel_mults) - 1:
                self.down.append(Downsample(in_ch))

        self.mid = nn.ModuleList([
            ResidualBlock(in_ch, in_ch, time_dim, dropout),
            AttentionBlock(in_ch),
            ResidualBlock(in_ch, in_ch, time_dim, dropout),
        ])

        self.up = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            for _ in range(num_res_blocks):
                self.up.append(ResAttnBlock(in_ch + out_ch, out_ch, time_dim, use_attn=False, dropout=dropout))
                in_ch = out_ch
            if i > 0:
                self.up.append(Upsample(in_ch))

        self.norm_out = _group_norm(in_ch)
        self.act = Swish()
        self.conv_out = nn.Conv2d(in_ch, image_channels, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_embed(t)
        x = self.conv_in(x)
        skips = []
        for block in self.down:
            if isinstance(block, Downsample):
                x = block(x)
            else:
                x = block(x, t_emb)
                skips.append(x)
        for layer in self.mid:
            x = layer(x, t_emb) if isinstance(layer, ResidualBlock) else layer(x)
        for block in self.up:
            if isinstance(block, Upsample):
                x = block(x)
            else:
                x = torch.cat([x, skips.pop()], dim=1)
                x = block(x, t_emb)
        return self.conv_out(self.act(self.norm_out(x)))
