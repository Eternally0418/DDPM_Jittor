import jittor as jt
from jittor import nn, Module

# ------------------------ 基础组件 ------------------------ #
class Swish(Module):
    def execute(self, x):
        return x * jt.sigmoid(x)
    #这个就是silu激活函数即swish激活函数的β此时是1

#这个类的作用是将时间步t转换为高维的时间嵌入向量，
class TimeEmbedding(Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.linear1 = nn.Linear(emb_dim // 4, emb_dim)
        self.linear2 = nn.Linear(emb_dim,        emb_dim)
        self.act     = Swish()                                 #  两层线性 + Swish 激活
        self.emb_dim = emb_dim

    def execute(self, t):
        if t.ndim == 0:
            t = t.unsqueeze(0)          #如果t的维度是0（标量），将其扩展成只存储一个数的数组
        t = t.float32().reshape((-1, 1))                # 将这个数组中的每一个数都变成一个数组，形状就是[B,1]，标量的情况就是【1，1】
        half = self.emb_dim // 8        # //8是对应上面设计的linear1是//4，因为cos和sin拼接有个两倍
        inv  = jt.exp(
            jt.arange(half, dtype=jt.float32)
              .multiply(-jt.log(jt.float32(10000.0)) / (half - 1))
        )                                                               #inv[i] = 1 / (10000 ** (i / (half - 1)))
                                                                        #用以构造一组从大到小的频率
        pos = jt.concat([jt.sin(t*inv), jt.cos(t*inv)], dim=1)      #t和inv乘积是广播【B，emb/8】，sin和cos拼接
        return self.linear2(self.act(self.linear1(pos)))        #通过正余弦编码对t先进行一个高维映射，然后通过非线性网络增强表示能力

#输入一个图像特征图x，加上一个时间嵌入t_emb，经过两个卷积+残差链接后输出新的特征图
class ResidualBlock(Module):
    def __init__(self, in_channels, out_channels,
                 time_emb_dim, dropout=0.1, groups=8):
        super().__init__()
        self.norm1   = nn.GroupNorm(groups, in_channels)        #是一种归一化操作
        self.conv1   = nn.Conv2d(in_channels, out_channels, 3, padding=1)       #卷积操作
        self.norm2   = nn.GroupNorm(groups, out_channels)       #是一种归一化操作
        self.conv2   = nn.Conv2d(out_channels, out_channels, 3, padding=1)     #卷积操作
        self.act     = Swish()
        self.drop    = nn.Dropout(dropout)      #防止过拟合，在训练时候随机丢弃一些神经元
        self.short   = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )           #如果通道数不相等，就先进行卷积来调整通道数，如果相等就返回本身
        self.time_proj = nn.Linear(time_emb_dim, out_channels)      #时间嵌入引入部分

    def execute(self, x, t_emb):
        h = self.conv1(self.act(self.norm1(x)))                 #第一层：归一化+激活+卷积
        h = h + self.time_proj(t_emb).unsqueeze(2).unsqueeze(3)         #加入时间嵌入扩展为 [B,C,1,1]，然后通过自动广播就加入到了【B，C，H，W】
        h = self.conv2(self.drop(self.act(self.norm2(h))))      # 第2层：归一化→激活→Dropout→卷积
        return h + self.short(x)         # 加上残差分支

class AttentionBlock(Module):
    def __init__(self, channels, num_heads=1, head_dim=None, groups=8):
        super().__init__()
        if head_dim is None:
            assert channels % num_heads == 0
            head_dim = channels // num_heads
        inner = num_heads * head_dim

        self.norm  = nn.GroupNorm(groups, channels)         #标准化
        self.qkv   = nn.Conv(channels, inner*3, 1)          #同时输出QKV
        self.proj  = nn.Conv(inner,   channels, 1)          #attention输出的投影
        self.scale = head_dim ** -0.5           #注意力缩放系数
        self.heads = num_heads
        self.hdim  = head_dim

    def execute(self, x, t_emb=None):
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x))        #先标准化，然后通过一层卷积输出【B，3C，H，W】
        q, k, v = jt.chunk(qkv, 3, dim=1)           # 按通道拆成 Q K V 分别是【B，C，H，W】

        def reshape(t):             #reshape 的目的是将 [B, C, H, W] → [B, heads, head_dim, H×W]，拆分多头注意力的输入结构，使得可以对每个头进行注意力计算
            return t.reshape((B, self.heads, self.hdim, H*W))
        q = reshape(q).permute(0,1,3,2)            # [B,h,N,d]
        k = reshape(k)                             # [B,h,d,N]
        v = reshape(v).permute(0,1,3,2)            # [B,h,N,d]

        attn = nn.softmax(jt.matmul(q, k)*self.scale, -1)       #q·k^T 得到注意力权重
        out  = jt.matmul(attn, v)                  # softmax 规范化后，乘以 v 得到输出
        out  = out.permute(0,1,3,2).reshape((B, -1, H, W))      #把 [B, h, N, d] 的结果 reshape 回 [B, C, H, W]
        return x + self.proj(out)       #进行残差链接，proj是将拆开的头再融合一起“各自学到的知识”融合成一个更强的全局表征

# ---------------------------- UNet ---------------------------- #
class UNet(Module):
    def __init__(self, image_channels=1, base_channels=32,
                 channel_mults=(1,2,4), num_res_blocks=2):
        super().__init__()
        self.channel_mults  = channel_mults     #每层特征通道数倍数
        self.num_res_blocks = num_res_blocks    #每层堆几个ResBlock
        time_dim            = base_channels * 4         #时间嵌入维度

        # 时间嵌入 & 输入投影
        self.time_embed = TimeEmbedding(time_dim)       #将整数时间步 t（0~999）编码为一个 time_dim 的向量
        self.in_conv    = nn.Conv2d(image_channels, base_channels, 3, padding=1)        #把图像从 [1, 28, 28] 映射为 [32, 28, 28] 特征（1变成32通道）

        # ---------------- Encoder ---------------- #
        self.down_blocks   = nn.ModuleList()
        self.downsamples   = nn.ModuleList()
        in_ch = base_channels

        for i, mult in enumerate(channel_mults):        #遍历每层特征通道数倍数，依次构造
            out_ch = base_channels * mult
            res_level = nn.ModuleList([
                ResidualBlock(in_ch if j==0 else out_ch,
                              out_ch, time_dim)
                for j in range(num_res_blocks)      #每层有多个res_blocks
            ])
            # 最深层加注意力（7×7）
            if i == len(channel_mults)-1:
                res_level.append(AttentionBlock(out_ch))
            self.down_blocks.append(res_level)
            in_ch = out_ch
            if i != len(channel_mults)-1:          # 下采样
                self.downsamples.append(
                    nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)     #图像H，W减半
                )

        # ---------------- Bottleneck ---------------- #
        self.mid_block1 = ResidualBlock(in_ch, in_ch, time_dim)
        self.mid_attn   = AttentionBlock(in_ch)
        self.mid_block2 = ResidualBlock(in_ch, in_ch, time_dim)     #mid_block1 和 mid_block2 是 ResBlock，Attention 是中间插入的

        # ---------------- Decoder ---------------- #
        self.up_blocks  = nn.ModuleList()   # 每个元素=ModuleList(若干ResBlock)
        self.upsamples  = nn.ModuleList()
        for i, mult in enumerate(reversed(channel_mults)):
            skip_ch = base_channels * mult
            res_level = nn.ModuleList()
            for j in range(num_res_blocks):             #第一个res_block块要做跳跃链接
                in_res = in_ch + skip_ch if j==0 else skip_ch
                res_level.append(ResidualBlock(in_res, skip_ch, time_dim))
                in_ch = skip_ch
            # 可在 14×14 或 7×7 额外插 Attention；这里只在 Bottleneck 已插
            #这里没有再加注意力层，就只有刚刚encoder的最底层加了一个注意力层
            self.up_blocks.append(res_level)
            if i != len(channel_mults)-1:          # 上采样
                self.upsamples.append(
                    nn.ConvTranspose(in_ch, in_ch, 4, stride=2, padding=1)
                )

        # 输出层
        self.out_norm = nn.GroupNorm(8, in_ch)
        self.out_act  = Swish()
        self.out_conv = nn.Conv2d(in_ch, image_channels, 3, padding=1)              #将通道数从中间的通道（如32）还原回原始的 image_channels（如1）

    # ------------------------ Forward ------------------------ #
    def execute(self, x, t):
        t_emb = self.time_embed(t)
        x = self.in_conv(x)         #输入图像 x 原始形状是 [B, 1, 28, 28]，卷积之后变成[B, 32, 28, 28]

        # -------- Encoder -------- #
        skips = []
        for i, res_level in enumerate(self.down_blocks):
            for block in res_level:
                x = block(x, t_emb) if isinstance(block, ResidualBlock) else block(x)
            skips.append(x)
            if i < len(self.downsamples):
                x = self.downsamples[i](x)  # 下采样

        # -------- Bottleneck -------- #
        x = self.mid_block2(self.mid_attn(self.mid_block1(x, t_emb)), t_emb)

        # -------- Decoder -------- #
        for i, res_level in enumerate(self.up_blocks):
            skip = skips.pop()
            x = jt.concat([x, skip], dim=1)     #这就是为什么刚才第一个要进行跳跃链接
            for block in res_level:
                x = block(x, t_emb)
            if i < len(self.upsamples):
                x = self.upsamples[i](x)

        return self.out_conv(self.out_act(self.out_norm(x)))        #用一组归一化 + 激活 + 卷积，把通道数变回原来的图像通道（比如 1 通道）。

