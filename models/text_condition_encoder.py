"""
TextConditionEncoder (v6.0)
============================
从 5 通道文本条件 (HR 尺度) 提取多分辨率特征, 供 TextAwareAdapter 使用.

输入:
    cond: [B, 5, H, W]  HR 尺度条件
        Ch.1 hq_gray              HR_bicubic 灰度    (LR bicubic 4× 上采后灰度化)
        Ch.2 text_mask            anno 渲染          (text region binary mask)
        Ch.3 density              字符密度图          (text_mask 的高斯模糊)
        Ch.4 sobel_x              HR_bicubic Sobel x
        Ch.5 sobel_y              HR_bicubic Sobel y

    数值约定 (Phase 1 build_cond_5ch 工具函数中归一化):
        Ch.1 hq_gray:      [-1, 1]  (跟 lr/hr 同 range)
        Ch.2 text_mask:    [0, 1]
        Ch.3 density:      [0, 1]
        Ch.4-5 sobel_x/y:  [-1, 1]  (除以 max abs 归一化, 防数值爆炸)

输出 (在 latent 尺度, 即 H/8 × W/8 后再下采):
    feat_q4:  [B, FEAT_CH, H/32, W/32]  → Adapter @ down_block_2 后  (UNet latent 1/4)
    feat_q8:  [B, FEAT_CH, H/64, W/64]  → Adapter @ mid_block 后    (UNet latent 1/8)

  例:
    HR=512  → latent 64×64  → feat_q4 = 16×16, feat_q8 = 8×8
    HR=768  → latent 96×96  → feat_q4 = 24×24, feat_q8 = 12×12
    HR=1024 → latent 128×128 → feat_q4 = 32×32, feat_q8 = 16×16

设计要点:
  - 总参数量 ~3M (FPN-lite, base=40)
  - 用 GroupNorm 而非 BatchNorm (适配小 batch + 多分辨率)
  - 输出 head 用 small random init (std=1e-4), 而非 zero init.
    原因: 若 head zero, 会和下游 Adapter.output_proj zero 形成双重梯度阻断,
    导致 stem/stage1-6 永远拿不到梯度. small random 让 feat 量级 ~1e-4,
    配合 Adapter.output_proj zero 仍能保证 delta=0 (因为 0×any=0), 同时梯度可流.
  - 6 个下采样 stage: 1x → 1/2 → 1/4 → 1/8 → 1/16 → 1/32 → 1/64
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# 输出特征通道数 (送给 Adapter 后会再投影到 UNet feature 通道, 1280)
FEAT_CH = 128


def conv3x3(in_ch: int, out_ch: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_ch: int, out_ch: int) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False)


def gn(num_channels: int) -> nn.GroupNorm:
    """GroupNorm: num_groups 取 32 或 num_channels 的因数."""
    num_groups = min(32, num_channels)
    while num_channels % num_groups != 0:
        num_groups -= 1
    return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)


class ResBlock(nn.Module):
    """简化的 ResNet block (GN + SiLU + Conv3x3 + GN + SiLU + Conv3x3)"""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.norm1 = gn(in_ch)
        self.act1 = nn.SiLU(inplace=True)
        self.conv1 = conv3x3(in_ch, out_ch, stride=stride)
        self.norm2 = gn(out_ch)
        self.act2 = nn.SiLU(inplace=True)
        self.conv2 = conv3x3(out_ch, out_ch, stride=1)

        # 残差连接 (通道/分辨率不一致时用 1x1 投影)
        if in_ch != out_ch or stride != 1:
            self.shortcut = nn.Sequential(
                gn(in_ch),
                conv1x1(in_ch, out_ch) if stride == 1 else
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.conv1(self.act1(self.norm1(x)))
        x = self.conv2(self.act2(self.norm2(x)))
        return x + residual


class TextConditionEncoder(nn.Module):
    """
    从 6 通道 HR 条件提取 1/32 和 1/64 两个分辨率的特征.

    架构 (HR=512 时为例):
        输入 [B, 5, 512, 512]
        ↓ Stem (Conv 3x3, 5→base)
        Stage0  [B, base, 512, 512]
        ↓ ResBlock + Down (stride 2)
        Stage1  [B, base, 256, 256]
        ↓ ResBlock + Down
        Stage2  [B, base*2, 128, 128]
        ↓ ResBlock + Down
        Stage3  [B, base*2, 64, 64]   (= UNet latent 1×, but unused)
        ↓ ResBlock + Down
        Stage4  [B, base*4, 32, 32]   (UNet latent 1/2, unused)
        ↓ ResBlock + Down
        Stage5  [B, base*4, 16, 16]  ⭐ feat_q4 (UNet latent 1/4)
        ↓ ResBlock + Down
        Stage6  [B, base*8, 8, 8]    ⭐ feat_q8 (UNet latent 1/8)

        输出头:
        feat_q4 → Conv 1x1 (zero init) → [B, FEAT_CH, H/32, W/32]
        feat_q8 → Conv 1x1 (zero init) → [B, FEAT_CH, H/64, W/64]

    参数量预算 (base=32):
        Stem + 6 Stage 各 1 ResBlock = ~3M
        2 个 head Conv 1x1            = ~50K
        合计 ~3M (< 5M 预算)

    若 base=48 (推荐):
        合计 ~6M (略超预算, 但容量更足)

    base=40:
        合计 ~4.5M (推荐起点)
    """
    def __init__(
        self,
        in_channels: int = 5,
        base_channels: int = 40,
        feat_channels: int = FEAT_CH,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.feat_channels = feat_channels

        c0 = base_channels       # 40
        c1 = base_channels       # 40
        c2 = base_channels * 2   # 80
        c3 = base_channels * 2   # 80
        c4 = base_channels * 4   # 160
        c5 = base_channels * 4   # 160  → feat_q4 source
        c6 = base_channels * 8   # 320  → feat_q8 source

        # ---- Stem: 1x → c0 ----
        self.stem = nn.Sequential(
            conv3x3(in_channels, c0),
            gn(c0),
            nn.SiLU(inplace=True),
        )

        # ---- 6 个下采样 stage ----
        # 每个 stage = ResBlock(in→out, stride=2)
        # 即 1x → 1/2 → 1/4 → 1/8 → 1/16 → 1/32 → 1/64
        self.stage1 = ResBlock(c0, c1, stride=2)  # 1/2
        self.stage2 = ResBlock(c1, c2, stride=2)  # 1/4
        self.stage3 = ResBlock(c2, c3, stride=2)  # 1/8 (= latent 1x)
        self.stage4 = ResBlock(c3, c4, stride=2)  # 1/16 (= latent 1/2)
        self.stage5 = ResBlock(c4, c5, stride=2)  # 1/32 (= latent 1/4) ⭐
        self.stage6 = ResBlock(c5, c6, stride=2)  # 1/64 (= latent 1/8) ⭐

        # ---- 输出 head (small random init, 配合 Adapter.output_proj zero 让训练初期 delta=0) ----
        # feat_q4: 给 down_blocks[1] 后的 Adapter (640 ch, 16×16)
        self.head_q4 = conv1x1(c5, feat_channels)
        # feat_q8: 给 down_blocks[2] 后的 Adapter (1280 ch, 8×8)
        self.head_q8 = conv1x1(c6, feat_channels)

        # ⚠️ head 不能 zero init! 否则会和 Adapter.output_proj zero init 形成
        # 双重梯度阻断 (Adapter.cond_proj 之后的反传会通过 head 阻塞回 stem/stage1-6).
        # 这里用 small random (std=1e-4): feat 量级 ~ 1e-4,
        # 配合 Adapter output_proj zero, 仍保证 delta=0 (因为 0×anything=0),
        # 但 head 自身有梯度可流通.
        nn.init.normal_(self.head_q4.weight, mean=0.0, std=1e-4)
        nn.init.normal_(self.head_q8.weight, mean=0.0, std=1e-4)

    def forward(self, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            cond: [B, 5, H, W]  HR 尺度 5 通道条件

        Returns:
            feat_q4: [B, FEAT_CH, H/32, W/32]
            feat_q8: [B, FEAT_CH, H/64, W/64]
        """
        x = self.stem(cond)            # [B, c0, H,    W   ]
        x = self.stage1(x)             # [B, c1, H/2,  W/2 ]
        x = self.stage2(x)             # [B, c2, H/4,  W/4 ]
        x = self.stage3(x)             # [B, c3, H/8,  W/8 ]
        x = self.stage4(x)             # [B, c4, H/16, W/16]
        f5 = self.stage5(x)            # [B, c5, H/32, W/32] ⭐
        f6 = self.stage6(f5)           # [B, c6, H/64, W/64] ⭐

        feat_q4 = self.head_q4(f5)     # [B, FEAT_CH, H/32, W/32]
        feat_q8 = self.head_q8(f6)     # [B, FEAT_CH, H/64, W/64]

        return feat_q4, feat_q8

    @torch.no_grad()
    def num_parameters(self, only_trainable: bool = True) -> int:
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


# ==================== 自检/单元测试 ====================

def _selftest():
    """快速验证模块 forward + 输出尺寸 + zero init."""
    print("=" * 60)
    print("TextConditionEncoder selftest")
    print("=" * 60)

    enc = TextConditionEncoder(in_channels=5, base_channels=40)
    n_params = enc.num_parameters()
    print(f"参数量: {n_params:,} ({n_params / 1e6:.2f} M)")
    assert n_params < 8_000_000, f"参数量超出预算 (8M): {n_params:,}"

    # 测试三种 HR 尺寸 (multi-scale training)
    test_cases = [
        (512, 16, 8),    # HR=512  → feat_q4=16×16,  feat_q8=8×8
        (768, 24, 12),   # HR=768  → feat_q4=24×24,  feat_q8=12×12
        (1024, 32, 16),  # HR=1024 → feat_q4=32×32, feat_q8=16×16
    ]

    print(f"\n{'HR':>6} | {'feat_q4 shape':^25} | {'feat_q8 shape':^25}")
    print("-" * 60)
    for hr, expected_q4, expected_q8 in test_cases:
        cond = torch.randn(2, 5, hr, hr)
        feat_q4, feat_q8 = enc(cond)
        assert feat_q4.shape == (2, FEAT_CH, expected_q4, expected_q4), \
            f"feat_q4 shape 错误: {feat_q4.shape}, 期望 (2, {FEAT_CH}, {expected_q4}, {expected_q4})"
        assert feat_q8.shape == (2, FEAT_CH, expected_q8, expected_q8), \
            f"feat_q8 shape 错误: {feat_q8.shape}, 期望 (2, {FEAT_CH}, {expected_q8}, {expected_q8})"
        print(f"{hr:>6} | {str(tuple(feat_q4.shape)):^25} | {str(tuple(feat_q8.shape)):^25}")

    # 验证 small random init: 训练初期输出极小但非零 (与 Adapter zero 配合)
    cond = torch.randn(2, 5, 512, 512)
    feat_q4, feat_q8 = enc(cond)
    feat_q4_mag = feat_q4.abs().max().item()
    feat_q8_mag = feat_q8.abs().max().item()
    # 期望 ~ 1e-4 ~ 1e-2 量级 (random conv1x1 weight std=1e-4 × 输入量级)
    assert feat_q4_mag < 1.0, f"feat_q4 量级过大 ({feat_q4_mag:.2e}), 检查 head init"
    assert feat_q8_mag < 1.0, f"feat_q8 量级过大 ({feat_q8_mag:.2e}), 检查 head init"
    assert feat_q4_mag > 0, f"feat_q4 全 0, 检查 head init (不应是 zero init)"
    print(f"\n✅ Small random init 验证通过:")
    print(f"   feat_q4 max abs = {feat_q4_mag:.2e}  (应 ~1e-3, 小于 1)")
    print(f"   feat_q8 max abs = {feat_q8_mag:.2e}  (应 ~1e-3, 小于 1)")

    # 验证梯度回传 (head 本来就有梯度可流, 不需要"打破 zero init")
    cond = torch.randn(2, 5, 512, 512, requires_grad=False)
    feat_q4, feat_q8 = enc(cond)
    loss = feat_q4.pow(2).mean() + feat_q8.pow(2).mean()
    loss.backward()
    grad_ok = all(p.grad is not None for p in enc.parameters() if p.requires_grad)
    assert grad_ok, "存在参数 grad=None"
    print(f"✅ 梯度回传验证通过 (loss = {loss.item():.4f})")

    # 显存占用估算 (HR=1024, batch=4, fp32)
    import time
    enc.cuda() if torch.cuda.is_available() else None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        cond = torch.randn(4, 5, 1024, 1024).cuda()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        for _ in range(3):
            feat_q4, feat_q8 = enc(cond)
        torch.cuda.synchronize()
        elapsed = (time.time() - t0) / 3
        peak_mem = torch.cuda.max_memory_allocated() / 1024 ** 3
        print(f"\nGPU 性能 (HR=1024, batch=4, fp32):")
        print(f"  单次 forward: {elapsed*1000:.1f} ms")
        print(f"  峰值显存:     {peak_mem:.2f} GB")

    print(f"\n✅ TextConditionEncoder selftest 全部通过!")
    print("=" * 60)


if __name__ == "__main__":
    _selftest()
