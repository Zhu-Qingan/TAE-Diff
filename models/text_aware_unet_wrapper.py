"""
TextAwareUNetWrapper (v6.0)
============================
把 TextConditionEncoder + DualAdapter 通过 forward hook 接入到 UNet 的两个注入点
(适配 SD 1.5 / SD 2.1-base 的 UNet, latent 64×64 输入):

    Hook A — down_blocks[1] 后:  unet feat = [B, 640, 16, 16]   ⭐ 字形/连字/衬线
    Hook B — down_blocks[2] 后:  unet feat = [B, 1280, 8, 8]    ⭐ 语义/布局

    (mid_block 不挂 — 避免破坏全局语义, SD 的风格瓶颈在 mid_block)

关键设计:
  * 完全不动 UNet 源码 (使用 PyTorch register_forward_hook)
  * cond_5ch 通过 wrapper 实例属性传递, hook 从 self 读取 (hook 闭包不支持传额外参数)
  * 训练初期 (TextConditionEncoder.head_q4/q8 zero + Adapter.output_proj zero)
    严格保证 ours = baseline (双重 zero)
  * 兼容 CODSR 的 modulation_params (照传 UNet, wrapper 不干预)

调用方式:
    wrapper = TextAwareUNetWrapper(unet)  # unet 是已经 load 好的 UNet2DConditionModel

    # 训练 / 推理时 (cond_5ch 已经 build 好):
    wrapper.set_condition(cond_5ch)
    out = wrapper.unet(latent, t, encoder_hidden_states=prompt, ...).sample
    wrapper.clear_condition()  # 用完后清掉, 防止下次 forward 误用旧 cond

  或用 context manager 自动清理:
    with wrapper.condition_scope(cond_5ch):
        out = wrapper.unet(latent, t, ...).sample

工具函数:
    build_cond_5ch(lr, text_mask)
        从 LR / text_mask 在 GPU 上构造 5 通道条件:
            ch1 hq_gray   = HR_bicubic 灰度 (bicubic 4× 上采)
            ch2 text_mask = anno 渲染 mask (传入)
            ch3 density   = text_mask 高斯模糊
            ch4 sobel_x   = HR_bicubic Sobel x (归一化到 [-1, 1])
            ch5 sobel_y   = HR_bicubic Sobel y (归一化到 [-1, 1])
"""

from contextlib import contextmanager
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# 兼容两种调用方式:
#   1. python models/text_aware_unet_wrapper.py  (作为脚本直接跑, 无 package)
#   2. from models.text_aware_unet_wrapper import ...  (作为模块被 train.py 导入)
try:
    from .text_condition_encoder import TextConditionEncoder
    from .text_aware_adapter import DualAdapter, TextAwareAdapter
except ImportError:
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from text_condition_encoder import TextConditionEncoder
    from text_aware_adapter import DualAdapter, TextAwareAdapter


# ============================================================================
# 工具函数: build_cond_5ch
# ============================================================================

# Sobel kernel (固定不学习, 注册成 buffer 由 build 函数内部用)
_SOBEL_X = torch.tensor(
    [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
).view(1, 1, 3, 3)
_SOBEL_Y = torch.tensor(
    [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
).view(1, 1, 3, 3)


def _gaussian_blur(x: torch.Tensor, ksize: int = 9, sigma: float = 3.0) -> torch.Tensor:
    """简易高斯模糊 (depthwise conv2d)."""
    b, c, h, w = x.shape
    half = ksize // 2
    coords = torch.arange(ksize, dtype=x.dtype, device=x.device) - half
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel_2d = g[:, None] * g[None, :]                                # [k, k]
    kernel = kernel_2d.view(1, 1, ksize, ksize).expand(c, 1, ksize, ksize)
    return F.conv2d(x, kernel, padding=half, groups=c)


def build_cond_5ch(
    lr: torch.Tensor,
    text_mask: torch.Tensor,
    upscale: int = 4,
) -> torch.Tensor:
    """
    在 GPU 上构造 5 通道条件张量.

    Args:
        lr:        [B, 3, H/4, W/4]   LR (range [-1, 1])
        text_mask: [B, 1, H,   W  ]   anno 渲染 mask (range [0, 1])
                                       (如 dataset 已经是 LR 尺度, 调用前需要 upsample 到 HR)
        upscale:   LR → HR 的上采系数 (默认 4)

    Returns:
        cond: [B, 5, H, W]
            ch1 hq_gray:    [-1, 1]
            ch2 text_mask:  [0, 1]
            ch3 density:    [0, 1]
            ch4 sobel_x:    [-1, 1]
            ch5 sobel_y:    [-1, 1]
    """
    B, _, h_lr, w_lr = lr.shape
    _, _, h_hr, w_hr = text_mask.shape
    assert h_hr == h_lr * upscale and w_hr == w_lr * upscale, (
        f"text_mask 的尺寸 {(h_hr, w_hr)} 与 lr × upscale={upscale} 后的尺寸 "
        f"{(h_lr * upscale, w_lr * upscale)} 不匹配"
    )

    device, dtype = lr.device, lr.dtype

    # ---- ch1: hq_gray = HR_bicubic 灰度 ----
    hr_bicubic = F.interpolate(lr, scale_factor=upscale, mode="bicubic", align_corners=False)
    hr_bicubic = hr_bicubic.clamp(-1.0, 1.0)
    # rec.601 灰度: 0.299R + 0.587G + 0.114B
    weights_gray = torch.tensor([0.299, 0.587, 0.114], device=device, dtype=dtype)
    hq_gray = (hr_bicubic * weights_gray.view(1, 3, 1, 1)).sum(dim=1, keepdim=True)  # [B,1,H,W]
    # hq_gray 已经在 [-1, 1] 范围

    # ---- ch2: text_mask (传入即可, 已在 [0,1]) ----
    text_mask = text_mask.clamp(0.0, 1.0)

    # ---- ch3: density = text_mask 的高斯模糊 ----
    # 模糊核 ksize=9 sigma=3 (相对 HR=512 大约覆盖一个字的尺度)
    density = _gaussian_blur(text_mask, ksize=9, sigma=3.0).clamp(0.0, 1.0)

    # ---- ch4-5: Sobel x/y on hq_gray, 归一化到 [-1, 1] ----
    sobel_x_kernel = _SOBEL_X.to(device=device, dtype=dtype)
    sobel_y_kernel = _SOBEL_Y.to(device=device, dtype=dtype)
    sx = F.conv2d(hq_gray, sobel_x_kernel, padding=1)
    sy = F.conv2d(hq_gray, sobel_y_kernel, padding=1)
    # Sobel 输出范围理论上 [-8, 8] (灰度 ∈ [-1, 1] 时), 用 max abs 归一化
    sx_max = sx.abs().amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
    sy_max = sy.abs().amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
    sx = sx / sx_max  # [-1, 1]
    sy = sy / sy_max

    cond = torch.cat([hq_gray, text_mask, density, sx, sy], dim=1)  # [B, 5, H, W]
    return cond


# ============================================================================
# TextAwareUNetWrapper
# ============================================================================

class TextAwareUNetWrapper(nn.Module):
    """
    包装 UNet (不修改 UNet 源码) + DualAdapter + TextConditionEncoder.

    持有的可训练模块:
        - text_encoder:  TextConditionEncoder  (~2.6M params)
        - dual_adapter:  DualAdapter            (~10.3M params)
        UNet 本体不在 wrapper 里更新 (训练时通常 freeze 或 LoRA, 我们这里只挂 hook)

    Attributes:
        unet:           外部传入的 UNet2DConditionModel (不被 wrapper 持有的参数管理)
        text_encoder:   TextConditionEncoder
        dual_adapter:   DualAdapter
        _cur_feat_q4:   当前 batch 的 cond feat (forward 前由 set_condition 填充)
        _cur_feat_q8:
        _hook_handles:  注册的 hook 句柄, dispose 时释放
    """

    def __init__(
        self,
        unet: nn.Module,
        cond_in_channels: int = 5,
        feat_channels: int = 128,
        adapter_unet_ch_q4: int = 640,    # down_blocks[1] 输出通道
        adapter_unet_ch_q8: int = 1280,   # down_blocks[2] 输出通道
        adapter_mid_ch: int = 192,
        text_encoder_base: int = 40,
        hook_mid_block: bool = False,     # v6 默认 False, 预留未来消融
    ):
        """
        Args:
            hook_mid_block: 是否在 mid_block 后也挂 hook.
                False (默认): 仅注入 down_blocks[1] + down_blocks[2], 跳过 mid_block.
                True: 额外在 mid_block 后挂第三个 adapter.

                ⚠️ mid_block 注入的消融候选 (暂未实现, 待后续讨论):
                  (1) 三层衰减权重: q4=1.0, q8=0.8, mid=0.2
                  (2) mid_block 只改 cross-attention Q/K (不改 V):
                       B_structure = MLP(cond) 加到 attention score
                       不动 value 特征, 只改"关注哪里"
                  (3) 当前实现 (hook_mid_block=True 时): 复用 q8 同款 adapter
                       (1280ch × 8×8, full feature 注入, 同 q8 强度)
        """
        super().__init__()
        self.unet = unet  # 注意: unet 是外部对象, wrapper 不接管它的 .parameters()
                          # 但 register_module 后会被自动归入 self.parameters().
                          # 为避免训练时把 unet 参数也丢进 optimizer, 我们用 _modules 方式手动挂载

        # 用 _modules 挂载 unet 但不让它出现在 .parameters() 里 → 行不通,
        # PyTorch 没有官方方式排除 submodule 参数. 简单方案:
        # 训练脚本里手动只把 wrapper.text_encoder + wrapper.dual_adapter 给 optimizer,
        # 不把 wrapper.unet 给 optimizer 即可.

        self.hook_mid_block = hook_mid_block

        self.text_encoder = TextConditionEncoder(
            in_channels=cond_in_channels,
            base_channels=text_encoder_base,
            feat_channels=feat_channels,
        )
        self.dual_adapter = DualAdapter(
            cond_ch=feat_channels,
            unet_ch_q4=adapter_unet_ch_q4,
            unet_ch_q8=adapter_unet_ch_q8,
            mid_ch=adapter_mid_ch,
        )

        # mid_block adapter (可选, 默认不创建参数, 省 ~5M)
        if hook_mid_block:
            self.adapter_mid = TextAwareAdapter(
                cond_ch=feat_channels,
                unet_ch=adapter_unet_ch_q8,  # 同 q8: 1280
                mid_ch=adapter_mid_ch,
            )
        else:
            self.adapter_mid = None

        # 当前 condition 缓存 (forward 前 set, forward 后 clear)
        self._cur_feat_q4: Optional[torch.Tensor] = None
        self._cur_feat_q8: Optional[torch.Tensor] = None
        # v6.2 路线 H: spatial gate (text_mask 下采到 unet 注入点分辨率)
        # 形状: [B, 1, h_feat, w_feat], 软门控 ∈ [0, 1].
        # set_condition 时传 text_mask 才生成, 否则为 None (adapter 退化到不门控行为).
        self._cur_gate_q4: Optional[torch.Tensor] = None
        self._cur_gate_q8: Optional[torch.Tensor] = None

        # tile 推理时的 latent 坐标 (y0, y1, x0, x1), 单位 = latent 像素
        # 训练时永远为 None (hook 走整图分支), 只有推理 tile 时才 set.
        # 见 set_tile_slice / clear_tile_slice 文档.
        self._tile_latent_yxyx: Optional[tuple] = None

        # 注册 forward hook
        self._hook_handles = []
        self._register_hooks()

    # ------------------------------------------------------------------
    # Hook 注册
    # ------------------------------------------------------------------
    def _register_hooks(self):
        """
        Hook 位置 (适配 SD 1.5 / SD 2.1-base):
            Hook A — down_blocks[1] 后:  unet feat = [B, 640, 16, 16]  ⭐ 字形/连字/衬线
            Hook B — down_blocks[2] 后:  unet feat = [B, 1280, 8, 8]   ⭐ 语义/布局
            Hook C — mid_block 后:       unet feat = [B, 1280, 8, 8]   (可选, 默认关)

        UNet down_blocks[i] 的 forward 返回 (sample, res_samples) tuple.
        UNet mid_block 的 forward 返回 sample (tensor).

        ⭐ Tile 推理支持 (v6 inference):
        训练时 cond_feat 与 unet_feat 空间尺寸严格相等 (整图 latent 64×64).
        推理 tile 时, cond_feat 是按整张大图算的 (例 latent 128×128 → feat_q4 32×32),
        而 unet_feat 是 tile 内的 (例 16×16). 此时 hook 内自动按 self._tile_latent_yxyx
        切出对应区域的 cond_feat 喂给 adapter.
            - feat_q4 对应 latent/4 比例 (UNet down_blocks[1] 后 spatial = latent/4)
            - feat_q8 对应 latent/8 比例 (UNet down_blocks[2] 后 spatial = latent/8)
        切片是无损的, 因为 feat_q4/feat_q8 是 cond_5ch 经固定 stride 卷积下采得到的局部特征.
        """

        def _slice_cond_feat(feat, ratio_div):
            """
            按 self._tile_latent_yxyx 从整图 cond_feat / gate 切出 tile 对应区域.
            ratio_div = latent_size / feat_spatial_size (即 4 或 8).
            如果未设 _tile_latent_yxyx (= 训练 / 推理整图), 直接返回原 feat.

            注: 此函数同时用于切 cond_feat (feat_q4/q8) 和 gate (gate_q4/q8),
            它们 spatial 与 latent 的比例完全一致, 所以可以复用.
            """
            if self._tile_latent_yxyx is None or feat is None:
                return feat
            y0, y1, x0, x1 = self._tile_latent_yxyx
            fy0 = y0 // ratio_div
            fy1 = y1 // ratio_div
            fx0 = x0 // ratio_div
            fx1 = x1 // ratio_div
            return feat[..., fy0:fy1, fx0:fx1]

        # Hook A: down_blocks[1] 输出后注入 q4 (640ch × latent/4)
        def hook_down1(module, inputs, output):
            if self._cur_feat_q4 is None:
                return output  # baseline 模式直通
            sample, res_samples = output
            # latent_size = sample_spatial × 4 (down_blocks[1] 后 spatial = latent/4)
            cond = _slice_cond_feat(self._cur_feat_q4, ratio_div=4)
            gate = _slice_cond_feat(self._cur_gate_q4, ratio_div=4)
            sample = self.dual_adapter.adapter_q4(sample, cond, spatial_gate=gate)
            return (sample, res_samples)

        # Hook B: down_blocks[2] 输出后注入 q8 (1280ch × latent/8)
        def hook_down2(module, inputs, output):
            if self._cur_feat_q8 is None:
                return output
            sample, res_samples = output
            cond = _slice_cond_feat(self._cur_feat_q8, ratio_div=8)
            gate = _slice_cond_feat(self._cur_gate_q8, ratio_div=8)
            sample = self.dual_adapter.adapter_q8(sample, cond, spatial_gate=gate)
            return (sample, res_samples)

        h1 = self.unet.down_blocks[1].register_forward_hook(hook_down1)
        h2 = self.unet.down_blocks[2].register_forward_hook(hook_down2)
        self._hook_handles = [h1, h2]

        # Hook C (可选): mid_block 后注入 (复用 q8 同款 feat, 不同 adapter 实例)
        if self.hook_mid_block:
            def hook_mid(module, inputs, output):
                if self._cur_feat_q8 is None:
                    return output
                # mid_block 返回 tensor 而非 tuple
                cond = _slice_cond_feat(self._cur_feat_q8, ratio_div=8)
                gate = _slice_cond_feat(self._cur_gate_q8, ratio_div=8)
                return self.adapter_mid(output, cond, spatial_gate=gate)

            h3 = self.unet.mid_block.register_forward_hook(hook_mid)
            self._hook_handles.append(h3)

    def remove_hooks(self):
        """释放 hook (训练结束/切换 baseline 时调用)."""
        for h in self._hook_handles:
            h.remove()
        self._hook_handles = []

    # ------------------------------------------------------------------
    # Condition 管理
    # ------------------------------------------------------------------
    def set_condition(
        self,
        cond_5ch: Optional[torch.Tensor],
        text_mask: Optional[torch.Tensor] = None,
    ):
        """
        设置当前 batch 的 5 通道条件 + 空间门控.

        Args:
            cond_5ch:  [B, 5, H, W] 或 None
                      None: 关闭 Adapter 注入, 走纯 UNet baseline.
            text_mask: [B, 1, H, W] or None (v6.2 路线 H 新增)
                      不为 None 时: 生成 gate_q4 / gate_q8 (mask 下采到 hook 分辨率),
                                    hook 内传给 adapter, 限制 delta 只在 mask 区生效.
                      None 时: gate_q4/q8 = None, adapter 退化到不门控 (旧行为, 兼容老 ckpt 推理).
                      训练时强烈建议传 text_mask, 这是路线 H 的核心.

        Note:
            text_mask 的 H, W 应该与 cond_5ch 完全一致 (HR 尺度). 下采到 unet feat 分辨率用
            F.interpolate(..., mode='area'), 既保留覆盖率信息又比 nearest 平滑.
        """
        if cond_5ch is None:
            self._cur_feat_q4 = None
            self._cur_feat_q8 = None
            self._cur_gate_q4 = None
            self._cur_gate_q8 = None
            return

        feat_q4, feat_q8 = self.text_encoder(cond_5ch)
        self._cur_feat_q4 = feat_q4
        self._cur_feat_q8 = feat_q8

        # ⭐ v6.2 路线 H: 计算空间门控
        if text_mask is not None:
            assert text_mask.dim() == 4 and text_mask.shape[1] == 1, (
                f"text_mask 期望 [B, 1, H, W], 收到 {tuple(text_mask.shape)}"
            )
            # 下采到 feat_q4 / feat_q8 的 spatial. area 模式适合 mask (保留覆盖率).
            h4, w4 = feat_q4.shape[-2:]
            h8, w8 = feat_q8.shape[-2:]
            # mask 是 [0, 1], 下采后仍 ∈ [0, 1], 直接用作软门控
            self._cur_gate_q4 = F.interpolate(
                text_mask.float(), size=(h4, w4), mode='area'
            ).to(feat_q4.dtype)
            self._cur_gate_q8 = F.interpolate(
                text_mask.float(), size=(h8, w8), mode='area'
            ).to(feat_q8.dtype)
        else:
            self._cur_gate_q4 = None
            self._cur_gate_q8 = None

    def clear_condition(self):
        """清掉缓存, 让下次 UNet forward 走 baseline (除非再次 set_condition)."""
        self._cur_feat_q4 = None
        self._cur_feat_q8 = None
        self._cur_gate_q4 = None
        self._cur_gate_q8 = None
        self._tile_latent_yxyx = None  # 同时清 tile 状态, 避免 baseline forward 误用旧偏移

    def set_tile_slice(self, y0: int, y1: int, x0: int, x1: int):
        """
        ⭐ 推理 tile 时使用. 告诉 hook 当前 UNet forward 是在跑哪个 tile,
        hook 会按 latent 坐标比例从整图 cond_feat 切出对应区域喂给 adapter.

        Args:
            y0, y1, x0, x1: latent 坐标 (单位 = latent 像素, 即 HR/8).
                            必须是 8 的倍数 (因 feat_q8 是 latent/8, 切片要整数).
                            必须是 4 的倍数 (因 feat_q4 是 latent/4).

        典型用法:
            wrapper.set_condition(cond_5ch_full)  # 整图 cond_5ch, 一次算完
            for tile in tiles:
                wrapper.set_tile_slice(y0, y1, x0, x1)
                tile_out = unet(tile_latent, ...)
            wrapper.clear_condition()  # 自动清 tile 状态

        训练时 / 推理整图时, 永远不调这个函数, hook 走整图分支 (cond_feat 与 unet_feat
        spatial 已经一致, _slice_cond_feat 直接返回原 feat).
        """
        assert y0 % 8 == 0 and y1 % 8 == 0 and x0 % 8 == 0 and x1 % 8 == 0, (
            f"tile latent 坐标必须是 8 的倍数, 收到 ({y0},{y1},{x0},{x1})"
        )
        self._tile_latent_yxyx = (y0, y1, x0, x1)

    def clear_tile_slice(self):
        """清 tile 偏移, 下次 hook 走整图分支."""
        self._tile_latent_yxyx = None

    @contextmanager
    def condition_scope(
        self,
        cond_5ch: Optional[torch.Tensor],
        text_mask: Optional[torch.Tensor] = None,
    ):
        """Context manager 版本, 自动 clear. 同时支持 text_mask 用作空间门控."""
        self.set_condition(cond_5ch, text_mask=text_mask)
        try:
            yield
        finally:
            self.clear_condition()

    # ------------------------------------------------------------------
    # 训练用属性: 只暴露我们要训的参数
    # ------------------------------------------------------------------
    def trainable_parameters(self):
        """返回需要被 optimizer 更新的参数 (text_encoder + dual_adapter [+ adapter_mid])."""
        params = list(self.text_encoder.parameters()) + list(self.dual_adapter.parameters())
        if self.adapter_mid is not None:
            params += list(self.adapter_mid.parameters())
        return params

    @property
    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.trainable_parameters() if p.requires_grad)

    def get_strength_dict(self) -> dict:
        d = self.dual_adapter.get_strength_dict()
        if self.adapter_mid is not None:
            d["strength_mid"] = self.adapter_mid.effective_strength
        return d


# ============================================================================
# 自检脚本
# ============================================================================

def _selftest():
    print("=" * 70)
    print("TextAwareUNetWrapper selftest")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- 1) build_cond_5ch ----------
    print("\n[1] build_cond_5ch 工具函数测试")
    B, H_LR = 2, 128
    H_HR = H_LR * 4
    lr = torch.randn(B, 3, H_LR, H_LR, device=device).clamp(-1, 1)
    text_mask = torch.zeros(B, 1, H_HR, H_HR, device=device)
    text_mask[:, :, 100:200, 100:300] = 1.0  # 假设有一段文字
    cond = build_cond_5ch(lr, text_mask, upscale=4)
    print(f"  cond shape: {tuple(cond.shape)}  (期望 (2, 5, {H_HR}, {H_HR}))")
    assert cond.shape == (B, 5, H_HR, H_HR)
    # 各通道范围检查
    print(f"  ch1 hq_gray   range: [{cond[:,0].min().item():.3f}, {cond[:,0].max().item():.3f}]")
    print(f"  ch2 text_mask range: [{cond[:,1].min().item():.3f}, {cond[:,1].max().item():.3f}]")
    print(f"  ch3 density   range: [{cond[:,2].min().item():.3f}, {cond[:,2].max().item():.3f}]")
    print(f"  ch4 sobel_x   range: [{cond[:,3].min().item():.3f}, {cond[:,3].max().item():.3f}]")
    print(f"  ch5 sobel_y   range: [{cond[:,4].min().item():.3f}, {cond[:,4].max().item():.3f}]")
    assert cond[:, 1].min() >= 0 and cond[:, 1].max() <= 1, "text_mask 应在 [0, 1]"
    assert cond[:, 2].min() >= 0 and cond[:, 2].max() <= 1, "density 应在 [0, 1]"
    print("  ✅ build_cond_5ch 通道值范围正确")

    # ---------- 2) Wrapper 实例化 (不需要真实 UNet, 用 mock) ----------
    print("\n[2] Wrapper 实例化 + Hook 行为测试 (mock UNet)")

    # 构造一个最简 mock UNet:
    #   down_blocks[1] 输出 (sample, res_samples), sample = 640×16×16  ← hook A
    #   down_blocks[2] 输出 (sample, res_samples), sample = 1280×8×8   ← hook B
    class MockDownBlock(nn.Module):
        def forward(self, hidden_states, **kwargs):
            return hidden_states, (hidden_states,)

    class MockUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.down_blocks = nn.ModuleList([
                nn.Identity(),       # block 0 (不挂 hook)
                MockDownBlock(),     # block 1 ← hook A (640×16×16)
                MockDownBlock(),     # block 2 ← hook B (1280×8×8)
                nn.Identity(),       # block 3
            ])
            # mid_block 留着, 但 wrapper 不挂它了
            self.mid_block = nn.Identity()

    mock_unet = MockUNet().to(device)
    wrapper = TextAwareUNetWrapper(mock_unet).to(device)
    print(f"  trainable params: {wrapper.num_trainable_params:,} "
          f"({wrapper.num_trainable_params / 1e6:.2f} M)")
    print(f"  hook 数量: {len(wrapper._hook_handles)} (期望 2)")
    assert len(wrapper._hook_handles) == 2

    # ---------- 3) 没设 condition 时, hook 应该返回原值 (baseline) ----------
    print("\n[3] 无 condition 时 hook 返回原值")
    # Hook A: down_blocks[1] 期望 640×16×16
    sample_in = torch.randn(2, 640, 16, 16, device=device)
    out_no_cond = mock_unet.down_blocks[1](sample_in)
    sample_out, _ = out_no_cond
    diff_no_cond = (sample_out - sample_in).abs().max().item()
    print(f"  hook A diff (无 condition): {diff_no_cond:.10f}  (期望 0)")
    assert diff_no_cond < 1e-6

    # Hook B: down_blocks[2] 期望 1280×8×8
    sample_in_b = torch.randn(2, 1280, 8, 8, device=device)
    out_b_no_cond = mock_unet.down_blocks[2](sample_in_b)
    sample_out_b, _ = out_b_no_cond
    diff_b_no_cond = (sample_out_b - sample_in_b).abs().max().item()
    print(f"  hook B diff (无 condition): {diff_b_no_cond:.10f}  (期望 0)")
    assert diff_b_no_cond < 1e-6

    # ---------- 4) 设了 condition 后, 因 zero init 仍应返回原值 ----------
    print("\n[4] 有 condition + zero init: hook 仍应返回原值")
    wrapper.set_condition(cond)
    print(f"  feat_q4 shape: {tuple(wrapper._cur_feat_q4.shape)}")
    print(f"  feat_q8 shape: {tuple(wrapper._cur_feat_q8.shape)}")

    sample_in = torch.randn(2, 640, 16, 16, device=device)
    sample_out, _ = mock_unet.down_blocks[1](sample_in)
    diff = (sample_out - sample_in).abs().max().item()
    print(f"  hook A diff (zero init): {diff:.10f}  (期望 0)")
    assert diff < 1e-6, f"zero init 失败! diff = {diff}"

    sample_in_b = torch.randn(2, 1280, 8, 8, device=device)
    sample_out_b, _ = mock_unet.down_blocks[2](sample_in_b)
    diff_b = (sample_out_b - sample_in_b).abs().max().item()
    print(f"  hook B diff (zero init): {diff_b:.10f}  (期望 0)")
    assert diff_b < 1e-6

    wrapper.clear_condition()

    # ---------- 5) condition_scope context manager ----------
    print("\n[5] condition_scope context manager")
    with wrapper.condition_scope(cond):
        assert wrapper._cur_feat_q4 is not None
        print("  ✅ 进入 scope: condition 已设置")
    assert wrapper._cur_feat_q4 is None
    print("  ✅ 离开 scope: condition 自动清空")

    # ---------- 6) remove_hooks ----------
    print("\n[6] remove_hooks 测试")
    wrapper.remove_hooks()
    print(f"  释放后 hook 数量: {len(wrapper._hook_handles)}")
    assert len(wrapper._hook_handles) == 0
    # 重新设 condition + forward, 应该不会触发 adapter (因 hook 已移除)
    wrapper.set_condition(cond)
    sample_in = torch.randn(2, 640, 16, 16, device=device)
    sample_out, _ = mock_unet.down_blocks[1](sample_in)
    diff = (sample_out - sample_in).abs().max().item()
    print(f"  hook 移除后 diff: {diff:.10f}  (期望 0, 因 hook 不再生效)")
    assert diff < 1e-6
    print("  ✅ remove_hooks 工作正常")

    print("\n" + "=" * 70)
    print("✅ TextAwareUNetWrapper selftest 全部通过!")
    print("(注意: 端到端真实 UNet + 梯度回传测试在 Step 4 配合训练数据做)")
    print("=" * 70)


if __name__ == "__main__":
    _selftest()
