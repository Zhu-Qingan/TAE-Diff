"""
TextAwareAdapter (v6.0)
========================
把 TextConditionEncoder 输出的多分辨率条件特征注入到 UNet 的指定层.

注入位置 (针对 SD 1.5 / CODSR 的 UNet, latent shape 以 HR=512 为例):
    - Hook A:  down_block_2 输出后    UNet feat = [B, 1280, 16, 16]
                                        条件 feat_q4 = [B, 128, 16, 16]
    - Hook B:  mid_block 输出后        UNet feat = [B, 1280, 8, 8]
                                        条件 feat_q8 = [B, 128, 8, 8]

数据流 (单个 Adapter):
    cond_feat  ─Conv1x1─►  cond_proj  [B, UNET_CH, h, w]
    unet_feat  ───────────────────────┐
                                      ├─Concat──►  fuse  [B, 2*UNET_CH, h, w]
                                      │           │
                                      │           ▼
                                      │   Conv3x3(2C → mid) + GN + SiLU
                                      │   Conv3x3(mid → mid) + GN + SiLU
                                      │   Conv1x1(mid → UNET_CH)  ← zero init
                                      │           │
                                      │           ▼
                                      │       delta  [B, UNET_CH, h, w]
                                      │           │
                                      │           ×  strength (标量, init=1e-3)
                                      │           │
    output =  unet_feat  + ───────────┴───────────┘

Zero init 设计 (单重保险, 但梯度可流通):
    1. 最后一层 Conv1x1 weight & bias 全 0  → delta 初始 = 0 (结构层保险)
    2. strength 标量 init = 1e-3           → 不为 0, 让 strength 自己有非零梯度

⚠️ 这是 ControlNet/IP-Adapter 的标准做法, 训练初期的"梯度行为":
    - Step 1: output_proj.weight = bias = 0 → delta_raw ≡ 0
              * output_proj.weight 自身梯度 ≠ 0   (∂L/∂W = strength · ∂L/∂out · fuse_out^T, strength=1e-3≠0)
              * strength 梯度 = 0 (数学必然: ∂L/∂s = ∂L/∂out · delta_raw = 0)
              * cond_proj / fuse_block 上游梯度 = 0  (被 output_proj weight=0 阻断)
              → 第一步只有 output_proj 自己在学
    - 优化器走一步, output_proj.weight 不再为 0, delta_raw 也不再为 0
    - Step 2 起: 全部参数 (含 strength) 都有非零梯度, 正常训练

    所以 strength=1e-3 的真正作用: 让 output_proj.weight 自身的梯度在 Step 1 就稳定非零,
    使训练能从第一步顺利启动. 这跟 ControlNet 原版完全一致.

设计参数 (中间通道 192, 单个 adapter ~5M):
    cond_proj    : 128 × 1280 + 1280            ≈ 0.16 M
    fuse_block_1 : 2560 × 192 × 9 + 192          ≈ 4.42 M
    fuse_block_2 : 192  × 192 × 9 + 192          ≈ 0.33 M
    output_proj  : 192  × 1280 + 1280            ≈ 0.25 M
                                          单 adapter ≈ 5.16 M
                                          双 adapter ≈ 10.3 M
"""

from typing import Optional
import torch
import torch.nn as nn


class TextAwareAdapter(nn.Module):
    """
    单个注入点的 Adapter. 训练时实例化两个: adapter_q4 (1/32) + adapter_q8 (1/64).

    Args:
        cond_ch:    输入条件通道数 (来自 TextConditionEncoder, 默认 128)
        unet_ch:    UNet 中间特征通道数 (down_block_2 / mid_block 默认 1280)
        mid_ch:     fuse block 中间通道 (默认 192, 平衡参数量)
        num_groups: GroupNorm 分组数 (默认 32, 192/32=6 通道每组)
    """

    def __init__(
        self,
        cond_ch: int = 128,
        unet_ch: int = 1280,
        mid_ch: int = 192,
        num_groups: int = 32,
    ):
        super().__init__()
        self.cond_ch = cond_ch
        self.unet_ch = unet_ch
        self.mid_ch = mid_ch

        # 条件通道对齐: cond_ch -> unet_ch (1×1 conv, 类似 lateral connection)
        self.cond_proj = nn.Conv2d(cond_ch, unet_ch, kernel_size=1, bias=True)

        # Fuse block: 把 [unet_feat, cond_proj] concat 后融合
        self.fuse_block = nn.Sequential(
            nn.Conv2d(unet_ch * 2, mid_ch, kernel_size=3, padding=1, bias=True),
            nn.GroupNorm(num_groups, mid_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1, bias=True),
            nn.GroupNorm(num_groups, mid_ch),
            nn.SiLU(inplace=True),
        )

        # 输出投影: mid_ch -> unet_ch (1×1 conv, zero init)
        self.output_proj = nn.Conv2d(mid_ch, unet_ch, kernel_size=1, bias=True)

        # 可学的注入强度标量
        # ⚠️ Sigmoid 约束的注入强度 (v6.1+):
        # 旧版 (v6.0) 用裸 strength 标量, 训练时无上限, 实测会爬到 1.14, 破坏 baseline 色彩平衡.
        # 新版用 sigmoid(strength_raw), 物理上限 1.0, 模型不会"过度自信注入".
        #
        # v6.2 路线 H 更新: strength_raw 起步从 0 改为 -3
        #   - sigmoid(-3) ≈ 0.047, sigmoid'(-3) ≈ 0.045 (梯度健康)
        #   - 解决 v6.1 起步 0.5 时"Adapter 一上来就强行注入未学好的 delta, 导致激进涂抹"的问题
        #   - 与 v6.0 起步 0.001 接近, 但有 sigmoid 上限保护
        #
        # ⚠️ 历史教训 (debug):
        # strength_raw=-7 时 sigmoid'(-7)≈0.0009 死区, 训不动.
        # strength_raw=0 时 sigmoid'(0)=0.25 最大, 但 effective=0.5 起步过头,
        # delta 还未学好就被乘进 unet_feat, 学到激进偏移方向.
        # strength_raw=-3 折中: 起步极弱 (0.047) 不污染 baseline, 梯度 (0.045) 仍能正常更新.
        #
        # ⚠️ 兼容性: 旧 ckpt 的 "strength" 字段无法直接加载到新代码 (参数名不同), 这是有意的:
        #   旧 ckpt 已经训坏, 不应该被加载.
        self.strength_raw = nn.Parameter(torch.full((1,), -3.0))

        self._init_weights()

    def _init_weights(self):
        """Kaiming for conv (除了最后的 zero-init), zero for output_proj & strength."""
        # cond_proj & fuse_block 用 kaiming
        nn.init.kaiming_normal_(self.cond_proj.weight, nonlinearity="linear")
        nn.init.zeros_(self.cond_proj.bias)

        for m in self.fuse_block.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GroupNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # output_proj zero init (结构层保险, 保证 delta 初始为 0)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

        # strength 已经在 __init__ 里 init 为 1e-3 (非 0, 保证梯度可流)

    def forward(
        self,
        unet_feat: torch.Tensor,
        cond_feat: torch.Tensor,
        spatial_gate: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            unet_feat:    [B, unet_ch, h, w]
            cond_feat:    [B, cond_ch, h, w]   (h, w 必须与 unet_feat 一致)
            spatial_gate: [B, 1, h, w] or None
                          v6.2 路线 H: text_mask 下采到 (h, w) 后的软门控 ∈ [0, 1].
                          作用: 把 delta 限制在文字区域, 防止 Adapter 影响非文字背景.
                          None 时退化为旧行为 (不门控), 保证旧 inference 脚本兼容.

        Returns:
            output:    [B, unet_ch, h, w]   = unet_feat + sigmoid(s) * delta * gate

        Note:
            - 训练初期 (zero init + strength_raw=-3): delta ≡ 0, 仍严格 ours == baseline
            - gate 不影响"零注入"特性 (0 * anything = 0)
            - 即使 cond_feat 不匹配尺寸, 也会主动 raise (不静默 resize)
        """
        if unet_feat.shape[-2:] != cond_feat.shape[-2:]:
            raise ValueError(
                f"TextAwareAdapter spatial size mismatch: "
                f"unet_feat={tuple(unet_feat.shape[-2:])} vs "
                f"cond_feat={tuple(cond_feat.shape[-2:])}. "
                f"请检查 TextConditionEncoder 输出与 UNet 注入点是否对齐."
            )

        # 1) 条件通道对齐
        cond_proj = self.cond_proj(cond_feat)  # [B, unet_ch, h, w]

        # 2) Concat + 融合
        fused = torch.cat([unet_feat, cond_proj], dim=1)  # [B, 2*unet_ch, h, w]
        delta = self.fuse_block(fused)                     # [B, mid_ch, h, w]
        delta = self.output_proj(delta)                    # [B, unet_ch, h, w], zero-init

        # 3) ⭐ v6.2 路线 H: 空间门控
        #    delta * gate: 在 mask=1 区域全注入, mask=0 区域不注入, mask∈(0,1) 软过渡
        #    broadcast: [B, 1, h, w] × [B, unet_ch, h, w] → [B, unet_ch, h, w]
        if spatial_gate is not None:
            if spatial_gate.shape[-2:] != delta.shape[-2:]:
                raise ValueError(
                    f"spatial_gate spatial size mismatch: "
                    f"gate={tuple(spatial_gate.shape[-2:])} vs delta={tuple(delta.shape[-2:])}"
                )
            delta = delta * spatial_gate.to(delta.dtype)

        # 4) 残差加 + sigmoid 约束的 strength scaling
        #    effective_strength = sigmoid(strength_raw) ∈ (0, 1) 永远不会失控
        output = unet_feat + torch.sigmoid(self.strength_raw) * delta

        return output

    @property
    def effective_strength(self) -> float:
        """返回当前 sigmoid 约束后的实际 strength (用于日志)."""
        return float(torch.sigmoid(self.strength_raw).detach().cpu().item())

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class DualAdapter(nn.Module):
    """
    把两个注入点 (down_blocks[1] 后 + down_blocks[2] 后) 的 Adapter 打包成一个模块,
    方便统一保存/加载/管理.

    SD1.5 / SD2.1-base 共用结构 (latent 64×64 输入时):
        down_blocks[0]: 320 ch,  32×32 (latent 1/2)
        down_blocks[1]: 640 ch,  16×16 (latent 1/4)  ⭐ hook A — 字形/连字/衬线
        down_blocks[2]: 1280 ch, 8×8   (latent 1/8)  ⭐ hook B — 语义/布局
        down_blocks[3]: 1280 ch, 8×8   (no downsample)
        mid_block:      1280 ch, 8×8                  (跳过, 不破坏全局语义)

    使用流程 (在 train.py / inference.py 里):
        text_encoder = TextConditionEncoder()
        dual_adapter = DualAdapter()  # adapter_q4 用 640ch, adapter_q8 用 1280ch

        # 前向 (UNet hook 里调用)
        feat_q4, feat_q8 = text_encoder(cond_5ch)
        # ...UNet down_blocks[1] forward 后...
        x = dual_adapter.adapter_q4(x, feat_q4)  # x: [B, 640, 16, 16]
        # ...UNet down_blocks[2] forward 后...
        x = dual_adapter.adapter_q8(x, feat_q8)  # x: [B, 1280, 8, 8]
    """

    def __init__(
        self,
        cond_ch: int = 128,
        unet_ch_q4: int = 640,    # down_blocks[1] 输出通道
        unet_ch_q8: int = 1280,   # down_blocks[2] 输出通道
        mid_ch: int = 192,
    ):
        super().__init__()
        self.adapter_q4 = TextAwareAdapter(cond_ch, unet_ch_q4, mid_ch)
        self.adapter_q8 = TextAwareAdapter(cond_ch, unet_ch_q8, mid_ch)

    def forward(self, *args, **kwargs):
        raise RuntimeError(
            "DualAdapter 不直接调用 forward. "
            "请显式使用 self.adapter_q4(...) 和 self.adapter_q8(...)."
        )

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def get_strength_dict(self) -> dict:
        """便于训练日志监控两个 adapter 的 effective strength (sigmoid 后)."""
        return {
            "strength_q4": self.adapter_q4.effective_strength,
            "strength_q8": self.adapter_q8.effective_strength,
        }

    def load_state_dict(self, state_dict, strict=True):
        """重写 load_state_dict, 兼容检测旧 ckpt (含 strength 字段) 并给出明确报错."""
        # 检测旧 ckpt 的字段名
        legacy_keys = [k for k in state_dict.keys() if k.endswith(".strength")]
        if legacy_keys:
            raise RuntimeError(
                f"⚠️ 检测到旧版 ckpt (含 'strength' 字段, 新版改为 'strength_raw' + sigmoid 约束):\n"
                f"  旧字段: {legacy_keys}\n"
                f"  原因: 旧版 strength 无上限, 实测训练时会涨到 1.14 导致色偏崩塌.\n"
                f"  解决: 不要加载旧 ckpt, 从头训练新版 (sigmoid 约束).\n"
                f"  如果一定要加载, 请先手动转换:\n"
                f"    旧 strength=s  →  新 strength_raw = logit(s) = log(s / (1-s))\n"
            )
        return super().load_state_dict(state_dict, strict=strict)


# ============================================================================
# 自检脚本
# ============================================================================

if __name__ == "__main__":
    import time

    print("=" * 60)
    print("TextAwareAdapter selftest")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- 1) 单个 Adapter 基础测试 ----------
    print("\n[1] 单个 TextAwareAdapter 基础测试")
    adapter = TextAwareAdapter(cond_ch=128, unet_ch=1280, mid_ch=192).to(device)
    print(f"  参数量: {adapter.num_params:,}  ({adapter.num_params / 1e6:.2f} M)")

    # ---------- 2) 多分辨率 shape 测试 ----------
    print("\n[2] 多分辨率 shape 测试")
    print(f"  {'HR':>6} | {'插入点':>8} | {'unet_feat':>20} | {'cond_feat':>20} | {'output':>20}")
    print("  " + "-" * 88)

    test_cases = [
        # (HR, 插入点, unet_h, cond_h)  注: unet_h = HR/32 (q4), HR/64 (q8)
        (512, "q4", 16, 16),
        (512, "q8", 8, 8),
        (768, "q4", 24, 24),
        (768, "q8", 12, 12),
        (1024, "q4", 32, 32),
        (1024, "q8", 16, 16),
    ]

    for hr, hook, h, _ in test_cases:
        unet_feat = torch.randn(2, 1280, h, h, device=device)
        cond_feat = torch.randn(2, 128, h, h, device=device)
        with torch.no_grad():
            out = adapter(unet_feat, cond_feat)
        print(
            f"  {hr:>6} | {hook:>8} | {str(tuple(unet_feat.shape)):>20} | "
            f"{str(tuple(cond_feat.shape)):>20} | {str(tuple(out.shape)):>20}"
        )

    # ---------- 3) Zero init (结构层) 测试 ----------
    print("\n[3] Zero init 结构层测试 (output_proj=0, 所以 delta=0, 与 strength 值无关)")
    adapter.eval()
    unet_feat = torch.randn(2, 1280, 16, 16, device=device)
    cond_feat = torch.randn(2, 128, 16, 16, device=device)
    with torch.no_grad():
        out = adapter(unet_feat, cond_feat)
        diff = (out - unet_feat).abs().max().item()
    print(f"  output - unet_feat 最大绝对差: {diff:.10f}")
    assert diff < 1e-6, f"Zero init 失败! diff = {diff}"
    print(f"  ✅ Zero init 验证通过 (output ≡ unet_feat, 因 delta=0)")
    print(f"  初始 strength_raw: {adapter.strength_raw.item():.4f}  "
          f"→ effective_strength: {adapter.effective_strength:.6f}  (sigmoid(0)=0.5)")

    # ---------- 4) 梯度回传 + 严格非零检查 (含 ControlNet 式 warmup) ----------
    print("\n[4] 梯度回传测试")
    print("  说明: 与 ControlNet 同理, 第一步反传只能更新 output_proj 自己")
    print("        (因 output_proj.weight=0 阻断上游梯度).")
    print("        必须做 1 步优化器 step 后, 上游参数才能在第二步拿到梯度.")

    adapter.train()
    optimizer = torch.optim.Adam(adapter.parameters(), lr=1e-3)

    # ===== 第一步: 只检查 output_proj 拿到梯度 (其它全为 0 是数学必然) =====
    print("\n  --- Step 1 (warmup): 仅 output_proj.weight 应有梯度 ---")
    print("  数学解释: output_proj.weight=bias=0 → delta_raw ≡ 0")
    print("    ∂L/∂strength_raw = sigmoid'(s) · ∂L/∂out · delta_raw = 0  (delta_raw=0)")
    print("    ∂L/∂上游 = output_proj.weight^T · ∂L/∂out = 0  (weight=0)")
    print("    ∂L/∂output_proj.weight = sigmoid(s) · ∂L/∂out · fuse_out^T ≠ 0")
    print("      (因 sigmoid(0)=0.5≠0)")
    optimizer.zero_grad()
    unet_feat = torch.randn(2, 1280, 16, 16, device=device, requires_grad=True)
    cond_feat = torch.randn(2, 128, 16, 16, device=device, requires_grad=True)
    out = adapter(unet_feat, cond_feat)
    target = torch.randn_like(out)
    loss1 = ((out - target) ** 2).mean()
    loss1.backward()

    g_op = adapter.output_proj.weight.grad.norm().item()
    g_st = adapter.strength_raw.grad.norm().item()
    g_cp = adapter.cond_proj.weight.grad.norm().item()
    g_fb = adapter.fuse_block[0].weight.grad.norm().item()
    print(f"\n    output_proj.weight  grad = {g_op:.6e}  (期望 > 0)")
    print(f"    strength            grad = {g_st:.6e}  (期望 = 0, 数学必然)")
    print(f"    cond_proj.weight    grad = {g_cp:.6e}  (期望 = 0, 被 output_proj 阻断)")
    print(f"    fuse_block[0]       grad = {g_fb:.6e}  (期望 = 0, 被 output_proj 阻断)")
    # 只严格要求 output_proj 有梯度 — 这是训练能启动的唯一条件
    assert g_op > 1e-12, f"❌ output_proj 梯度 = 0! 这才是真死锁, 训练无法启动."
    assert g_st < 1e-10, f"strength_raw 不应该有梯度 (delta_raw=0), 但实际 = {g_st}"
    assert g_cp < 1e-10, f"cond_proj 不应该有梯度, 但实际 = {g_cp}"
    print("    ✅ Step 1 行为符合 ControlNet 标准: output_proj 自学, 其余等待")

    # 走一步优化器, 让 output_proj.weight 不再为 0
    optimizer.step()
    op_max_after = adapter.output_proj.weight.abs().max().item()
    print(f"    Step 1 后 output_proj.weight 最大绝对值: {op_max_after:.6e} (应 > 0)")
    assert op_max_after > 1e-8, "Step 1 后 output_proj 还是 0, 优化失败"

    # ===== 第二步: 现在上游梯度应该全部活了 =====
    print("\n  --- Step 2 (正式): 全部参数应有非零梯度 ---")
    optimizer.zero_grad()
    unet_feat = torch.randn(2, 1280, 16, 16, device=device, requires_grad=True)
    cond_feat = torch.randn(2, 128, 16, 16, device=device, requires_grad=True)
    out = adapter(unet_feat, cond_feat)
    target = torch.randn_like(out)
    loss2 = ((out - target) ** 2).mean()
    loss2.backward()

    grad_checks = {
        "cond_proj.weight": adapter.cond_proj.weight.grad,
        "fuse_block[0].weight": adapter.fuse_block[0].weight.grad,
        "fuse_block[3].weight": adapter.fuse_block[3].weight.grad,
        "output_proj.weight": adapter.output_proj.weight.grad,
        "strength_raw": adapter.strength_raw.grad,
    }
    for name, g in grad_checks.items():
        assert g is not None, f"{name} 没有梯度!"
        assert torch.isfinite(g).all(), f"{name} 梯度有 NaN/Inf!"
        gnorm = g.norm().item()
        assert gnorm > 1e-12, (
            f"❌ {name} grad norm = {gnorm:.2e} 接近 0! Step 2 应该全部活."
        )
        print(f"    ✅ {name:<25} grad norm = {gnorm:.6e}")
    print(f"\n  loss1 = {loss1.item():.4f}, loss2 = {loss2.item():.4f}")

    # ---------- 5) 尺寸不匹配应该报错 ----------
    print("\n[5] 尺寸不匹配应该报错")
    unet_feat = torch.randn(2, 1280, 16, 16, device=device)
    cond_feat = torch.randn(2, 128, 8, 8, device=device)  # 故意错
    try:
        _ = adapter(unet_feat, cond_feat)
        raise AssertionError("应该抛 ValueError 但是没有抛!")
    except ValueError as e:
        print(f"  ✅ 正确捕获 ValueError: {str(e)[:80]}...")

    # ---------- 6) DualAdapter 测试 ----------
    print("\n[6] DualAdapter 测试 (adapter_q4: 640ch×16, adapter_q8: 1280ch×8)")
    dual = DualAdapter(cond_ch=128, unet_ch_q4=640, unet_ch_q8=1280, mid_ch=192).to(device)
    print(f"  双 adapter 总参数量: {dual.num_params:,}  ({dual.num_params / 1e6:.2f} M)")
    print(f"  初始 strength: {dual.get_strength_dict()}")

    # 模拟一次完整流程 (HR=512 时的 latent 形状)
    feat_q4 = torch.randn(2, 128, 16, 16, device=device)
    feat_q8 = torch.randn(2, 128, 8, 8, device=device)
    unet_x_q4 = torch.randn(2, 640, 16, 16, device=device)
    unet_x_q8 = torch.randn(2, 1280, 8, 8, device=device)

    with torch.no_grad():
        out_q4 = dual.adapter_q4(unet_x_q4, feat_q4)
        out_q8 = dual.adapter_q8(unet_x_q8, feat_q8)
    print(f"  out_q4 shape: {tuple(out_q4.shape)}, 零初始 diff = {(out_q4 - unet_x_q4).abs().max().item():.2e}")
    print(f"  out_q8 shape: {tuple(out_q8.shape)}, 零初始 diff = {(out_q8 - unet_x_q8).abs().max().item():.2e}")

    # ---------- 7) GPU 性能 (HR=1024, latent 128×128) ----------
    if device.type == "cuda":
        print("\n[7] GPU 性能 (HR=1024, batch=4, fp32)")
        dual = DualAdapter().to(device)

        # HR=1024 时: latent 128×128, hook A → 640ch × 32×32, hook B → 1280ch × 16×16
        feat_q4 = torch.randn(4, 128, 32, 32, device=device)
        feat_q8 = torch.randn(4, 128, 16, 16, device=device)
        unet_x_q4 = torch.randn(4, 640, 32, 32, device=device)
        unet_x_q8 = torch.randn(4, 1280, 16, 16, device=device)

        # warmup
        for _ in range(3):
            _ = dual.adapter_q4(unet_x_q4, feat_q4)
            _ = dual.adapter_q8(unet_x_q8, feat_q8)
        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        n_iter = 20
        for _ in range(n_iter):
            o1 = dual.adapter_q4(unet_x_q4, feat_q4)
            o2 = dual.adapter_q8(unet_x_q8, feat_q8)
        torch.cuda.synchronize()
        t1 = time.time()
        peak_mem = torch.cuda.max_memory_allocated() / 1024**3

        print(f"  双 adapter 单次 forward: {(t1 - t0) * 1000 / n_iter:.2f} ms")
        print(f"  峰值显存:                {peak_mem:.2f} GB")

    print("\n" + "=" * 60)
    print("✅ TextAwareAdapter selftest 全部通过!")
    print("=" * 60)
