"""
TAE-DiffSR v6.0 训练脚本
===========================
基于 CODSR baseline + TextAwareUNetWrapper (Adapter 注入) 的单步扩散文本超分.

设计要点 (与 v5.1 的差别):
  - 替换 v5.1 的 ControlNet 旁路为 v6 的 TextAwareUNetWrapper (forward hook 注入)
  - Hook 位置: down_blocks[1] (字形) + down_blocks[2] (语义), mid_block 默认不挂
  - 训练参数从 v5.1 的 30M+ 降到 10M (TextEncoder 2.6M + DualAdapter 7.7M)
  - 训练初期严格保证 ours = baseline (zero init via Adapter.output_proj=0)

Loss 组成:
  loss = MSE_latent + λ_pixel * L1_pixel + (mask_weight - 1) * MSE_latent * mask
  其中:
    - MSE_latent: pred_noise vs noise (latent 空间, 与 baseline 一致)
    - L1_pixel:   decode(pred_clean) vs hr (像素空间, λ_pixel 可设 0 退化)
    - mask 加权: text 区域 loss 乘 mask_weight (默认 3.0)

数据流 (与 v5.1 一致, 不动 CODSR baseline 路径):
  hq_bicubic (HR ×4 上采) → VAE encode → latent
  hq_bicubic → PixelUnshuffle → SFT → (scale, shift) → modulation_params
  hq_bicubic → Sobel → mix_weight → noise = randn * mix_weight
  latent_noisy = scheduler.add_noise(latent, noise, t=100)
  lr + text_mask_hr → build_cond_5ch → cond
  wrapper.set_condition(cond)
  pred_noise = unet(latent_noisy, t, prompt_embeds, modulation_params).sample
  pred_clean = (latent_noisy - coeff * pred_noise) / 1  [简化形式]
  loss = ...

学习率:
  lr=1e-4 + linear warmup 1000 step → 恒定 (不加 cosine 衰减, 简单点)

用法:
  python train_v6.py \
      --pretrained_model_name_or_path /workspace/.../stable-diffusion-2-1-base \
      --codsr_path preset/models/codsr.pkl \
      --hr_dir /workspace/.../RealCE_13mm/HR \
      --lr_dir /workspace/.../RealCE_13mm/LR \
      --anno_dir /workspace/.../RealCE_13mm/anno_txt \
      --output_dir output/v6.0_phase1 \
      --batch_size 4 --num_epochs 50 \
      --learning_rate 1e-4 --warmup_steps 1000 \
      --text_loss_weight 3.0 --pixel_lambda 0.1
"""

import argparse
import os
import sys
import time
import json
import logging
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler

from transformers import AutoTokenizer, CLIPTextModel
from diffusers import DDPMScheduler
from peft import LoraConfig
from diffusers.utils.peft_utils import set_weights_and_activate_adapters

from models.autoencoder_kl import AutoencoderKL
from models.unet_2d_condition import UNet2DConditionModel
from models.text_aware_unet_wrapper import TextAwareUNetWrapper, build_cond_5ch
from codsr import SFTLayer, ChannelwiseSobel

from dataset import TAEDiffSRDataset, build_concat_dataset


logger = logging.getLogger(__name__)





# ============================================================================
# Argparse
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser()

    # --- 模型路径 ---
    p.add_argument("--pretrained_model_name_or_path", type=str,
                   default="/workspace/team_code_codec_new/diffproj/preset/stable-diffusion-2-1-base")
    p.add_argument("--codsr_path", type=str, default="preset/models/codsr.pkl")

    # --- 数据 (单数据集模式) ---
    p.add_argument("--hr_dir", type=str, default=None,
                   help="单数据集模式: HR 目录")
    p.add_argument("--lr_dir", type=str, default=None,
                   help="单数据集模式: LR 目录")
    p.add_argument("--anno_dir", type=str, default=None,
                   help="单数据集模式: anno txt 目录 (可选)")
    p.add_argument("--data_config", type=str, default=None,
                   help="多数据集模式: JSON 配置文件 (优先级高于 hr_dir/lr_dir)")

    # --- 训练超参 ---
    p.add_argument("--patch_size", type=int, default=512)
    p.add_argument("--scale_factor", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_epochs", type=int, default=50)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int, default=1000,
                   help="Linear warmup 步数 (0 = 不预热)")
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--mixed_precision", type=str, default="fp16",
                   choices=["no", "fp16"])

    # --- Loss 权重 ---
    p.add_argument("--text_loss_weight", type=float, default=3.0,
                   help="text 区 loss 加权系数 (1.0 = 不加权)")
    p.add_argument("--pixel_lambda", type=float, default=0.1,
                   help="像素空间 L1 loss 权重 (0 = 关闭)")

    # --- v6 wrapper 配置 ---
    p.add_argument("--hook_mid_block", action="store_true",
                   help="是否在 mid_block 也挂 adapter (默认 False)")
    p.add_argument("--gradient_checkpointing", action="store_true",
                   help="开启 UNet gradient checkpointing (显存 ~50%↓, 速度 ~30%↓). "
                        "v6 hook 注入模式下强烈推荐, 因为 adapter 输出被 UNet 后半部分使用, "
                        "默认会保留大量中间激活.")

    # --- 输出 / 日志 ---
    p.add_argument("--output_dir", type=str, default="output/v6.0_phase1")
    p.add_argument("--log_every_n_steps", type=int, default=20)
    p.add_argument("--save_every_n_epochs", type=int, default=5)
    p.add_argument("--visualize_every_n_epochs", type=int, default=2,
                   help="保存 [LR|Ours|HR] 三联对比图的频率 (0 = 关闭)")
    p.add_argument("--num_vis_samples", type=int, default=4)

    # --- Prompt ---
    p.add_argument("--default_prompt", type=str,
                   default="text, high quality, sharp")

    # --- 续训 ---
    p.add_argument("--resume_from", type=str, default=None,
                   help="checkpoint 路径, 续训用")

    return p.parse_args()


# ============================================================================
# CODSR baseline 加载 (与 step4 一致, 抽出来复用)
# ============================================================================

def load_codsr_baseline(args, device, dtype):
    """加载 + LoRA merge + freeze 全部 baseline 组件."""
    logger.info(f"[load] SD base: {args.pretrained_model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder"
    )
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    noise_scheduler.set_timesteps(1, device=device)
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae"
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet"
    )

    sft = SFTLayer(192, 320)

    logger.info(f"[load] CODSR ckpt: {args.codsr_path}")
    model_ckp = torch.load(args.codsr_path, map_location=device)

    # SFT
    _sft = sft.state_dict()
    for k in model_ckp["state_dict_sft"]:
        _sft[k] = model_ckp["state_dict_sft"][k]
    sft.load_state_dict(_sft)

    # VAE LoRA
    vae_lora = LoraConfig(
        r=model_ckp["rank_vae"], init_lora_weights="gaussian",
        target_modules=model_ckp["vae_lora_encoder_modules"]
    )
    vae.add_adapter(vae_lora, adapter_name="default_encoder")
    for n, p in vae.named_parameters():
        if "lora" in n:
            p.data.copy_(model_ckp["state_dict_vae"][n])
    vae.set_adapter(['default_encoder'])

    # UNet default LoRA
    for cfg_name, mod_key in [
        ("default_encoder", "unet_lora_encoder_modules_default"),
        ("default_decoder", "unet_lora_decoder_modules_default"),
        ("default_others",  "unet_lora_others_modules_default"),
    ]:
        cfg = LoraConfig(r=model_ckp["rank_unet"], init_lora_weights="gaussian",
                         target_modules=model_ckp[mod_key])
        unet.add_adapter(cfg, adapter_name=cfg_name)
    for n, p in unet.named_parameters():
        if "lora" in n:
            p.data.copy_(model_ckp["state_dict_unet"][n])
    set_weights_and_activate_adapters(
        unet, ["default_encoder", "default_decoder", "default_others"], [1.0, 1.0, 1.0]
    )
    unet.merge_and_unload()

    # Semantic LoRA
    for cfg_name, mod_key in [
        ("default_encoder_alignment", "unet_lora_encoder_modules_sam"),
        ("default_decoder_alignment", "unet_lora_decoder_modules_sam"),
        ("default_others_alignment",  "unet_lora_others_modules_sam"),
    ]:
        cfg = LoraConfig(r=model_ckp["rank_unet"], init_lora_weights="gaussian",
                         target_modules=model_ckp[mod_key])
        unet.add_adapter(cfg, adapter_name=cfg_name)
    for n, p in unet.named_parameters():
        if "lora" in n:
            p.data.copy_(model_ckp["state_dict_unet"][n])
    set_weights_and_activate_adapters(
        unet,
        ["default_encoder_alignment", "default_decoder_alignment", "default_others_alignment"],
        [1.0, 1.0, 1.0]
    )
    unet.merge_and_unload()

    text_encoder = text_encoder.to(device, dtype=dtype)
    vae = vae.to(device, dtype=dtype)
    unet = unet.to(device, dtype=dtype)
    sft = sft.to(device, dtype=dtype)

    # Freeze 所有 baseline
    for m in [text_encoder, vae, unet, sft]:
        for p in m.parameters():
            p.requires_grad = False
        m.eval()

    timesteps = torch.tensor([100], device=device).long()
    noise_scheduler.alphas_cumprod = noise_scheduler.alphas_cumprod.to(device)

    return {
        "tokenizer": tokenizer, "text_encoder": text_encoder,
        "noise_scheduler": noise_scheduler, "vae": vae, "unet": unet,
        "sft": sft, "timesteps": timesteps,
    }


# ============================================================================
# CODSR baseline 工具函数 (与 codsr.py / train.py 对齐)
# ============================================================================

def rgb_to_gray(x):
    return (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])


def compute_spatial_weight(lq, target_hw, sobel_layer):
    """复现 CODSR GraygradToWeight_Patchwise_Sobel."""
    gray = rgb_to_gray(lq * 0.5 + 0.5)
    gradient = sobel_layer(gray)
    B, C, h, w = gradient.shape
    block_size = 16
    pad_h = (-h) % block_size
    pad_w = (-w) % block_size
    if pad_h or pad_w:
        gradient = F.pad(gradient, (0, pad_w, 0, pad_h), mode='replicate')
    patch_avg = F.avg_pool2d(gradient, kernel_size=block_size, stride=block_size)
    result = torch.zeros_like(patch_avg)
    low = patch_avg <= 0.15
    mid = (patch_avg > 0.15) & (patch_avg <= 0.25)
    high = patch_avg > 0.25
    result[low] = 0.3
    result[mid] = 7.0 * (patch_avg[mid] - 0.15) + 0.3
    result[high] = 1.0
    weight = result.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)
    H8, W8 = target_hw
    return weight[:, :, :H8, :W8].to(device=lq.device, dtype=lq.dtype)


def eps_to_mu_coeff(scheduler, sample, t):
    alphas_cumprod = scheduler.alphas_cumprod.to(device=sample.device, dtype=sample.dtype)
    alpha = alphas_cumprod[t]
    while len(alpha.shape) < len(sample.shape):
        alpha = alpha.unsqueeze(-1)
    beta = 1 - alpha
    return beta ** 0.5 / alpha ** 0.5


def encode_prompt(prompts, tokenizer, text_encoder, device, dtype):
    ids = tokenizer(prompts, max_length=tokenizer.model_max_length,
                    padding="max_length", truncation=True,
                    return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        emb = text_encoder(ids)[0]
    return emb.to(dtype=dtype)


# ============================================================================
# 学习率调度: linear warmup → 恒定
# ============================================================================

def get_lr_factor(step, warmup_steps):
    if warmup_steps <= 0:
        return 1.0
    if step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    return 1.0


# ============================================================================
# 数据集构建
# ============================================================================

def build_dataset(args):
    """优先用 data_config (多数据集), 否则用 hr_dir/lr_dir/anno_dir (单数据集)."""
    common_kwargs = dict(
        patch_size=args.patch_size,
        scale_factor=args.scale_factor,
        phase="phase2",
        augment=True,
    )
    if args.data_config is not None:
        with open(args.data_config) as f:
            configs = json.load(f)
        return build_concat_dataset(configs, **common_kwargs)
    else:
        assert args.hr_dir and args.lr_dir, "需指定 --hr_dir 和 --lr_dir, 或 --data_config"
        return TAEDiffSRDataset(
            hr_dir=args.hr_dir, lr_dir=args.lr_dir,
            anno_dir=args.anno_dir,
            **common_kwargs,
        )


# ============================================================================
# 可视化 (保存 [LR|Ours|HR] 三联图)
# ============================================================================

@torch.no_grad()
def save_visualization(wrapper, vae, sft, baseline, sobel_layer, batch, args, save_path, device, dtype):
    """生成 [LR_bicubic | Ours | HR] 三联对比图."""
    import torchvision.utils as vutils

    n = min(args.num_vis_samples, batch["hq_bicubic"].shape[0])
    hq_bicubic = batch["hq_bicubic"][:n].to(device, dtype=dtype)
    lr = batch["lr"][:n].to(device, dtype=dtype)
    text_mask_hr = batch["text_mask_hr"][:n].to(device, dtype=dtype)
    hr = batch["hr"][:n].to(device, dtype=dtype)

    tokenizer = baseline["tokenizer"]
    text_encoder = baseline["text_encoder"]
    unet = baseline["unet"]
    noise_scheduler = baseline["noise_scheduler"]
    timesteps = baseline["timesteps"]

    # 完整推理一遍 (与训练循环一致, 用 autocast 兼容 fp32 wrapper 参数 + fp16 baseline)
    with autocast("cuda", enabled=(args.mixed_precision == "fp16"), dtype=torch.float16):
        encoded_control = vae.encode(hq_bicubic).latent_dist.sample() * vae.config.scaling_factor
        unshuffle = nn.PixelUnshuffle(downscale_factor=8)
        sft_cond = unshuffle(hq_bicubic)
        scale, shift = sft(sft_cond)
        H8, W8 = encoded_control.shape[-2], encoded_control.shape[-1]
        mix_weight = compute_spatial_weight(hq_bicubic, (H8, W8), sobel_layer)
        g = torch.Generator(device=device).manual_seed(42)
        noise = torch.randn(encoded_control.shape, generator=g, device=device, dtype=dtype)
        noise_a = noise * mix_weight
        lq_latent = noise_scheduler.add_noise(encoded_control, noise_a, timesteps)
        coeff = eps_to_mu_coeff(noise_scheduler, lq_latent, timesteps)
        unet_params = (scale / coeff, shift / coeff)
        prompt_embeds = encode_prompt([args.default_prompt] * n, tokenizer, text_encoder, device, dtype)

        cond_5ch = build_cond_5ch(lr, text_mask_hr, upscale=args.scale_factor)
        # v6.2 路线 H: 传 text_mask, wrapper 内自动下采为空间门控
        wrapper.set_condition(cond_5ch, text_mask=text_mask_hr)
        pred_noise = unet(lq_latent, timesteps, encoder_hidden_states=prompt_embeds,
                          modulation_params=unet_params).sample
        wrapper.clear_condition()

        x_denoised = encoded_control + coeff * (noise_a - pred_noise)
        ours = vae.decode(x_denoised.to(dtype) / vae.config.scaling_factor).sample.clamp(-1, 1)

    # 拼接 [LR_bicubic | Ours | HR]  (转 fp32 + cpu, 避免 save_image 偶发 fp16 问题)
    hq_bicubic_vis = hq_bicubic.float().cpu()
    ours_vis = ours.float().cpu()
    hr_vis = hr.float().cpu()
    grid_rows = []
    for i in range(n):
        row = torch.cat([
            (hq_bicubic_vis[i] + 1) / 2,
            (ours_vis[i] + 1) / 2,
            (hr_vis[i] + 1) / 2,
        ], dim=-1)  # 横向拼接
        grid_rows.append(row)
    grid = torch.stack(grid_rows, dim=0)  # [n, 3, H, W*3]

    # 保存为 JPG 而非 PNG, 文件大小从 ~5MB 降到 ~1MB.
    # save_image 默认 PNG (无损), 改用 PIL 显式 JPG quality=85.
    import torchvision.utils as vutils
    from PIL import Image
    import numpy as _np

    # vutils.make_grid 拼成单张大图, [3, H_total, W_total]
    big = vutils.make_grid(grid, nrow=1, padding=2, normalize=False).clamp(0, 1)
    big_np = (big.permute(1, 2, 0).numpy() * 255).astype(_np.uint8)
    # 若保存路径以 .png 结尾, 自动改为 .jpg
    if save_path.endswith(".png"):
        save_path = save_path[:-4] + ".jpg"
    Image.fromarray(big_np).save(save_path, format="JPEG", quality=85, optimize=True)


# ============================================================================
# 主训练循环
# ============================================================================

def main():
    args = parse_args()
    device = torch.device("cuda")
    dtype = torch.float16 if args.mixed_precision == "fp16" else torch.float32

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "vis"), exist_ok=True)

    # 配置 logger: 输出到 stdout (会被 nohup 重定向到 train.log)
    # 格式与 v5.1 一致, 带时间戳 + 模块名, 易于扫读
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        stream=sys.stdout,
        force=True,  # 覆盖外部库的 logger 配置 (如 diffusers/transformers)
    )

    # 保存 args
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    logger.info("=" * 70)
    logger.info("TAE-DiffSR v6.0 Training")
    logger.info("=" * 70)
    logger.info(f"  output_dir: {args.output_dir}")
    logger.info(f"  dtype: {dtype}, batch: {args.batch_size}, patch: {args.patch_size}")
    logger.info(f"  lr: {args.learning_rate}, warmup: {args.warmup_steps}")
    logger.info(f"  text_loss_weight: {args.text_loss_weight}, pixel_lambda: {args.pixel_lambda}")
    logger.info(f"  hook_mid_block: {args.hook_mid_block}")

    # ---------- 1) 加载 baseline + wrapper ----------
    baseline = load_codsr_baseline(args, device, dtype)
    unet = baseline["unet"]
    vae = baseline["vae"]
    sft = baseline["sft"]
    tokenizer = baseline["tokenizer"]
    text_encoder = baseline["text_encoder"]
    noise_scheduler = baseline["noise_scheduler"]
    timesteps = baseline["timesteps"]

    wrapper = TextAwareUNetWrapper(
        unet,
        cond_in_channels=5,
        hook_mid_block=args.hook_mid_block,
    )
    # ⚠️ 关键: 训练参数 (text_encoder + dual_adapter) 必须保持 fp32!
    # 否则 GradScaler.unscale_ 会抛 "Attempting to unscale FP16 gradients".
    # fp16 训练的标准做法: 参数 fp32 + autocast 下激活/前向 fp16.
    wrapper.text_encoder.to(device=device, dtype=torch.float32)
    wrapper.dual_adapter.to(device=device, dtype=torch.float32)
    if wrapper.adapter_mid is not None:
        wrapper.adapter_mid.to(device=device, dtype=torch.float32)
    # unet 已经在 load_codsr_baseline 里设过 dtype, 这里不动

    # ⚠️ 显存优化: UNet gradient checkpointing
    # v6 hook 注入模式下, adapter 的输出被 UNet 后半部分使用,
    # 反传 ∂L/∂adapter 必须穿过 UNet 后半部分, 这些层的激活默认全部保留 → 显存爆.
    # 开启 gradient checkpointing 后, UNet 的中间激活反传时重算, 显存大幅下降 (~50%).
    # 代价: 训练速度 ~30% 慢.
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        logger.info(f"  ✅ UNet gradient checkpointing 已开启 (显存↓, 速度~70%)")

    logger.info(f"  Wrapper trainable params: {wrapper.num_trainable_params:,} "
                f"({wrapper.num_trainable_params / 1e6:.2f} M)")
    # 校验 trainable 参数 dtype
    _first_p = next(iter(wrapper.trainable_parameters()))
    logger.info(f"  trainable param dtype: {_first_p.dtype}  (期望 torch.float32)")
    assert _first_p.dtype == torch.float32, \
        "trainable params 必须是 fp32, 否则 GradScaler 会报错"

    wrapper.text_encoder.train()
    wrapper.dual_adapter.train()
    if wrapper.adapter_mid is not None:
        wrapper.adapter_mid.train()

    sobel_layer = ChannelwiseSobel(mode='fast').to(device, dtype=dtype)

    # ---------- 2) Optimizer + Scheduler ----------
    optimizer = torch.optim.AdamW(
        wrapper.trainable_parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: get_lr_factor(step, args.warmup_steps),
    )
    scaler = GradScaler("cuda", enabled=(args.mixed_precision == "fp16"))

    # ---------- 3) Dataset / DataLoader ----------
    dataset = build_dataset(args)
    logger.info(f"  dataset size: {len(dataset)}")
    loader = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers,
        pin_memory=True, drop_last=True,
    )
    logger.info(f"  total steps per epoch: {len(loader)}, "
                f"total steps for {args.num_epochs} epochs: {len(loader) * args.num_epochs}")

    # ---------- 4) 续训 (如有) ----------
    start_epoch = 0
    global_step = 0
    if args.resume_from and os.path.exists(args.resume_from):
        logger.info(f"[resume] {args.resume_from}")
        ckpt = torch.load(args.resume_from, map_location=device)
        wrapper.text_encoder.load_state_dict(ckpt["text_encoder"])
        wrapper.dual_adapter.load_state_dict(ckpt["dual_adapter"])
        if wrapper.adapter_mid is not None and "adapter_mid" in ckpt:
            wrapper.adapter_mid.load_state_dict(ckpt["adapter_mid"])
        optimizer.load_state_dict(ckpt["optimizer"])
        lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"]
        global_step = ckpt["global_step"]
        logger.info(f"  resumed at epoch {start_epoch}, step {global_step}")

    # ---------- 5) 训练循环 ----------
    unshuffle_layer = nn.PixelUnshuffle(downscale_factor=8)

    for epoch in range(start_epoch, args.num_epochs):
        epoch_t0 = time.time()
        running_loss = 0.0
        running_loss_latent = 0.0
        running_loss_pixel = 0.0
        n_batches = 0

        for batch_idx, batch in enumerate(loader):
            hq_bicubic = batch["hq_bicubic"].to(device, dtype=dtype, non_blocking=True)
            lr = batch["lr"].to(device, dtype=dtype, non_blocking=True)
            text_mask_hr = batch["text_mask_hr"].to(device, dtype=dtype, non_blocking=True)
            hr = batch["hr"].to(device, dtype=dtype, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=(args.mixed_precision == "fp16"),
                          dtype=torch.float16):

                # --- baseline 前向 (无梯度部分) ---
                with torch.no_grad():
                    encoded_control = vae.encode(hq_bicubic).latent_dist.sample() \
                                       * vae.config.scaling_factor
                    sft_cond = unshuffle_layer(hq_bicubic)
                    scale_p, shift_p = sft(sft_cond)
                    H8, W8 = encoded_control.shape[-2], encoded_control.shape[-1]
                    mix_w = compute_spatial_weight(hq_bicubic, (H8, W8), sobel_layer)
                    noise = torch.randn_like(encoded_control)
                    noise_a = noise * mix_w
                    lq_latent = noise_scheduler.add_noise(encoded_control, noise_a, timesteps)
                    coeff = eps_to_mu_coeff(noise_scheduler, lq_latent, timesteps)
                    unet_params = (scale_p / coeff, shift_p / coeff)
                    prompt_embeds = encode_prompt(
                        [args.default_prompt] * hq_bicubic.shape[0],
                        tokenizer, text_encoder, device, dtype,
                    )

                # --- v6 注入 + UNet 前向 (需要梯度) ---
                cond_5ch = build_cond_5ch(lr, text_mask_hr, upscale=args.scale_factor)
                # v6.2 路线 H: 传 text_mask, wrapper 内自动下采为空间门控
                wrapper.set_condition(cond_5ch, text_mask=text_mask_hr)

                # ⚠️ gradient_checkpointing 兼容:
                # checkpoint use_reentrant=True 要求至少一个输入 requires_grad=True,
                # 但我们的 lq_latent 来自 frozen VAE (requires_grad=False).
                # 显式打开 lq_latent.requires_grad_ 让 checkpoint 链路通畅.
                # (这不会让 lq_latent 被更新, 它没有进入 optimizer)
                if args.gradient_checkpointing:
                    lq_latent = lq_latent.detach().requires_grad_(True)

                pred_noise = unet(
                    lq_latent, timesteps,
                    encoder_hidden_states=prompt_embeds,
                    modulation_params=unet_params,
                ).sample
                wrapper.clear_condition()

                # ============= Loss =============
                # 1) Latent MSE (基础)
                per_pixel_mse = (pred_noise - noise_a) ** 2  # [B, 4, H8, W8]

                # 2) Text 区加权: weighted = mse * (1 + (w-1) * mask_lat)
                if args.text_loss_weight > 1.0:
                    mask_lat = F.interpolate(text_mask_hr, size=(H8, W8), mode="nearest")
                    mask_lat = mask_lat.to(dtype=per_pixel_mse.dtype)
                    weight_map = 1.0 + (args.text_loss_weight - 1.0) * mask_lat
                    loss_latent = (per_pixel_mse * weight_map).mean()
                else:
                    loss_latent = per_pixel_mse.mean()

                # 3) Pixel L1 (可选, 通过 decode pred_clean)
                if args.pixel_lambda > 0:
                    # pred_clean: 反推干净 latent (单步扩散)
                    pred_clean = encoded_control + coeff * (noise_a - pred_noise)
                    pred_img = vae.decode(
                        pred_clean / vae.config.scaling_factor
                    ).sample.clamp(-1, 1)
                    loss_pixel = F.l1_loss(pred_img, hr)
                else:
                    loss_pixel = torch.tensor(0.0, device=device, dtype=dtype)

                loss = loss_latent + args.pixel_lambda * loss_pixel

            # --- 反传 ---
            if args.mixed_precision == "fp16":
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    wrapper.trainable_parameters(), args.grad_clip
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    wrapper.trainable_parameters(), args.grad_clip
                )
                optimizer.step()

            lr_scheduler.step()

            # NaN/Inf 兜底
            loss_val = loss.detach().item()
            if not (loss_val == loss_val) or loss_val > 1e4:  # NaN or 爆炸
                logger.warning(f"[SKIP] step {global_step}: loss={loss_val}, 跳过此 step")
                global_step += 1
                continue

            running_loss += loss_val
            running_loss_latent += loss_latent.detach().item()
            running_loss_pixel += loss_pixel.detach().item() if args.pixel_lambda > 0 else 0
            n_batches += 1

            # --- log ---
            if global_step % args.log_every_n_steps == 0:
                cur_lr = optimizer.param_groups[0]["lr"]
                strength = wrapper.get_strength_dict()
                pix_val = loss_pixel.detach().item() if args.pixel_lambda > 0 else 0
                # 格式与 v5.1 对齐: 带时间戳 + Epoch/Step/Loss/LR + v6 特有 strength
                msg = (
                    f"Epoch {epoch+1}/{args.num_epochs} | Step {global_step:>6} | "
                    f"Loss: {loss_val:.4f} | Lat: {loss_latent.detach().item():.4f} | "
                    f"Pix: {pix_val:.4f} | LR: {cur_lr:.2e} | "
                    f"s_q4: {strength['strength_q4']:.4f} | s_q8: {strength['strength_q8']:.4f}"
                )
                if "strength_mid" in strength:
                    msg += f" | s_mid: {strength['strength_mid']:.4f}"
                logger.info(msg)

            global_step += 1

        # ---- epoch 结束 ----
        epoch_t = time.time() - epoch_t0
        avg_loss = running_loss / max(n_batches, 1)
        avg_lat = running_loss_latent / max(n_batches, 1)
        avg_pix = running_loss_pixel / max(n_batches, 1)
        logger.info(
            f"Epoch {epoch+1} finished. Avg Loss: {avg_loss:.4f} | "
            f"Avg Lat: {avg_lat:.4f} | Avg Pix: {avg_pix:.4f} | "
            f"Time: {epoch_t:.1f}s ({epoch_t/max(n_batches,1):.2f}s/step)"
        )

        # ---- Visualization ----
        if args.visualize_every_n_epochs > 0 and \
           (epoch + 1) % args.visualize_every_n_epochs == 0:
            try:
                vis_path = os.path.join(args.output_dir, "vis", f"epoch_{epoch+1:03d}.png")
                save_visualization(
                    wrapper, vae, sft, baseline, sobel_layer,
                    batch, args, vis_path, device, dtype,
                )
                logger.info(f"  vis saved: {vis_path}")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning(f"  [vis] OOM, skipped")
                    torch.cuda.empty_cache()
                else:
                    raise

        # ---- Checkpoint ----
        if (epoch + 1) % args.save_every_n_epochs == 0 or epoch + 1 == args.num_epochs:
            ckpt_path = os.path.join(args.output_dir, "checkpoints", f"epoch_{epoch+1:03d}.pth")
            ckpt = {
                "epoch": epoch + 1, "global_step": global_step,
                "text_encoder": wrapper.text_encoder.state_dict(),
                "dual_adapter": wrapper.dual_adapter.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "args": vars(args),
            }
            if wrapper.adapter_mid is not None:
                ckpt["adapter_mid"] = wrapper.adapter_mid.state_dict()
            torch.save(ckpt, ckpt_path)
            logger.info(f"  ckpt saved: {ckpt_path}")

    logger.info(f"[done] 训练完成. 输出: {args.output_dir}")


if __name__ == "__main__":
    main()
