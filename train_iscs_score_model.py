import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import SimpleITK as sitk
import contextlib
import torch
import torch.nn.functional as F
import yaml
from ml_collections import config_dict
from torch import nn
from torch.optim import Adam
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid, save_image

# Keep the import below for model registration.
import models.ncsnpp  # noqa: F401
import sampling
import sde_lib
from models.ema import ExponentialMovingAverage
from models import utils as mutils


# -----------------------------
# Utilities
# -----------------------------
def grad_and_param_stats(model):
    grad_sq = 0.0
    param_sq = 0.0
    grad_found = False
    for p in model.parameters():
        param_sq += float(p.detach().float().pow(2).sum().item())
        if p.grad is not None:
            g = p.grad.detach().float()
            grad_sq += float(g.pow(2).sum().item())
            grad_found = True
    grad_norm = math.sqrt(grad_sq) if grad_found else 0.0
    param_norm = math.sqrt(param_sq)
    return grad_norm, param_norm

def amp_autocast(device: torch.device, enabled: bool):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=enabled)
    return contextlib.nullcontext()

def make_json_serializable(obj):
    if isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.device):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DotConfig(config_dict.ConfigDict):
    """Small alias only for readability."""



def _ensure_cfg(cfg: DotConfig, key: str, value):
    if key not in cfg:
        cfg[key] = value



def load_config(config_path: str, args: argparse.Namespace) -> DotConfig:
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    cfg = DotConfig(data)

    _ensure_cfg(cfg, "training", DotConfig())
    _ensure_cfg(cfg, "sampling", DotConfig())
    _ensure_cfg(cfg, "data", DotConfig())
    _ensure_cfg(cfg, "model", DotConfig())
    _ensure_cfg(cfg, "optim", DotConfig())

    # Fill training fields missing from the ISCS inference configs.
    _ensure_cfg(cfg.training, "continuous", True)
    _ensure_cfg(cfg.training, "reduce_mean", False)
    _ensure_cfg(cfg.training, "sde", "vesde")
    _ensure_cfg(cfg.training, "likelihood_weighting", False)
    _ensure_cfg(cfg.training, "batch_size", args.batch_size)
    _ensure_cfg(cfg.training, "eval_batch_size", args.eval_batch_size)
    _ensure_cfg(cfg.training, "n_iters", args.n_iters)
    _ensure_cfg(cfg.training, "log_freq", args.log_freq)
    _ensure_cfg(cfg.training, "eval_freq", args.eval_freq)
    _ensure_cfg(cfg.training, "snapshot_freq", args.snapshot_freq)
    _ensure_cfg(cfg.training, "snapshot_freq_for_preemption", args.snapshot_freq)
    _ensure_cfg(cfg.training, "snapshot_sampling", args.snapshot_sampling)

    # Data defaults.
    _ensure_cfg(cfg.data, "num_channels", 1)
    _ensure_cfg(cfg.data, "centered", False)

    # Optimizer defaults.
    _ensure_cfg(cfg.optim, "optimizer", "Adam")
    _ensure_cfg(cfg.optim, "lr", args.lr)
    _ensure_cfg(cfg.optim, "beta1", args.beta1)
    _ensure_cfg(cfg.optim, "eps", args.adam_eps)
    _ensure_cfg(cfg.optim, "weight_decay", args.weight_decay)
    _ensure_cfg(cfg.optim, "warmup", args.warmup)
    _ensure_cfg(cfg.optim, "grad_clip", args.grad_clip)

    # CLI overrides.
    cfg.training.batch_size = args.batch_size
    cfg.training.eval_batch_size = args.eval_batch_size
    cfg.training.n_iters = args.n_iters
    cfg.training.log_freq = args.log_freq
    cfg.training.eval_freq = args.eval_freq
    cfg.training.snapshot_freq = args.snapshot_freq
    cfg.training.snapshot_freq_for_preemption = args.snapshot_freq
    cfg.training.snapshot_sampling = args.snapshot_sampling
    cfg.training.likelihood_weighting = args.likelihood_weighting

    cfg.optim.lr = args.lr
    cfg.optim.beta1 = args.beta1
    cfg.optim.eps = args.adam_eps
    cfg.optim.weight_decay = args.weight_decay
    cfg.optim.warmup = args.warmup
    cfg.optim.grad_clip = args.grad_clip

    cfg.device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    )
    return cfg



def get_data_scaler(centered: bool):
    if centered:
        return lambda x: x * 2.0 - 1.0
    return lambda x: x



def get_data_inverse_scaler(centered: bool):
    if centered:
        return lambda x: (x + 1.0) / 2.0
    return lambda x: x



def build_sde(cfg: DotConfig):
    name = cfg.training.sde.lower()
    if name == "vesde":
        sde = sde_lib.VESDE(
            sigma_min=cfg.model.sigma_min,
            sigma_max=cfg.model.sigma_max,
            N=cfg.model.num_scales,
        )
        eps = 1e-5
    elif name == "vpsde":
        sde = sde_lib.VPSDE(
            beta_min=cfg.model.beta_min,
            beta_max=cfg.model.beta_max,
            N=cfg.model.num_scales,
        )
        eps = 1e-3
    elif name == "subvpsde":
        sde = sde_lib.subVPSDE(
            beta_min=cfg.model.beta_min,
            beta_max=cfg.model.beta_max,
            N=cfg.model.num_scales,
        )
        eps = 1e-3
    else:
        raise NotImplementedError(f"Unsupported SDE: {cfg.training.sde}")
    return sde, eps


# -----------------------------
# Dataset
# -----------------------------


class NiftiSliceDataset(Dataset):
    def __init__(
        self,
        root: str,
        image_size: int,
        plane: str = "axial",
        normalization: str = "ct",
        hu_min: float = -1024.0,
        hu_max: float = 3072.0,
        cache_in_memory: bool = False,
        random_flip: bool = False,
        min_slice_std: float = 0.0,
    ):
        self.root = Path(root)
        self.image_size = int(image_size)
        self.plane = plane.lower()
        self.normalization = normalization.lower()
        self.hu_min = float(hu_min)
        self.hu_max = float(hu_max)
        self.cache_in_memory = cache_in_memory
        self.random_flip = random_flip
        self.min_slice_std = float(min_slice_std)

        if self.plane not in {"axial", "coronal", "sagittal"}:
            raise ValueError("plane must be one of: axial, coronal, sagittal")
        if self.normalization not in {"ct", "minmax", "none"}:
            raise ValueError("normalization must be one of: ct, minmax, none")

        self.files = self._discover_files(self.root)
        if not self.files:
            raise FileNotFoundError(f"No .nii or .nii.gz files found under {self.root}")

        self._vol_cache: Dict[int, np.ndarray] = {}
        self.index_map: List[Tuple[int, int]] = []
        self._build_index()
        if not self.index_map:
            raise RuntimeError("No usable slices were found. Check your plane/normalization/min_slice_std settings.")

    @staticmethod
    def _discover_files(root: Path) -> List[Path]:
        nii = sorted(root.rglob("*.nii"))
        niigz = sorted(root.rglob("*.nii.gz"))
        # Keep stable unique order.
        seen = set()
        files = []
        for p in nii + niigz:
            if p.as_posix() not in seen:
                files.append(p)
                seen.add(p.as_posix())
        return files

    def _axis_length(self, shape: Tuple[int, int, int]) -> int:
        d, h, w = shape
        if self.plane == "axial":
            return d
        if self.plane == "coronal":
            return h
        return w

    def _read_volume(self, file_idx: int) -> np.ndarray:
        if file_idx in self._vol_cache:
            return self._vol_cache[file_idx]
        img = sitk.ReadImage(str(self.files[file_idx]))
        vol = sitk.GetArrayFromImage(img).astype(np.float32)  # [D, H, W]
        if self.cache_in_memory:
            self._vol_cache[file_idx] = vol
        return vol

    def _extract_slice(self, volume: np.ndarray, slice_idx: int) -> np.ndarray:
        if self.plane == "axial":
            sl = volume[slice_idx, :, :]
        elif self.plane == "coronal":
            sl = volume[:, slice_idx, :]
        else:
            sl = volume[:, :, slice_idx]
        return sl.astype(np.float32)

    def _build_index(self) -> None:
        for file_idx, path in enumerate(self.files):
            img = sitk.ReadImage(str(path))
            size_w, size_h, size_d = img.GetSize()
            shape = (size_d, size_h, size_w)
            n_slices = self._axis_length(shape)

            if self.min_slice_std <= 0:
                self.index_map.extend((file_idx, s) for s in range(n_slices))
                continue

            volume = self._read_volume(file_idx)
            for s in range(n_slices):
                sl = self._extract_slice(volume, s)
                if float(sl.std()) >= self.min_slice_std:
                    self.index_map.append((file_idx, s))

    def _normalize(self, sl: np.ndarray) -> np.ndarray:
        if self.normalization == "ct":
            sl = np.clip(sl, self.hu_min, self.hu_max)
            denom = max(self.hu_max - self.hu_min, 1e-8)
            sl = (sl - self.hu_min) / denom
        elif self.normalization == "minmax":
            mn, mx = float(sl.min()), float(sl.max())
            if mx - mn < 1e-8:
                sl = np.zeros_like(sl, dtype=np.float32)
            else:
                sl = (sl - mn) / (mx - mn)
        else:
            sl = sl.astype(np.float32)
        return np.clip(sl, 0.0, 1.0).astype(np.float32)

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> torch.Tensor:
        file_idx, slice_idx = self.index_map[idx]
        volume = self._read_volume(file_idx)
        sl = self._extract_slice(volume, slice_idx)
        sl = self._normalize(sl)

        x = torch.from_numpy(sl).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        x = F.interpolate(
            x,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        x = x.squeeze(0)  # [1,H,W]

        if self.random_flip:
            if torch.rand(1).item() < 0.5:
                x = torch.flip(x, dims=(-1,))
            if torch.rand(1).item() < 0.5:
                x = torch.flip(x, dims=(-2,))

        return x.contiguous()


# -----------------------------
# Training loss / optimizer
# -----------------------------


def get_optimizer(cfg: DotConfig, params):
    if cfg.optim.optimizer != "Adam":
        raise NotImplementedError(f"Unsupported optimizer: {cfg.optim.optimizer}")
    return Adam(
        params,
        lr=cfg.optim.lr,
        betas=(cfg.optim.beta1, 0.999),
        eps=cfg.optim.eps,
        weight_decay=cfg.optim.weight_decay,
    )



def optimization_manager(cfg: DotConfig):
    def optimize_fn(optimizer, params, step: int, scaler=None):
        lr = cfg.optim.lr
        warmup = cfg.optim.warmup
        grad_clip = cfg.optim.grad_clip

        if warmup > 0:
            scaled_lr = lr * min(step / warmup, 1.0)
            for group in optimizer.param_groups:
                group["lr"] = scaled_lr

        if grad_clip >= 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)

        if scaler is None:
            optimizer.step()
        else:
            scaler.step(optimizer)
            scaler.update()

    return optimize_fn



def get_sde_loss_fn(
    sde,
    train: bool,
    reduce_mean: bool = True,
    continuous: bool = True,
    likelihood_weighting: bool = False,
    eps: float = 1e-5,
):
    reduce_op = torch.mean if reduce_mean else lambda *a, **kw: 0.5 * torch.sum(*a, **kw)

    def loss_fn(model: nn.Module, batch: torch.Tensor) -> torch.Tensor:
        score_fn = mutils.get_score_fn(sde, model, train=train, continuous=continuous)
        t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
        z = torch.randn_like(batch)
        mean, std = sde.marginal_prob(batch, t)
        perturbed = mean + std[:, None, None, None] * z
        score = score_fn(perturbed, t)

        if not likelihood_weighting:
            losses = torch.square(score * std[:, None, None, None] + z)
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
        else:
            g2 = sde.sde(torch.zeros_like(batch), t)[1] ** 2
            losses = torch.square(score + z / std[:, None, None, None])
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2
        return torch.mean(losses)

    return loss_fn



def get_step_fn(
    sde,
    train: bool,
    optimize_fn=None,
    reduce_mean: bool = False,
    continuous: bool = True,
    likelihood_weighting: bool = False,
    profile_freq: int = 10,
):
    loss_fn = get_sde_loss_fn(
        sde,
        train=train,
        reduce_mean=reduce_mean,
        continuous=continuous,
        likelihood_weighting=likelihood_weighting,
        eps=1e-5,
    )

    def step_fn(state: Dict, batch: torch.Tensor) -> torch.Tensor:
        model = state["model"]

        if train:
            optimizer = state["optimizer"]
            scaler = state.get("scaler", None)
            amp_enabled = bool(state.get("amp_enabled", False)) and batch.is_cuda

            optimizer.zero_grad(set_to_none=True)

            with amp_autocast(batch.device, amp_enabled):
                loss = loss_fn(model, batch)

            if scaler is None or not amp_enabled:
                loss.backward()
                optimize_fn(optimizer, model.parameters(), step=state["step"] + 1, scaler=None)
            else:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                optimize_fn(optimizer, model.parameters(), step=state["step"] + 1, scaler=scaler)

            state["step"] += 1
            state["ema"].update(model.parameters())

            return loss.detach()

        with torch.no_grad():
            ema = state["ema"]
            ema.store(model.parameters())
            ema.copy_to(model.parameters())
            with amp_autocast(batch.device, bool(state.get("amp_enabled", False)) and batch.is_cuda):
                loss = loss_fn(model, batch)
            ema.restore(model.parameters())
            return loss.detach()

    return step_fn


# -----------------------------
# Checkpointing / logging
# -----------------------------
def save_checkpoint(path: Path, state: Dict, cfg: DotConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "optimizer": state["optimizer"].state_dict(),
        "model": state["model"].state_dict(),
        "ema": state["ema"].state_dict(),
        "step": int(state["step"]),
        "config": cfg.to_dict(),
        "amp_enabled": bool(state.get("amp_enabled", False)),
    }
    scaler = state.get("scaler", None)
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    torch.save(payload, str(path))



def restore_checkpoint(path: Path, state: Dict, device: torch.device) -> Dict:
    if not path.exists():
        return state
    loaded = torch.load(str(path), map_location=device, weights_only=False)
    state["optimizer"].load_state_dict(loaded["optimizer"])
    state["model"].load_state_dict(loaded["model"], strict=True)
    state["ema"].load_state_dict(loaded["ema"])
    if "scaler" in loaded and state.get("scaler", None) is not None:
        state["scaler"].load_state_dict(loaded["scaler"])
    state["step"] = int(loaded.get("step", 0))
    state["amp_enabled"] = bool(loaded.get("amp_enabled", state.get("amp_enabled", False)))
    return state



def cycle(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch



def build_preview_sampler(cfg: DotConfig, sde, eps: float, device: torch.device):
    inverse_scaler = get_data_inverse_scaler(bool(cfg.data.centered))
    shape = (
        cfg.training.eval_batch_size,
        cfg.data.num_channels,
        cfg.data.image_size,
        cfg.data.image_size,
    )
    return sampling.get_sampling_fn(cfg, sde, shape, inverse_scaler, eps)



def save_sample_preview(
    sample_dir: Path,
    step: int,
    model: nn.Module,
    ema: ExponentialMovingAverage,
    sampling_fn,
) -> None:
    sample_dir.mkdir(parents=True, exist_ok=True)
    ema.store(model.parameters())
    ema.copy_to(model.parameters())
    with torch.no_grad():
        sample, _ = sampling_fn(model)
    ema.restore(model.parameters())

    sample = sample.clamp(0.0, 1.0)
    nrow = max(1, int(math.sqrt(sample.shape[0])))
    grid = make_grid(sample, nrow=nrow, padding=2)
    save_image(grid, sample_dir / f"samples_step_{step:07d}.png")


# -----------------------------
# Main
# -----------------------------



def main(args: argparse.Namespace) -> None:
    cfg = load_config(args.config_path, args)
    set_seed(args.seed if args.seed is not None else int(getattr(cfg, "seed", 42)))

    workdir = Path(args.workdir)
    ckpt_dir = workdir / "checkpoints"
    meta_ckpt = workdir / "checkpoints-meta" / "checkpoint.pth"
    sample_dir = workdir / "samples"
    workdir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    meta_ckpt.parent.mkdir(parents=True, exist_ok=True)

    dataset = NiftiSliceDataset(
        root=args.train_dir,
        image_size=cfg.data.image_size,
        plane=args.plane,
        normalization=args.normalization,
        hu_min=args.hu_min,
        hu_max=args.hu_max,
        cache_in_memory=args.cache_in_memory,
        random_flip=args.random_flip,
        min_slice_std=args.min_slice_std,
    )

    if args.val_dir:
        train_dataset = dataset
        val_dataset = NiftiSliceDataset(
            root=args.val_dir,
            image_size=cfg.data.image_size,
            plane=args.plane,
            normalization=args.normalization,
            hu_min=args.hu_min,
            hu_max=args.hu_max,
            cache_in_memory=args.cache_in_memory,
            random_flip=False,
            min_slice_std=args.min_slice_std,
        )
    else:
        n_total = len(dataset)
        n_val = max(1, int(round(n_total * args.val_ratio))) if n_total > 1 else 0
        n_train = max(1, n_total - n_val)
        if n_train + n_val > n_total:
            n_val = n_total - n_train
        generator = torch.Generator().manual_seed(args.split_seed)
        if n_val > 0:
            train_dataset, val_dataset = random_split(dataset, [n_train, n_val], generator=generator)
        else:
            train_dataset = dataset
            val_dataset = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.training.eval_batch_size,
            shuffle=False,
            num_workers=max(1, args.num_workers // 2) if args.num_workers > 0 else 0,
            pin_memory=True,
            drop_last=False,
            persistent_workers=args.num_workers > 0,
        )

    # Save a run manifest.
    manifest = {
        "args": vars(args),
        "num_train_slices": len(train_dataset),
        "num_val_slices": len(val_dataset) if val_dataset is not None else 0,
        "resolved_config": cfg.to_dict(),
        "device": str(cfg.device),
        "train_files": [str(p) for p in dataset.files],
    }
    with open(workdir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(make_json_serializable(manifest), f, indent=2, ensure_ascii=False)

    score_model = mutils.create_model(cfg)
    optimizer = get_optimizer(cfg, score_model.parameters())
    ema = ExponentialMovingAverage(score_model.parameters(), decay=cfg.model.ema_rate)

    amp_enabled = (cfg.device.type == "cuda") and (not getattr(args, "disable_amp", False))
    scaler = GradScaler(enabled=amp_enabled)

    state = {
        "optimizer": optimizer,
        "model": score_model,
        "ema": ema,
        "step": 0,
        "scaler": scaler,
        "amp_enabled": amp_enabled,
    }
    state = restore_checkpoint(meta_ckpt, state, cfg.device)
    initial_step = int(state["step"])

    sde, sampling_eps = build_sde(cfg)
    optimize_fn = optimization_manager(cfg)
    train_step_fn = get_step_fn(
        sde,
        train=True,
        optimize_fn=optimize_fn,
        reduce_mean=cfg.training.reduce_mean,
        continuous=cfg.training.continuous,
        likelihood_weighting=cfg.training.likelihood_weighting,
        profile_freq=-1,
    )
    eval_step_fn = get_step_fn(
        sde,
        train=False,
        optimize_fn=optimize_fn,
        reduce_mean=cfg.training.reduce_mean,
        continuous=cfg.training.continuous,
        likelihood_weighting=cfg.training.likelihood_weighting,
    )

    sampling_fn = None
    if args.snapshot_sampling:
        sampling_fn = build_preview_sampler(cfg, sde, sampling_eps, cfg.device)

    writer = SummaryWriter(log_dir=str(workdir / "tensorboard"))
    train_iter = cycle(train_loader)

    print(f"[INFO] device          : {cfg.device}", flush=True)
    print(f"[INFO] AMP enabled     : {state['amp_enabled']}", flush=True)
    print(f"[INFO] train slices    : {len(train_dataset)}", flush=True)
    print(f"[INFO] val slices      : {len(val_dataset) if val_dataset is not None else 0}", flush=True)
    print(f"[INFO] start step      : {initial_step}", flush=True)
    print(f"[INFO] total iters     : {cfg.training.n_iters}", flush=True)
    print(f"[INFO] checkpoint dir  : {ckpt_dir}", flush=True)

    for step in range(initial_step + 1, cfg.training.n_iters + 1):
        batch = next(train_iter).to(cfg.device, non_blocking=True).float()
        batch = get_data_scaler(bool(cfg.data.centered))(batch)
        train_loss = train_step_fn(state, batch)

        if step % cfg.training.log_freq == 0:
            current_lr = state["optimizer"].param_groups[0]["lr"]
            print(f"[train] step={step:06d} loss={train_loss.item():.6} lr={current_lr:.3e}")
            writer.add_scalar("loss/train", float(train_loss.item()), step)
            writer.add_scalar("lr", current_lr, step)

        if step % cfg.training.eval_freq == 0 and val_loader is not None:
            losses = []
            for val_batch in val_loader:
                val_batch = val_batch.to(cfg.device, non_blocking=True).float()
                val_batch = get_data_scaler(bool(cfg.data.centered))(val_batch)
                loss = eval_step_fn(state, val_batch)
                losses.append(float(loss.item()))
            val_loss = float(np.mean(losses)) if losses else float("nan")
            print(f"[ eval] step={step:07d} loss={val_loss:.6e}")
            writer.add_scalar("loss/val", val_loss, step)

        if step % cfg.training.snapshot_freq == 0 or step == cfg.training.n_iters:
            save_checkpoint(meta_ckpt, state, cfg)
            save_checkpoint(ckpt_dir / f"checkpoint_{step:07d}.pth", state, cfg)
            if sampling_fn is not None:
                save_sample_preview(sample_dir, step, state["model"], state["ema"], sampling_fn)

    # Save final EMA-only inference-friendly checkpoint alias.
    final_ckpt = ckpt_dir / "checkpoint_final.pth"
    save_checkpoint(final_ckpt, state, cfg)
    writer.close()
    print(f"[DONE] final checkpoint saved to: {final_ckpt}")



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a 2D score model for ISCS using the repo's NCSN++/SDE components."
    )
    parser.add_argument("--config-path", type=str, required=True, help="Path to an ISCS YAML config file.")
    parser.add_argument("--train-dir", type=str, required=True, help="Root folder containing training .nii/.nii.gz files.")
    parser.add_argument("--val-dir", type=str, default="", help="Optional validation folder with .nii/.nii.gz files.")
    parser.add_argument("--workdir", type=str, required=True, help="Output directory for checkpoints/logs.")
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device index. Use -1 for CPU.")

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--n-iters", type=int, default=200000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup", type=int, default=5000)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--log-freq", type=int, default=100)
    parser.add_argument("--eval-freq", type=int, default=2000)
    parser.add_argument("--snapshot-freq", type=int, default=5000)
    parser.add_argument("--snapshot-sampling", action="store_true")
    parser.add_argument("--likelihood-weighting", action="store_true")
    parser.add_argument("--disable-amp", action="store_true", help="Disable CUDA automatic mixed precision.")

    parser.add_argument("--plane", type=str, default="axial", choices=["axial", "coronal", "sagittal"])
    parser.add_argument("--normalization", type=str, default="ct", choices=["ct", "minmax", "none"])
    parser.add_argument("--hu-min", type=float, default=-1024.0)
    parser.add_argument("--hu-max", type=float, default=3072.0)
    parser.add_argument("--min-slice-std", type=float, default=0.0, help="Skip near-empty slices if > 0.")
    parser.add_argument("--random-flip", action="store_true")
    parser.add_argument("--cache-in-memory", action="store_true")

    parser.add_argument("--val-ratio", type=float, default=0.05, help="Used only when --val-dir is omitted.")
    parser.add_argument("--split-seed", type=int, default=1234)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())