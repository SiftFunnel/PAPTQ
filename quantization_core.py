# -*- coding: utf-8 -*-
"""
Mixed-precision QDQ quantization core
=================================

Three-stage method:
  Stage 1: Fisher-weighted quantization sensitivity estimation
           Compute U_{k,b} and R_{k,b} for each weight layer and bit width.
  Stage 2: Sensitivity-guided bit allocation
           Rank by R_{k,ref_bit}; allocate the bit budget using Stage-1 costs.
  Stage 3: Class-ambiguity-aware scale optimization
           With fixed bit widths, optimize activation and optional weight scale/rounding.
           The refinement balances low-bit behavior, task utility, and geometry:
           Stage 2 controls the average budget; Stage 3 refines W/A scales and hard-STE rounding,
           recovering accuracy while retaining aspects of the initial quantized geometry.

Important conventions:
  - All errors use real-valued QDQ differences: W_hat - W / A_hat - A.
  - Integer codes are never subtracted directly from full-precision values.
  - The quantizable model and data adapters are bundled in this package.

Public-source copy: pass local data, FP32 checkpoint, and output paths explicitly.
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import sys
import random
import collections
from collections import Counter, OrderedDict
from typing import Dict, Iterable, List, Tuple, Optional
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm

# ===== Local dependencies =====
# Resolve bundled modules relative to this public-source copy.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from resnet50_mp import *          # resnet50_ptq, ResNet50_PTQ, ActQuantLayerwise
from public_data import FaceScrub, Palm, IRIS_thousand



# -----------------------------
# Base configuration
# -----------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Mixed-precision QDQ quantization with W/A/total bit diagnostics")

    # Data and model
    parser.add_argument("--dataset", default="facescrub", type=str)
    parser.add_argument("--dataset_root", default="data", type=str)
    parser.add_argument(
        "--ckpt_path",
        default="checkpoints/fp32.pth",
        type=str,
    )
    parser.add_argument("--save_dir", default="outputs", type=str)
    parser.add_argument("--save_name", default="quantized_model.pth", type=str)

    # Pass architecture parameters that match the FP32 checkpoint
    parser.add_argument("--nc", default=3, type=int, help="input channels; FaceScrub often uses 3, grayscale uses 1")
    parser.add_argument("--nz", default=530, type=int, help="num classes for FaceScrub")
    parser.add_argument("--img_size", default=64, type=int)

    # loader
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--test_batch_size", default=64, type=int)
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--no_cuda", action="store_true", default=False)

    # Stage 1
    parser.add_argument("--bits", default="6,5,4,3", type=str)
    parser.add_argument("--ref_bit", default=4, type=int, help="bit used for ranking, e.g. R_{k,4}")
    parser.add_argument("--fisher_batches", default=50, type=int)
    parser.add_argument("--eps", default=1e-12, type=float)
    parser.add_argument("--weight_quant_granularity", default="per_channel", choices=["per_channel", "per_tensor"])

    # Stage 2
    parser.add_argument("--target_avg_bit", default=-1.0, type=float,
                        help="Convenience overall target. If >0, used for W/A targets unless the corresponding separate target is given.")
    parser.add_argument("--target_avg_w_bit", default=-1.0, type=float,
                        help="Parameter-weighted average weight bit budget. If <=0, uses --target_avg_bit or falls back to 4.0.")
    parser.add_argument("--target_avg_a_bit", default=-1.0, type=float,
                        help="Activation-size-weighted average activation bit budget. If <=0, uses --target_avg_bit or resolved W target.")
    parser.add_argument("--allow_empty_bucket", action="store_true", default=True)
    parser.add_argument("--force_nonempty_bucket", action="store_true", default=False)
    parser.add_argument("--alloc_mode", default="budget_greedy_v2", choices=["budget_greedy", "budget_greedy_v2", "fixed_ratio", "monotonic_search"],
                        help="budget_greedy_v2 uses Stage-1 absolute Fisher cost plus relative sensitivity for automatic demotion; fixed_ratio is kept only for ablation.")
    parser.add_argument("--bit_param_ratios", default="0.10,0.30,0.55,0.05", type=str,
                        help="only used by fixed_ratio ablation; budget_greedy ignores this and allocates automatically from Stage-1 costs.")
    parser.add_argument("--alloc_rel_sensitivity_alpha", default=1.0, type=float,
                        help="budget_greedy_v2: weight on relative sensitivity increase in the automatic demotion score")
    parser.add_argument("--alloc_score_saved_power", default=1.0, type=float,
                        help="budget_greedy_v2: divide normalized demotion cost by (saved parameter-bit fraction)^power")


    # Activation quantization policy
    parser.add_argument("--act_bits", default=4, type=int, help="uniform activation bit before scale optimization")
    parser.add_argument("--max_lowest_bit_param_ratio", default=-1.0, type=float,
                        help="hard cap on parameter ratio assigned to the lowest bit in fixed-ratio allocation; use -1 to disable")

    parser.add_argument("--act_bit_mode", default="from_weight", choices=["uniform", "from_weight"],
                        help="uniform: all activations use --act_bits; from_weight: call assign_act_bits_from_weight_cfg if available")
    parser.add_argument("--min_act_bits", default=3, type=int,
                        help="Minimum activation bit when using mixed activation bits. Use 3 when testing W/A=3.5 or W/A=3.0.")

    # Stage 3
    parser.add_argument("--scale_epochs", default=50, type=int)
    parser.add_argument("--phase1_epochs", default=0, type=int,
                        help="Stage3 phase-A utility recovery epochs before inter-contraction phase; 0 disables two-phase behavior")
    parser.add_argument("--scale_lr", default=3e-5, type=float)
    parser.add_argument("--scale_lr_schedule", default="", type=str,
                        help="Optional comma-separated Stage3 base LR values, e.g. 5e-5,3e-5,1e-5. Overrides --scale_lr by epoch.")
    parser.add_argument("--scale_lr_milestones", default="", type=str,
                        help="Optional comma-separated 1-based epoch starts for --scale_lr_schedule, e.g. 1,41,76 for 100 epochs.")
    parser.add_argument("--rho", default=math.log(2.0), type=float, help="log-scale projection radius")
    parser.add_argument("--out_kl_eps", default=0.30, type=float)
    parser.add_argument("--lambda_kl", default=1.0, type=float)
    parser.add_argument("--lambda_nll", default=6.0, type=float, help="task recovery loss; v13 uses label-smoothing CE by default")
    parser.add_argument("--nll_loss_mode", default="raw", choices=["raw", "hinge"],
                        help="raw: use CE directly; hinge: penalize CE only when above --ce_eps")
    parser.add_argument("--ce_eps", default=0.25, type=float,
                        help="CE/NLL tolerance for hinge CE constraint")
    parser.add_argument("--label_smoothing", default=0.10, type=float,
                        help="generalized label smoothing for task recovery CE. Positive values soften labels; negative values enable experimental label sharpening / negative label smoothing.")
    parser.add_argument("--kl_loss_mode", default="raw", choices=["raw", "hinge"],
                        help="raw: use KL directly with a small weight; hinge: only penalize KL above --out_kl_eps")
    parser.add_argument("--lambda_out_inter_retain", default=2.0, type=float,
                        help="adaptive inter-retention loss: penalize Stage-3 if output inter distance grows beyond pre-Stage3 quantized baseline")
    parser.add_argument("--lambda_feat_inter_retain", default=0.5, type=float,
                        help="adaptive inter-retention loss for target feature layer relative to pre-Stage3 quantized baseline")
    parser.add_argument("--target_feature_module", default="auto", type=str,
                        help="module name for intra-class feature loss; auto chooses avgpool/layer4/last ActQuantLayerwise")
    parser.add_argument("--calib_batches_scale", default=0, type=int,
                        help="0 means use full calibration loader during scale optimization")

    # Stage 3 class-balanced / prototype options
    parser.add_argument("--target_feature_idx", default=21, type=int,
                        help="returned feature index used by Stage 3 losses; set <0 to use --target_feature_module hook")
    parser.add_argument("--scale_classes_per_batch", default=32, type=int,
                        help="P in class-balanced scale optimization batch: P classes x K samples")
    parser.add_argument("--scale_samples_per_class", default=4, type=int,
                        help="K in class-balanced scale optimization batch: P classes x K samples")
    parser.add_argument("--scale_batches_per_epoch", default=0, type=int,
                        help="number of class-balanced batches per scale epoch; 0 uses ceil(N/(P*K))")
    parser.add_argument("--lambda_intra", default=0.0, type=float,
                        help="weight for batch class-balanced intra-class compactness loss; v9 uses direct output/feature contraction objectives")
    parser.add_argument("--lambda_proto", default=0.0, type=float,
                        help="weight for attraction to full-training-set FP32 class prototypes")
    parser.add_argument("--lambda_scale_reg", default=1e-4, type=float,
                        help="regularization for activation log-scale deviation from calibrated base scale")
    parser.add_argument("--lambda_out_proto", default=0.0, type=float,
                        help="weight for output class-prototype consistency loss; v9 emphasizes direct intra-class output contraction")
    parser.add_argument("--lambda_out_intra", default=0.0, type=float,
                        help="weight for batch intra-class probability consistency loss")
    parser.add_argument("--lambda_inter", default=0.0, type=float,
                        help="diagnostic/optional: preserving FP32 inter-class geometry; v10 default disables active inter-class preservation")

    # v10 controlled class-ambiguity losses.
    # These actively shrink inter-class distances under CE/margin/KL utility constraints.
    parser.add_argument("--lambda_out_inter_contract", default=0.0, type=float,
                        help="penalize when output class-center distance is not reduced relative to FP32 batch baseline")
    parser.add_argument("--rho_out_inter", default=0.90, type=float,
                        help="target ratio for output inter-class distance: require D_inter^Q <= rho_out_inter * D_inter^FP")
    parser.add_argument("--lambda_feat_inter_contract", default=0.0, type=float,
                        help="penalize when target-feature class-center distance is not reduced relative to FP32 batch baseline")
    parser.add_argument("--rho_feat_inter", default=0.95, type=float,
                        help="target ratio for feature inter-class distance: require D_inter^Q <= rho_feat_inter * D_inter^FP")
    parser.add_argument("--inter_loss_mode", default="ratio", choices=["hinge", "ratio", "target"],
                        help="hinge: max(0, ratio-rho)^2; ratio: directly minimize ratio; target: (ratio-target)^2")
    parser.add_argument("--out_inter_target", default=0.80, type=float,
                        help="target ratio for output inter distance when --inter_loss_mode target")
    parser.add_argument("--feat_inter_target", default=0.85, type=float,
                        help="target ratio for feature inter distance when --inter_loss_mode target")
    parser.add_argument("--inter_temp", default=1.0, type=float,
                        help="temperature used for output inter-class distance; v11 default 1.0 to align with final output metrics")
    parser.add_argument("--lambda_margin", default=0.0, type=float,
                        help="utility-preserving margin loss on log probabilities")
    parser.add_argument("--margin_target", default=2.0, type=float,
                        help="target log-probability margin for the true class")

    # v9/v10 direct contraction losses. These avoid tiny raw probability-MSE losses.
    parser.add_argument("--lambda_out_contract", default=0.0, type=float,
                        help="directly penalize when intra-class output distance is not smaller than FP32 batch baseline")
    parser.add_argument("--rho_out", default=0.95, type=float,
                        help="target contraction ratio for output intra distance: require D_q <= rho_out * D_fp")
    parser.add_argument("--output_temp", default=4.0, type=float,
                        help="temperature applied to log-probabilities before computing output intra distances")
    parser.add_argument("--output_proto_eta", default=0.3, type=float,
                        help="blend FP32 output prototype with one-hot class target: target=(1-eta)*proto+eta*onehot")
    parser.add_argument("--output_proto_sharpen", default=1.5, type=float,
                        help="sharpen FP32 output prototypes before one-hot blending")
    parser.add_argument("--lambda_logp_intra", default=0.0, type=float,
                        help="intra-class consistency on normalized log-probability vectors; gives stronger gradients than saturated probabilities")

    parser.add_argument("--lambda_feat_contract", default=0.0, type=float,
                        help="directly penalize when target-layer feature intra distance is not smaller than FP32 batch baseline")
    parser.add_argument("--rho_feat", default=0.98, type=float,
                        help="target contraction ratio for feature intra distance: require D_q <= rho_feat * D_fp")

    # Optional privacy-aware weight quantization refinement.
    parser.add_argument("--optimize_weight_scale", dest="optimize_weight_scale", action="store_true", default=True,
                        help="optimize weight quantization scales during Stage 3")
    parser.add_argument("--no_optimize_weight_scale", dest="optimize_weight_scale", action="store_false")
    parser.add_argument("--optimize_weight_rounding", dest="optimize_weight_rounding", action="store_true", default=True,
                        help="also optimize soft rounding direction; default off in v8 to avoid soft-to-hard gap during output-consistency experiments")
    parser.add_argument("--no_optimize_weight_rounding", dest="optimize_weight_rounding", action="store_false")
    parser.add_argument("--weight_round_forward", default="hard_ste", choices=["soft", "hard_ste"],
                        help="soft: differentiable sigmoid rounding in forward; hard_ste: hard rounding forward with STE gradient to avoid soft-to-hard gap")
    parser.add_argument("--weight_round_temperature", default=1.0, type=float,
                        help="temperature for sigmoid soft rounding")
    parser.add_argument("--weight_round_init_logit", default=0.0, type=float,
                        help="legacy fixed-logit init; v13 uses fractional init by default when --round_init_mode frac")
    parser.add_argument("--round_init_mode", default="frac", choices=["frac", "nearest_logit"],
                        help="frac initializes soft rounding from fractional W/s; nearest_logit reproduces old large-logit nearest-rounding relaxation")
    parser.add_argument("--weight_scale_lr_mult", default=1.0, type=float,
                        help="learning-rate multiplier for weight log-scale deltas")
    parser.add_argument("--weight_rho", default=math.log(1.25), type=float,
                        help="clamp radius for weight log-scale deltas")
    parser.add_argument("--lambda_weight_scale_reg", default=1e-5, type=float,
                        help="regularization for weight log-scale deltas")
    parser.add_argument("--lambda_round_reg", default=1e-4, type=float,
                        help="regularization encouraging soft rounding variables to become hard")
    parser.add_argument("--round_temp_anneal", default=0.97, type=float,
                        help="multiply weight rounding temperature by this factor after each epoch; 1 disables annealing")
    parser.add_argument("--min_round_temp", default=0.25, type=float,
                        help="lower bound for rounding temperature annealing")
    parser.add_argument("--hard_eval_each_epoch", action="store_true", default=True,
                        help="evaluate hard-forward model every epoch when weight rounding is enabled")

    # Diagnostics
    parser.add_argument("--dry_run", action="store_true", default=False)

    return parser


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(args) -> torch.device:
    use_cuda = torch.cuda.is_available() and not args.no_cuda
    return torch.device("cuda" if use_cuda else "cpu")


# -----------------------------
# Dataset and model loading
# -----------------------------
def build_transform(nc: int, dataset: str = "facescrub"):
    """
    Deterministic transform for PTQ calibration / Fisher estimation / evaluation.

    Important:
      - Do NOT use RandomHorizontalFlip / ColorJitter here. Those are training augmentations.
      - FaceScrub ResNet50 was trained with RGB CIFAR-style normalization:
          mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
        so PTQ and FP32 evaluation must use the same normalization.
    """
    dataset = dataset.lower()

    if nc == 1:
        # Keep grayscale datasets consistent with the previous code path.
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])

    if dataset in ["facescrub", "face"]:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010),
            ),
        ])

    # Fallback for other RGB datasets. Override this if their training transform differs.
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def resolve_dataset_class(dataset_name: str):
    """Resolve legacy dataset names against public adapters."""
    name = dataset_name.lower()
    candidates = []
    if name in ["facescrub", "face"]:
        candidates = ["FaceScrub", "Facescrub", "facescrub", "FACE"]
    elif name == "palm":
        candidates = ["Palm"]
    elif name in ["thousand", "iris_thousand", "iris"]:
        candidates = ["IRIS_thousand", "Iris_thousand", "IRISThousand"]
    else:
        candidates = [dataset_name]

    g = globals()
    for cls_name in candidates:
        if cls_name in g:
            return g[cls_name]
    raise RuntimeError(
        f"Cannot find dataset class for dataset={dataset_name}. Tried: {candidates}. "
        f"Please check target_model_packages.py class name."
    )


def build_loaders(args):
    transform = build_transform(args.nc, args.dataset)
    DatasetCls = resolve_dataset_class(args.dataset)

    calib_dataset = DatasetCls(root=args.dataset_root, train=True, transform=transform)
    test_set = DatasetCls(root=args.dataset_root, train=False, transform=transform)

    calib_loader = torch.utils.data.DataLoader(
        calib_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return calib_loader, test_loader, calib_dataset


def _label_to_int(y):
    if torch.is_tensor(y):
        return int(y.detach().cpu().view(-1)[0].item())
    return int(y)


class ClassBalancedBatchSampler(torch.utils.data.Sampler):
    """Yield batches with P classes and K samples per class for Stage-3 scale optimization.

    This fixes the 530-class random-batch issue where most classes appear only once in a
    mini-batch, making the intra-class compactness loss almost gradient-free.
    """

    def __init__(self, dataset, num_classes: int, classes_per_batch: int, samples_per_class: int,
                 batches_per_epoch: int = 0, seed: int = 1):
        self.dataset = dataset
        self.num_classes = int(num_classes)
        self.classes_per_batch = int(classes_per_batch)
        self.samples_per_class = int(samples_per_class)
        self.batch_size = self.classes_per_batch * self.samples_per_class
        self.seed = int(seed)
        self.epoch = 0

        label_to_indices = {c: [] for c in range(self.num_classes)}
        for idx in tqdm(range(len(dataset)), desc="[ScaleSampler] build label_to_indices", leave=False):
            item = dataset[idx]
            y = _label_to_int(item[1])
            if 0 <= y < self.num_classes:
                label_to_indices[y].append(idx)
        self.label_to_indices = {c: inds for c, inds in label_to_indices.items() if len(inds) > 0}
        self.valid_classes = sorted(self.label_to_indices.keys())
        if len(self.valid_classes) == 0:
            raise RuntimeError("ClassBalancedBatchSampler found no valid classes.")
        if self.classes_per_batch > len(self.valid_classes):
            print(f"[ScaleSampler][WARN] classes_per_batch={self.classes_per_batch} > covered classes={len(self.valid_classes)}; sampling classes with replacement.")
        if batches_per_epoch and batches_per_epoch > 0:
            self.batches_per_epoch = int(batches_per_epoch)
        else:
            self.batches_per_epoch = int(math.ceil(len(dataset) / max(self.batch_size, 1)))
        print(
            f"[ScaleSampler] covered_classes={len(self.valid_classes)}/{self.num_classes}, "
            f"P={self.classes_per_batch}, K={self.samples_per_class}, "
            f"batch_size={self.batch_size}, batches_per_epoch={self.batches_per_epoch}"
        )

    def __len__(self):
        return self.batches_per_epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        valid = np.array(self.valid_classes, dtype=np.int64)
        class_replace = len(valid) < self.classes_per_batch
        for _ in range(self.batches_per_epoch):
            chosen_classes = rng.choice(valid, size=self.classes_per_batch, replace=class_replace)
            batch = []
            for c in chosen_classes.tolist():
                inds = self.label_to_indices[int(c)]
                replace = len(inds) < self.samples_per_class
                sampled = rng.choice(np.array(inds, dtype=np.int64), size=self.samples_per_class, replace=replace)
                batch.extend(sampled.tolist())
            rng.shuffle(batch)
            yield batch


def build_scale_loader(args, calib_dataset):
    sampler = ClassBalancedBatchSampler(
        calib_dataset,
        num_classes=args.nz,
        classes_per_batch=args.scale_classes_per_batch,
        samples_per_class=args.scale_samples_per_class,
        batches_per_epoch=args.scale_batches_per_epoch,
        seed=args.seed,
    )
    loader = torch.utils.data.DataLoader(
        calib_dataset,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    return loader


def load_resnet50_ptq(args, device: torch.device):
    model = resnet50_ptq(in_channels=args.nc, num_classes=args.nz, img_size=args.img_size).to(device)
    print("\n--- Load FP32 checkpoint ---")
    print(f"ckpt_path: {args.ckpt_path}")
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    state_dict = {k.replace("_module.", ""): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if len(missing) > 0:
        print(f"[WARN] missing keys: {len(missing)}; first few: {missing[:5]}")
    if len(unexpected) > 0:
        print(f"[WARN] unexpected keys: {len(unexpected)}; first few: {unexpected[:5]}")
    if isinstance(ckpt, dict):
        print(f"epoch={ckpt.get('epoch', 'NA')}, best_cl_acc={ckpt.get('best_cl_acc', 'NA')}")
    model.eval()
    return model


@torch.no_grad()
def eval_classifier(model: nn.Module, loader, device: torch.device, max_batches: int = 0) -> float:
    model.eval()
    correct, total = 0, 0
    for b_idx, (x, y) in enumerate(loader):
        if max_batches and b_idx >= max_batches:
            break
        x = x.to(device)
        y = y.to(device)
        logp = model(x)
        pred = logp.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


# -----------------------------
# Quantize-dequantize helpers
# -----------------------------
def qdq_symmetric_weight(w: torch.Tensor, bits: int, granularity: str = "per_channel", eps: float = 1e-12) -> torch.Tensor:
    """
    Symmetric fake quantization for weights.
    Returns dequantized real tensor W_hat, NOT integer code.
    """
    if bits >= 16:
        return w.detach().clone()

    qmax = (2 ** (bits - 1)) - 1
    qmin = -(2 ** (bits - 1))
    if qmax <= 0:
        raise ValueError(f"invalid bits={bits}")

    w_det = w.detach()
    if granularity == "per_channel" and w_det.dim() in [2, 4]:
        if w_det.dim() == 4:
            max_abs = w_det.abs().amax(dim=(1, 2, 3), keepdim=True)
        else:
            max_abs = w_det.abs().amax(dim=1, keepdim=True)
    else:
        max_abs = w_det.abs().amax()

    scale = torch.clamp(max_abs / float(qmax), min=eps)
    q = torch.round(w_det / scale).clamp(qmin, qmax)
    w_hat = q * scale
    return w_hat.to(dtype=w.dtype)


# -----------------------------
# Stage 1: Fisher + quant sensitivity
# -----------------------------
def iter_weight_modules(model) -> Iterable[Tuple[str, nn.Module]]:
    if hasattr(model, "_iter_weight_modules"):
        yield from model._iter_weight_modules()
    else:
        for name, m in model.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                yield name, m


def compute_weight_fisher_tensors(
    model,
    loader,
    device: torch.device,
    max_batches: int = 50,
) -> "OrderedDict[str, torch.Tensor]":
    """Accumulate diagonal Fisher tensors E[(dL/dW)^2] for Conv/Linear weights."""
    if hasattr(model, "set_fp32_train_mode"):
        model.set_fp32_train_mode()
    model.to(device)
    model.train()

    fisher = OrderedDict()
    for name, m in iter_weight_modules(model):
        fisher[name] = torch.zeros_like(m.weight.detach(), device="cpu")

    n_batches = 0
    for b_idx, (x, y) in enumerate(tqdm(loader, desc="[Stage1] Fisher", leave=False)):
        if max_batches and b_idx >= max_batches:
            break
        x = x.to(device)
        y = y.to(device)

        model.zero_grad(set_to_none=True)
        logp = model(x)
        loss = F.nll_loss(logp, y.long())
        loss.backward()

        for name, m in iter_weight_modules(model):
            if m.weight.grad is None:
                continue
            fisher[name] += (m.weight.grad.detach().float().cpu() ** 2)
        n_batches += 1

    if n_batches > 0:
        for name in fisher.keys():
            fisher[name] /= float(n_batches)

    return fisher


def compute_weight_sensitivity_tables(
    model,
    fisher: "OrderedDict[str, torch.Tensor]",
    bits: List[int],
    granularity: str,
    eps: float,
):
    """
    Returns:
      U[name][bit] = 0.5 * sum F * (W_hat - W)^2       absolute predicted task cost
      R[name][bit] = sum F * (W_hat - W)^2 / (sum F*W^2 + eps) relative sensitivity
      n_params[name]
    """
    U = OrderedDict()
    R = OrderedDict()
    n_params = OrderedDict()

    module_dict = OrderedDict(iter_weight_modules(model))
    for name, m in module_dict.items():
        if name not in fisher:
            continue
        w = m.weight.detach().float().cpu()
        Fw = fisher[name].float().cpu()
        denom = (Fw * (w ** 2)).sum().item() + eps
        U[name] = OrderedDict()
        R[name] = OrderedDict()
        n_params[name] = int(w.numel())

        for b in bits:
            w_hat = qdq_symmetric_weight(w, bits=b, granularity=granularity, eps=eps).float().cpu()
            err2 = (w_hat - w) ** 2
            num = (Fw * err2).sum().item()
            U[name][b] = 0.5 * num
            R[name][b] = num / denom

    return U, R, n_params


# -----------------------------
# Stage 2: bit allocation
# -----------------------------
def enumerate_monotonic_bucket_allocation(
    U: Dict[str, Dict[int, float]],
    R: Dict[str, Dict[int, float]],
    n_params: Dict[str, int],
    bits: List[int],
    ref_bit: int,
    target_avg_bit: float,
    force_nonempty_bucket: bool = False,
):
    """
    Sort layers by R[name][ref_bit] descending, then search monotonic bucket boundaries.
    bits must be descending, e.g. [8,6,4,3].
    Objective: minimize sum U[k][assigned_bit]
    Constraint: sum n_k*b_k <= target_avg_bit * sum n_k
    """
    bits = sorted(bits, reverse=True)
    assert len(bits) == 4, "This implementation expects exactly 4 bit candidates, e.g. 8,6,4,3."
    b0, b1, b2, b3 = bits

    names = list(R.keys())
    if ref_bit not in bits:
        raise ValueError(f"ref_bit={ref_bit} not in bits={bits}")

    names.sort(key=lambda n: R[n][ref_bit], reverse=True)
    K = len(names)
    total_params = sum(n_params[n] for n in names)
    budget = target_avg_bit * total_params

    best = None
    candidates = 0

    # boundaries r0 <= r1 <= r2; ranges:
    # [0,r0)->b0, [r0,r1)->b1, [r1,r2)->b2, [r2,K)->b3
    for r0 in range(0, K + 1):
        for r1 in range(r0, K + 1):
            for r2 in range(r1, K + 1):
                sizes = [r0, r1 - r0, r2 - r1, K - r2]
                if force_nonempty_bucket and any(s <= 0 for s in sizes):
                    continue

                bit_cfg = OrderedDict()
                for i, name in enumerate(names):
                    if i < r0:
                        b = b0
                    elif i < r1:
                        b = b1
                    elif i < r2:
                        b = b2
                    else:
                        b = b3
                    bit_cfg[name] = int(b)

                cost = sum(n_params[n] * bit_cfg[n] for n in names)
                if cost > budget + 1e-9:
                    continue

                obj = sum(U[n][bit_cfg[n]] for n in names)
                candidates += 1
                if best is None or obj < best["obj"]:
                    best = {
                        "obj": obj,
                        "cost": cost,
                        "budget": budget,
                        "avg_bit": cost / max(total_params, 1),
                        "boundaries": (r0, r1, r2),
                        "bit_cfg": dict(bit_cfg),
                        "sorted_names": names,
                        "sizes": sizes,
                    }

    if best is None:
        raise RuntimeError(
            f"No feasible allocation found. target_avg_bit={target_avg_bit}, bits={bits}. "
            f"Try larger --target_avg_w_bit or allow empty buckets."
        )

    print("\n[Stage2] Best monotonic bucket allocation")
    print(f"  candidates={candidates}")
    print(f"  boundaries={best['boundaries']}, sizes={best['sizes']}")
    print(f"  avg_bit={best['avg_bit']:.4f}, objective={best['obj']:.6e}")
    print(f"  bit distribution={Counter(best['bit_cfg'].values())}")
    print("\n[Stage2] Sorted sensitivity and assigned bits:")
    for name in best["sorted_names"]:
        print(f"  {name:50s} R@{ref_bit}={R[name][ref_bit]:.3e} U@bit={U[name][best['bit_cfg'][name]]:.3e} bit={best['bit_cfg'][name]}")

    return best["bit_cfg"], best




def _parse_float_list(s: str) -> List[float]:
    vals = [float(x.strip()) for x in str(s).split(",") if x.strip()]
    if len(vals) == 0:
        raise ValueError("empty float list")
    return vals




def budget_greedy_bit_allocation(
    U: Dict[str, Dict[int, float]],
    R: Dict[str, Dict[int, float]],
    n_params: Dict[str, int],
    bits: List[int],
    ref_bit: int,
    target_avg_bit: float,
):
    """Automatic Stage-2 allocation using only Stage-1 costs and the average-bit budget.

    Start from the highest bit for every layer. Repeatedly demote one layer to the
    next lower candidate bit with the smallest incremental Fisher task cost per
    saved parameter-bit until the weighted average bit is within budget.

    This is not a hand-coded layer constraint or a forced ratio: all decisions are
    made from U[k][b] and layer parameter counts. The resulting allocation can use
    6/5/4/3 as needed, and normally uses 3-bit only when its incremental cost is
    justified by the Stage-1 table.
    """
    bits = sorted([int(b) for b in bits], reverse=True)
    names = list(U.keys())
    total_params = sum(int(n_params[n]) for n in names)
    budget = float(target_avg_bit) * total_params
    bit_cfg = OrderedDict((n, bits[0]) for n in names)
    current_cost = sum(n_params[n] * bit_cfg[n] for n in names)
    steps = []

    # Map current bit to next lower bit.
    next_lower = {bits[i]: bits[i + 1] for i in range(len(bits) - 1)}

    while current_cost > budget + 1e-9:
        candidates = []
        for n in names:
            b = int(bit_cfg[n])
            if b not in next_lower:
                continue
            nb = int(next_lower[b])
            saved = int(n_params[n]) * (b - nb)
            if saved <= 0:
                continue
            delta_u = float(U[n][nb]) - float(U[n][b])
            # Tie-breaker prefers lower R@ref_bit and smaller absolute U.
            score = delta_u / max(saved, 1)
            candidates.append((score, float(R[n].get(ref_bit, 0.0)), delta_u, n, b, nb, saved))
        if not candidates:
            break
        score, rref, delta_u, n, b, nb, saved = min(candidates, key=lambda x: (x[0], x[1], x[2]))
        bit_cfg[n] = nb
        current_cost -= saved
        steps.append({"name": n, "from": b, "to": nb, "saved": saved, "delta_u": delta_u, "score": score})

    obj = sum(float(U[n][int(bit_cfg[n])]) for n in names)
    bucket_params = [sum(n_params[n] for n in names if int(bit_cfg[n]) == b) for b in bits]
    alloc_info = {
        "mode": "budget_greedy",
        "bits": bits,
        "target_avg_bit": float(target_avg_bit),
        "avg_bit": current_cost / max(total_params, 1),
        "budget": budget,
        "cost": current_cost,
        "objective": obj,
        "num_demotions": len(steps),
        "bucket_params": bucket_params,
        "bucket_param_ratios_actual": [p / max(total_params, 1) for p in bucket_params],
        "steps_preview": steps[:50],
    }

    print("\n[Stage2] Budget-greedy automatic mixed-bit allocation")
    print(f"  bits={bits}, target_avg={target_avg_bit:.4f}")
    print(f"  avg_bit={alloc_info['avg_bit']:.4f}, objective={obj:.6e}, demotions={len(steps)}")
    print(f"  layer bit distribution={Counter(bit_cfg.values())}")
    print(f"  bucket_params={bucket_params}")
    print(f"  actual param ratios={alloc_info['bucket_param_ratios_actual']}")
    print("\n[Stage2] Assigned bits sorted by R@ref:")
    sorted_names = sorted(names, key=lambda n: R[n][ref_bit], reverse=True)
    for name in sorted_names:
        b = int(bit_cfg[name])
        print(f"  {name:50s} R@{ref_bit}={R[name][ref_bit]:.3e} U@bit={U[name][b]:.3e} bit={b}")
    return dict(bit_cfg), alloc_info



def budget_greedy_v2_bit_allocation(
    U: Dict[str, Dict[int, float]],
    R: Dict[str, Dict[int, float]],
    n_params: Dict[str, int],
    bits: List[int],
    ref_bit: int,
    target_avg_bit: float,
    rel_alpha: float = 1.0,
    saved_power: float = 1.0,
    eps: float = 1e-12,
):
    """Automatic Stage-2 allocation with normalized demotion cost.

    v13 keeps the allocation automatic, but avoids the failure mode where a large
    semantic layer is demoted because its absolute Fisher cost per saved bit looks
    small.  For each possible one-step demotion b -> nb, the score is

        ( relative increase in Fisher task cost
          + alpha * relative increase in quantization sensitivity )
        / (saved_parameter_bit_fraction ** saved_power)

    The layer with the smallest score is demoted until the average-bit budget is
    reached.  This is still fully determined by Stage-1 U/R tables and parameter
    counts; there are no hand-coded layer constraints.
    """
    bits = sorted([int(b) for b in bits], reverse=True)
    names = list(U.keys())
    total_params = sum(int(n_params[n]) for n in names)
    budget = float(target_avg_bit) * total_params
    bit_cfg = OrderedDict((n, bits[0]) for n in names)
    current_cost = sum(n_params[n] * bit_cfg[n] for n in names)
    steps = []
    next_lower = {bits[i]: bits[i + 1] for i in range(len(bits) - 1)}

    while current_cost > budget + 1e-9:
        candidates = []
        for n in names:
            b = int(bit_cfg[n])
            if b not in next_lower:
                continue
            nb = int(next_lower[b])
            saved = int(n_params[n]) * (b - nb)
            if saved <= 0:
                continue
            u_b = float(U[n][b])
            u_nb = float(U[n][nb])
            r_b = float(R[n].get(b, R[n].get(ref_bit, 0.0)))
            r_nb = float(R[n].get(nb, R[n].get(ref_bit, 0.0)))
            delta_u = max(u_nb - u_b, 0.0)
            delta_r = max(r_nb - r_b, 0.0)
            rel_u = delta_u / (abs(u_b) + eps)
            rel_r = delta_r / (abs(r_b) + eps)
            saved_frac = max(saved / max(total_params, 1), eps)
            score = (rel_u + float(rel_alpha) * rel_r) / (saved_frac ** float(saved_power))
            # Tie breakers: lower absolute U increase, lower R@ref, more saved bits.
            candidates.append((score, delta_u, float(R[n].get(ref_bit, 0.0)), -saved, n, b, nb, saved, rel_u, rel_r))
        if not candidates:
            break
        score, delta_u, rref, neg_saved, n, b, nb, saved, rel_u, rel_r = min(candidates, key=lambda x: (x[0], x[1], x[2], x[3]))
        bit_cfg[n] = nb
        current_cost -= saved
        steps.append({
            "name": n, "from": b, "to": nb, "saved": saved,
            "delta_u": delta_u, "rel_u": rel_u, "rel_r": rel_r,
            "score": score,
        })

    obj = sum(float(U[n][int(bit_cfg[n])]) for n in names)
    bucket_params = [sum(n_params[n] for n in names if int(bit_cfg[n]) == b) for b in bits]
    alloc_info = {
        "mode": "budget_greedy_v2",
        "bits": bits,
        "target_avg_bit": float(target_avg_bit),
        "avg_bit": current_cost / max(total_params, 1),
        "budget": budget,
        "cost": current_cost,
        "objective": obj,
        "num_demotions": len(steps),
        "rel_sensitivity_alpha": float(rel_alpha),
        "saved_power": float(saved_power),
        "bucket_params": bucket_params,
        "bucket_param_ratios_actual": [p / max(total_params, 1) for p in bucket_params],
        "steps_preview": steps[:80],
    }
    print("\n[Stage2] Budget-greedy-v2 automatic mixed-bit allocation")
    print(f"  bits={bits}, target_avg={target_avg_bit:.4f}, rel_alpha={rel_alpha}, saved_power={saved_power}")
    print(f"  avg_bit={alloc_info['avg_bit']:.4f}, objective={obj:.6e}, demotions={len(steps)}")
    print(f"  layer bit distribution={Counter(bit_cfg.values())}")
    print(f"  bucket_params={bucket_params}")
    print(f"  actual param ratios={[round(x, 4) for x in alloc_info['bucket_param_ratios_actual']]}")
    print("  demotion preview:")
    for st in steps[:20]:
        print(f"    {st['name']:<48s} {st['from']}->{st['to']} saved={st['saved']} rel_u={st['rel_u']:.3e} rel_r={st['rel_r']:.3e} score={st['score']:.3e}")
    return bit_cfg, alloc_info

def fixed_ratio_bit_allocation(
    U: Dict[str, Dict[int, float]],
    R: Dict[str, Dict[int, float]],
    n_params: Dict[str, int],
    bits: List[int],
    ref_bit: int,
    bit_param_ratios: str,
    target_avg_bit: float,
    max_lowest_bit_param_ratio: float = -1.0,
):
    """Assign bits by sorted task sensitivity and fixed parameter ratios.

    High Fisher sensitivity layers get higher bits. Low Fisher sensitivity layers get
    lower bits, forcing both high-bit and low-bit buckets to appear. This is more
    suitable for the current exploratory defense setting than a pure task-cost search,
    which tends to exclude 3-bit layers entirely.
    """
    bits = sorted([int(b) for b in bits], reverse=True)
    ratios = _parse_float_list(bit_param_ratios)
    if len(bits) != len(ratios):
        raise ValueError(f"bits={bits} but ratios={ratios}; lengths must match")
    ratios = np.asarray(ratios, dtype=np.float64)
    if np.any(ratios < 0):
        raise ValueError(f"bit_param_ratios must be non-negative: {ratios}")
    ratios = ratios / max(ratios.sum(), 1e-12)
    implied_avg = float(sum(b * r for b, r in zip(bits, ratios)))
    if abs(implied_avg - float(target_avg_bit)) > 0.05:
        print(f"[Stage2][WARN] bit_param_ratios imply avg_bit={implied_avg:.4f}, but target_avg_w_bit={target_avg_bit:.4f}.")
        print("               The fixed-ratio allocation will follow ratios; adjust --bit_param_ratios if this is unintended.")

    names = list(R.keys())
    if ref_bit not in bits:
        raise ValueError(f"ref_bit={ref_bit} not in bits={bits}")
    names.sort(key=lambda n: R[n][ref_bit], reverse=True)
    total_params = sum(n_params[n] for n in names)
    targets = ratios * total_params

    bit_cfg = OrderedDict()
    bucket_params = [0 for _ in bits]
    bucket_names = [[] for _ in bits]
    bi = 0
    cum_target = float(targets[0]) if len(targets) > 0 else float("inf")
    cum_seen = 0
    for name in names:
        # Move to next bucket only after current bucket has reached its target.
        if bi < len(bits) - 1 and cum_seen >= cum_target and bucket_params[bi] > 0:
            bi += 1
            cum_target += float(targets[bi])
        b = bits[bi]
        bit_cfg[name] = int(b)
        bucket_params[bi] += int(n_params[name])
        bucket_names[bi].append(name)
        cum_seen += int(n_params[name])

    # Budget correction: the layer-wise parameter sizes are uneven, so fixed
    # ratio buckets can overshoot the requested average bit.  Demote the least
    # sensitive layers one bit at a time until the average bit is at or below
    # the requested budget, while keeping at least one layer in the highest and
    # lowest bit buckets when possible.  This is important for controlled
    # comparisons at avg 4.5 bit.
    def _current_cost():
        return sum(n_params[n] * bit_cfg[n] for n in names)

    min_bit = min(bits)
    max_bit = max(bits)
    lower_bit = {bits[i]: bits[i + 1] for i in range(len(bits) - 1)}
    budget = float(target_avg_bit) * float(total_params)
    cost = _current_cost()
    demotions = []
    if max_lowest_bit_param_ratio is None or float(max_lowest_bit_param_ratio) < 0:
        lowest_cap_params = float("inf")
    else:
        lowest_cap_params = float(max_lowest_bit_param_ratio) * float(total_params)

    def _bit_param_sum(bit_value: int) -> int:
        return sum(int(n_params[n]) for n in names if int(bit_cfg[n]) == int(bit_value))

    while cost > budget + 1e-9:
        counts = Counter(bit_cfg.values())
        candidates = []
        cur_lowest_params = _bit_param_sum(min_bit)
        for n in names:
            b = bit_cfg[n]
            if b <= min_bit:
                continue
            if b == max_bit and counts[b] <= 1:
                continue
            nb = lower_bit[b]
            if nb == min_bit and cur_lowest_params + int(n_params[n]) > lowest_cap_params:
                # Keep 3-bit as a small probing component; do not use it as the
                # main budget-correction sink.
                continue
            saving = n_params[n] * (b - nb)
            # Prefer demoting lower-sensitivity layers first; among ties, larger
            # parameter layers correct the budget faster.
            candidates.append((R[n][ref_bit], -saving, n, b, nb, saving))
        if not candidates:
            print(f"[Stage2][WARN] could not further correct avg bit under lowest-bit cap; current={cost/max(total_params,1):.4f}, target={target_avg_bit:.4f}, lowest_cap={max_lowest_bit_param_ratio}")
            break
        _, _, n, b, nb, saving = min(candidates, key=lambda x: (x[0], x[1]))
        bit_cfg[n] = int(nb)
        cost -= saving
        demotions.append((n, b, nb, saving))

    obj = sum(U[n][bit_cfg[n]] for n in names)
    if demotions:
        print(f"[Stage2] budget correction demotions={len(demotions)}; avg_bit={cost/max(total_params,1):.4f} <= target={target_avg_bit:.4f}")
        for n, b, nb, saving in demotions[:12]:
            print(f"    demote {n:45s} {b}->{nb}, saving_params_bits={saving}")
        if len(demotions) > 12:
            print(f"    ... {len(demotions)-12} more demotions")

    # Recompute bucket statistics after budget correction.
    bucket_params = [sum(int(n_params[n]) for n in names if bit_cfg[n] == b) for b in bits]
    bucket_names = [[n for n in names if bit_cfg[n] == b] for b in bits]

    alloc_info = {
        "mode": "fixed_ratio",
        "obj": obj,
        "cost": cost,
        "budget": target_avg_bit * total_params,
        "avg_bit": cost / max(total_params, 1),
        "sorted_names": names,
        "bits": bits,
        "ratios": ratios.tolist(),
        "implied_avg_bit": implied_avg,
        "bucket_params": bucket_params,
        "bucket_param_ratios_actual": [bp / max(total_params, 1) for bp in bucket_params],
        "bucket_sizes": [len(x) for x in bucket_names],
        "bit_cfg": dict(bit_cfg),
    }
    print("\n[Stage2] Fixed-ratio mixed-bit allocation")
    print(f"  bits={bits}, target ratios={ratios.tolist()}, implied_avg={implied_avg:.4f}")
    print(f"  avg_bit={alloc_info['avg_bit']:.4f}, objective={obj:.6e}")
    print(f"  layer bit distribution={Counter(bit_cfg.values())}")
    print(f"  bucket_params={bucket_params}")
    print(f"  actual param ratios={alloc_info['bucket_param_ratios_actual']}")
    print("\n[Stage2] Sorted sensitivity and assigned bits:")
    for name in names:
        print(f"  {name:50s} R@{ref_bit}={R[name][ref_bit]:.3e} U@bit={U[name][bit_cfg[name]]:.3e} bit={bit_cfg[name]}")
    return dict(bit_cfg), alloc_info

# -----------------------------
# Bit-budget helpers
# -----------------------------
def resolve_bit_budgets(args) -> None:
    """Resolve convenience and separate W/A average-bit budgets in-place."""
    overall = float(getattr(args, "target_avg_bit", -1.0))
    w = float(getattr(args, "target_avg_w_bit", -1.0))
    a = float(getattr(args, "target_avg_a_bit", -1.0))
    if w <= 0:
        w = overall if overall > 0 else 4.0
    if a <= 0:
        a = overall if overall > 0 else w
    args.target_avg_w_bit = float(w)
    args.target_avg_a_bit = float(a)
    print(f"[Budget] target_avg_bit={overall if overall > 0 else 'unset'} | target_avg_w_bit={args.target_avg_w_bit:.4f} | target_avg_a_bit={args.target_avg_a_bit:.4f}")


def weighted_average_bits(bit_cfg: Dict[str, int], weights: Dict[str, int]) -> float:
    total = float(sum(int(weights.get(k, 0)) for k in bit_cfg.keys()))
    if total <= 0:
        return float('nan')
    return float(sum(int(weights.get(k, 0)) * int(v) for k, v in bit_cfg.items()) / total)


def bit_bucket_weights(bit_cfg: Dict[str, int], weights: Dict[str, int], bits: List[int]) -> List[int]:
    return [int(sum(int(weights.get(k, 0)) for k, v in bit_cfg.items() if int(v) == int(b))) for b in bits]


def compute_global_bit_summary(
    weight_bit_cfg: Dict[str, int],
    weight_size_weights: Dict[str, int],
    act_bit_cfg: Dict[str, int],
    act_size_weights: Dict[str, int],
    bits: List[int],
) -> Dict[str, float]:
    """Compute W/A-separated and overall average bit diagnostics.

    Definitions:
      - W average: parameter-count weighted average over quantized weight modules.
      - A average: activation-element-count weighted average over activation quantizers.
      - total_count_weighted: combines W parameter counts and per-sample activation element counts.
      - total_balanced: arithmetic mean of W average and A average; useful when one wants W and A to contribute equally as two budgets.

    The two total averages answer different questions, so both are printed and saved.
    """
    w_units = float(sum(int(weight_size_weights.get(k, 0)) for k in weight_bit_cfg.keys()))
    a_units = float(sum(int(act_size_weights.get(k, 0)) for k in act_bit_cfg.keys()))
    avg_w = weighted_average_bits(weight_bit_cfg, weight_size_weights)
    avg_a = weighted_average_bits(act_bit_cfg, act_size_weights)
    total_count = float('nan')
    if (w_units + a_units) > 0:
        total_count = float((avg_w * w_units + avg_a * a_units) / (w_units + a_units))
    total_balanced = float((avg_w + avg_a) / 2.0) if math.isfinite(avg_w) and math.isfinite(avg_a) else float('nan')
    w_bucket = bit_bucket_weights(weight_bit_cfg, weight_size_weights, bits)
    a_bucket = bit_bucket_weights(act_bit_cfg, act_size_weights, bits)
    return {
        "actual_avg_w_bit": float(avg_w),
        "actual_avg_a_bit": float(avg_a),
        "actual_avg_total_bit_count_weighted": float(total_count),
        "actual_avg_total_bit_balanced": float(total_balanced),
        "weight_units": int(w_units),
        "activation_units_per_sample": int(a_units),
        "weight_unit_fraction_count_weighted": float(w_units / max(w_units + a_units, 1.0)),
        "activation_unit_fraction_count_weighted": float(a_units / max(w_units + a_units, 1.0)),
        "weight_bucket_units": w_bucket,
        "activation_bucket_units": a_bucket,
        "weight_bucket_ratios": [float(x / max(sum(w_bucket), 1)) for x in w_bucket],
        "activation_bucket_ratios": [float(x / max(sum(a_bucket), 1)) for x in a_bucket],
        "bits_order": [int(b) for b in bits],
    }


def print_global_bit_summary(summary: Dict[str, float]) -> None:
    print("[BitSummary] actual_avg_w_bit={:.4f} | actual_avg_a_bit={:.4f} | total_count_weighted={:.4f} | total_balanced={:.4f}".format(
        summary.get("actual_avg_w_bit", float('nan')),
        summary.get("actual_avg_a_bit", float('nan')),
        summary.get("actual_avg_total_bit_count_weighted", float('nan')),
        summary.get("actual_avg_total_bit_balanced", float('nan')),
    ))
    print("[BitSummary] weight_units={} ({:.4f}) | activation_units_per_sample={} ({:.4f})".format(
        summary.get("weight_units", 0),
        summary.get("weight_unit_fraction_count_weighted", 0.0),
        summary.get("activation_units_per_sample", 0),
        summary.get("activation_unit_fraction_count_weighted", 0.0),
    ))
    print(f"[BitSummary] bits_order={summary.get('bits_order')}")
    print(f"[BitSummary] weight_bucket_ratios={[round(float(x), 4) for x in summary.get('weight_bucket_ratios', [])]}")
    print(f"[BitSummary] activation_bucket_ratios={[round(float(x), 4) for x in summary.get('activation_bucket_ratios', [])]}")


def _next_lower_bit(b: int, allowed_bits_desc: List[int]) -> Optional[int]:
    asc = sorted([int(x) for x in allowed_bits_desc])
    lows = [x for x in asc if x < int(b)]
    return max(lows) if lows else None


def _next_higher_bit(b: int, allowed_bits_desc: List[int]) -> Optional[int]:
    asc = sorted([int(x) for x in allowed_bits_desc])
    highs = [x for x in asc if x > int(b)]
    return min(highs) if highs else None


def rebalance_bit_cfg_to_target(
    bit_cfg: Dict[str, int],
    weights: Dict[str, int],
    allowed_bits_desc: List[int],
    target_avg_bit: float,
    min_bit: Optional[int] = None,
    max_steps: int = 100000,
) -> Tuple[Dict[str, int], Dict]:
    """Adjust a mixed-bit config to a weighted-average target by one-bit moves.

    This is deliberately generic.  For activations we start from the weight-derived
    bit pattern and make the smallest discrete moves needed to match the activation
    average budget, avoiding hand-written layer rules.
    """
    cfg = OrderedDict((k, int(v)) for k, v in bit_cfg.items())
    allowed = sorted([int(b) for b in allowed_bits_desc], reverse=True)
    if min_bit is not None:
        allowed = [b for b in allowed if b >= int(min_bit)]
    if not allowed:
        raise ValueError("No allowed bits left after applying min_bit")
    total_w = float(sum(int(weights.get(k, 0)) for k in cfg.keys()))
    if total_w <= 0:
        return dict(cfg), {"avg_bit": float('nan'), "steps": [], "warn": "zero weights"}

    def avg(c):
        return float(sum(int(weights.get(k, 0)) * int(v) for k, v in c.items()) / total_w)

    steps = []
    cur_avg = avg(cfg)
    target = float(target_avg_bit)
    for _ in range(max_steps):
        cur_avg = avg(cfg)
        if abs(cur_avg - target) < 1e-6:
            break
        direction = "down" if cur_avg > target else "up"
        candidates = []
        for name, b in cfg.items():
            w = int(weights.get(name, 0))
            if w <= 0:
                continue
            nb = _next_lower_bit(b, allowed) if direction == "down" else _next_higher_bit(b, allowed)
            if nb is None:
                continue
            new_avg = cur_avg + (int(nb) - int(b)) * w / total_w
            # Prefer moves that get closer to target. Tie-breaker: larger saved/added weight.
            candidates.append((abs(new_avg - target), -w, name, int(b), int(nb), float(new_avg)))
        if not candidates:
            break
        candidates.sort()
        _, negw, name, b, nb, new_avg = candidates[0]
        # Stop if the best move would make the average farther away.
        if abs(new_avg - target) > abs(cur_avg - target) + 1e-12:
            break
        cfg[name] = nb
        steps.append({"name": name, "from": b, "to": nb, "weight": int(-negw), "avg_after": new_avg})

    return dict(cfg), {
        "target_avg_bit": target,
        "avg_bit": avg(cfg),
        "num_steps": len(steps),
        "steps_preview": steps[:30],
    }


def estimate_activation_numels(model: nn.Module, loader, device: torch.device, max_batches: int = 1) -> Dict[str, int]:
    """Estimate per-activation-layer tensor sizes using forward hooks.

    The size is averaged per sample, so it can be used as an activation-memory
    weight when computing average activation bit-width.
    """
    sizes: Dict[str, int] = OrderedDict()
    handles = []

    def make_hook(name):
        def hook(_mod, _inp, out):
            if name in sizes:
                return
            if torch.is_tensor(out):
                bsz = int(out.shape[0]) if out.dim() > 0 else 1
                sizes[name] = int(out.numel() // max(bsz, 1))
            elif isinstance(out, (tuple, list)) and len(out) > 0 and torch.is_tensor(out[0]):
                t = out[0]
                bsz = int(t.shape[0]) if t.dim() > 0 else 1
                sizes[name] = int(t.numel() // max(bsz, 1))
        return hook

    for name, m in model.named_modules():
        if hasattr(m, "bits") and "ActQuant" in type(m).__name__:
            handles.append(m.register_forward_hook(make_hook(name)))
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for i, (x, _) in enumerate(loader):
                if i >= max_batches:
                    break
                x = x.to(device)
                _ = model(x)
    finally:
        for h in handles:
            h.remove()
        if was_training:
            model.train()
    # Fallback for layers not seen by hooks.
    for name, m in model.named_modules():
        if hasattr(m, "bits") and "ActQuant" in type(m).__name__ and name not in sizes:
            sizes[name] = 1
    return dict(sizes)


# -----------------------------
# Activation quant calibration / bit assignment
# -----------------------------
def collect_activation_bit_cfg(model: nn.Module) -> Dict[str, int]:
    act_bit_cfg = OrderedDict()
    for name, m in model.named_modules():
        if hasattr(m, "bits") and "ActQuant" in type(m).__name__:
            act_bit_cfg[name] = int(m.bits)
    return dict(act_bit_cfg)


def apply_activation_bit_cfg(model: nn.Module, act_bit_cfg: Dict[str, int]) -> None:
    modules = dict(model.named_modules())
    for name, bits in act_bit_cfg.items():
        m = modules.get(name, None)
        if m is not None and hasattr(m, "set_bits"):
            m.set_bits(int(bits))


def assign_activation_bits(model, bit_cfg: Dict[str, int], args, calib_loader=None, device: Optional[torch.device] = None) -> Dict[str, int]:
    bits_allowed = sorted([int(x.strip()) for x in str(args.bits).split(",") if x.strip()], reverse=True)
    min_act_bits = int(getattr(args, "min_act_bits", 3))
    if min_act_bits > 0:
        bits_allowed = [b for b in bits_allowed if b >= min_act_bits]
    if not bits_allowed:
        raise ValueError(f"No activation bit choices left after min_act_bits={min_act_bits}")

    if args.act_bit_mode == "from_weight" and hasattr(model, "assign_act_bits_from_weight_cfg"):
        print("[ActBits] mixed activation bits from weight bit_cfg")
        model.assign_act_bits_from_weight_cfg(bit_cfg, default_bits=args.act_bits)
    elif hasattr(model, "set_all_act_bits"):
        print(f"[ActBits] uniform activation bits: set_all_act_bits({args.act_bits})")
        model.set_all_act_bits(args.act_bits)
    else:
        print("[WARN] model has no activation bit assignment method. Skip.")

    # Clamp to allowed activation range first.
    for _, m in model.named_modules():
        if hasattr(m, "bits") and hasattr(m, "set_bits") and "ActQuant" in type(m).__name__:
            b = int(m.bits)
            b = min(max(b, min(bits_allowed)), max(bits_allowed))
            # Snap to the nearest available bit if needed.
            b = min(bits_allowed, key=lambda x: abs(int(x) - int(b)))
            m.set_bits(int(b))
    print(f"[ActBits] allowed activation bits={bits_allowed}, min_act_bits={min_act_bits}")

    act_bit_cfg = collect_activation_bit_cfg(model)
    act_size_weights = {}
    if calib_loader is not None and device is not None:
        act_size_weights = estimate_activation_numels(model, calib_loader, device, max_batches=1)
    if not act_size_weights:
        act_size_weights = {k: 1 for k in act_bit_cfg.keys()}

    init_avg = weighted_average_bits(act_bit_cfg, act_size_weights)
    target_a = float(getattr(args, "target_avg_a_bit", -1.0))
    print(f"[Stage2-A] initial activation avg_bit={init_avg:.4f}, target_avg_a_bit={target_a:.4f}")

    if target_a > 0:
        act_bit_cfg, act_alloc_info = rebalance_bit_cfg_to_target(
            act_bit_cfg,
            act_size_weights,
            allowed_bits_desc=bits_allowed,
            target_avg_bit=target_a,
            min_bit=min_act_bits,
        )
        apply_activation_bit_cfg(model, act_bit_cfg)
        print(f"[Stage2-A] rebalanced activation avg_bit={act_alloc_info['avg_bit']:.4f}, steps={act_alloc_info['num_steps']}")
        if act_alloc_info["steps_preview"]:
            print("[Stage2-A] rebalance preview:")
            for st in act_alloc_info["steps_preview"][:12]:
                print(f"  {st['name']:45s} {st['from']}->{st['to']} weight={st['weight']} avg_after={st['avg_after']:.4f}")
    else:
        act_alloc_info = {"target_avg_bit": target_a, "avg_bit": init_avg, "num_steps": 0, "steps_preview": []}

    act_bit_cfg = collect_activation_bit_cfg(model)
    final_avg = weighted_average_bits(act_bit_cfg, act_size_weights)
    print(f"[ActBits] activation bit distribution={Counter(act_bit_cfg.values())}")
    print(f"[ActBits] activation-size-weighted avg_bit={final_avg:.4f}")
    bucket_sizes = bit_bucket_weights(act_bit_cfg, act_size_weights, bits_allowed)
    total_size = max(sum(bucket_sizes), 1)
    print(f"[ActBits] bucket_activation_sizes={bucket_sizes}")
    print(f"[ActBits] bucket_activation_ratios={[round(x/total_size, 4) for x in bucket_sizes]}")

    # Stash diagnostics on args so they can be saved in the checkpoint without changing APIs everywhere.
    setattr(args, "_act_size_weights", dict(act_size_weights))
    setattr(args, "_act_alloc_info", dict(act_alloc_info))
    setattr(args, "_actual_avg_a_bit", float(final_avg))
    return act_bit_cfg


def calibrate_activations(model, loader, device: torch.device) -> None:
    if not all(hasattr(model, m) for m in ["reset_act_calib", "set_calib_mode", "finalize_act_calib"]):
        print("[WARN] model does not expose activation calibration APIs. Skip calibration.")
        return
    print("\n[ActCalib] reset -> calib forward -> finalize")
    model.reset_act_calib()
    model.set_calib_mode()
    model.eval()
    with torch.no_grad():
        for x, _ in tqdm(loader, desc="[ActCalib]", leave=False):
            x = x.to(device)
            _ = model(x)
    model.finalize_act_calib()


# -----------------------------
# Stage 3: class-aware scale optimization
# -----------------------------
def find_feature_module_name(model: nn.Module, requested: str = "auto") -> str:
    modules = dict(model.named_modules())
    if requested != "auto":
        if requested not in modules:
            raise RuntimeError(f"target_feature_module={requested} not found. Available examples: {list(modules.keys())[:30]}")
        return requested

    # Prefer avgpool for ResNet-style feature vector.
    if "avgpool" in modules:
        return "avgpool"
    if "layer4" in modules:
        return "layer4"

    # Otherwise choose last ActQuantLayerwise.
    act_names = [name for name, m in model.named_modules() if "ActQuantLayerwise" in type(m).__name__]
    if len(act_names) > 0:
        return act_names[-1]

    # Last non-root module fallback.
    all_names = [n for n, _ in model.named_modules() if n]
    if len(all_names) == 0:
        raise RuntimeError("No module found for feature hook.")
    return all_names[-1]


class FeatureHook:
    def __init__(self, model: nn.Module, module_name: str):
        self.model = model
        self.module_name = module_name
        self.features = None
        modules = dict(model.named_modules())
        if module_name not in modules:
            raise RuntimeError(f"Module {module_name} not found.")
        self.handle = modules[module_name].register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        if torch.is_tensor(out):
            self.features = out
        elif isinstance(out, (list, tuple)):
            self.features = next((o for o in out if torch.is_tensor(o)), None)
        else:
            self.features = None

    def close(self):
        self.handle.remove()


def _unpack_return_feats(out):
    if not isinstance(out, tuple):
        raise RuntimeError("model(x, return_feats=True) did not return a tuple")
    if len(out) == 2:
        feats, logp = out
    elif len(out) == 3:
        feats, _, logp = out
    elif len(out) == 4:
        feats, _, _, logp = out
    else:
        raise RuntimeError(f"Unexpected return_feats tuple length: {len(out)}")
    return feats, logp


def forward_target_feature(model: nn.Module, x: torch.Tensor, target_feature_idx: int,
                           hook: Optional[FeatureHook] = None):
    """Return target feature and log-probability for Stage-3 loss."""
    if target_feature_idx is not None and target_feature_idx >= 0:
        feats, logp = _unpack_return_feats(model(x, return_feats=True))
        if target_feature_idx >= len(feats):
            raise RuntimeError(f"target_feature_idx={target_feature_idx} but model returned {len(feats)} features")
        return feats[target_feature_idx], logp
    if hook is None:
        raise RuntimeError("hook is required when target_feature_idx < 0")
    hook.features = None
    logp = model(x)
    return hook.features, logp


@torch.no_grad()
def compute_fp32_class_prototypes(fp_model: nn.Module, loader, device: torch.device, args,
                                  target_name: str) -> torch.Tensor:
    """Compute full-training-set FP32 class prototypes on normalized target features."""
    print("\n[Stage3] Computing full-train FP32 class prototypes...")
    fp_model.eval()
    hook = None
    if args.target_feature_idx < 0:
        hook = FeatureHook(fp_model, target_name)
    sums = None
    counts = torch.zeros(args.nz, dtype=torch.float64)
    try:
        for x, y in tqdm(loader, desc="[Stage3] FP32 prototypes", leave=False):
            x = x.to(device)
            y = y.to(device).long()
            feat, _ = forward_target_feature(fp_model, x, args.target_feature_idx, hook)
            z = feat.view(feat.shape[0], -1).float()
            z = F.normalize(z, p=2, dim=1, eps=args.eps).detach().cpu().double()
            y_cpu = y.detach().cpu().long()
            if sums is None:
                sums = torch.zeros(args.nz, z.shape[1], dtype=torch.float64)
            sums.index_add_(0, y_cpu, z)
            ones = torch.ones_like(y_cpu, dtype=torch.float64)
            counts.index_add_(0, y_cpu, ones)
    finally:
        if hook is not None:
            hook.close()
    if sums is None:
        raise RuntimeError("No prototypes computed; empty loader?")
    valid = counts > 0
    protos = torch.zeros_like(sums)
    protos[valid] = sums[valid] / counts[valid].unsqueeze(1)
    protos[valid] = F.normalize(protos[valid].float(), p=2, dim=1, eps=args.eps).double()
    print(f"[Stage3] prototypes covered_classes={int(valid.sum().item())}/{args.nz}")
    return protos.float().to(device)


def prototype_feature_loss(features: torch.Tensor, labels: torch.Tensor, prototypes: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if features is None:
        raise RuntimeError("Target feature was not captured.")
    z = features.view(features.shape[0], -1)
    z = F.normalize(z, p=2, dim=1, eps=eps)
    target = prototypes[labels.long()]
    return ((z - target) ** 2).sum(dim=1).mean()


def scale_regularization_loss(model: nn.Module) -> torch.Tensor:
    losses = []
    for m in model.modules():
        if hasattr(m, "log_scale") and hasattr(m, "base_log_scale"):
            losses.append((m.log_scale - m.base_log_scale).pow(2).mean())
    if len(losses) == 0:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    return torch.stack(losses).mean()


def intra_class_feature_loss(features: torch.Tensor, labels: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Class-balanced intra-class dispersion on normalized features."""
    if features is None:
        raise RuntimeError("Feature hook did not capture features.")
    z = features.view(features.shape[0], -1)
    z = F.normalize(z, p=2, dim=1, eps=eps)

    losses = []
    for c in labels.unique():
        mask = labels == c
        if mask.sum() <= 1:
            continue
        zc = z[mask]
        mu = zc.mean(dim=0, keepdim=True)
        mu = F.normalize(mu, p=2, dim=1, eps=eps)
        losses.append(((zc - mu) ** 2).sum(dim=1).mean())

    if len(losses) == 0:
        # No class has >=2 samples in this batch. Return zero but keep graph.
        return z.sum() * 0.0
    return torch.stack(losses).mean()


def project_log_scales(model: nn.Module, rho: float) -> None:
    """Clamp log_scale around base_log_scale if modules expose both attributes."""
    with torch.no_grad():
        for m in model.modules():
            if hasattr(m, "log_scale") and hasattr(m, "base_log_scale"):
                m.log_scale.data.copy_(torch.max(torch.min(m.log_scale.data, m.base_log_scale + rho), m.base_log_scale - rho))


def get_scale_params(model: nn.Module) -> List[nn.Parameter]:
    """Return activation scale parameters, regardless of current requires_grad.

    Important: in class_aware_scale_optimization we freeze the whole model first
    and then re-enable only these parameters. Therefore this helper must NOT
    filter by requires_grad, otherwise it will return an empty list after the
    global freeze.
    """
    if hasattr(model, "act_scale_parameters"):
        params = list(model.act_scale_parameters())
    else:
        params = []
        for m in model.modules():
            if hasattr(m, "log_scale") and isinstance(m.log_scale, nn.Parameter):
                params.append(m.log_scale)
    # Remove duplicates while preserving order.
    seen = set()
    uniq = []
    for p in params:
        if id(p) not in seen:
            uniq.append(p)
            seen.add(id(p))
    return uniq




def collect_fp_weight_refs(model: nn.Module) -> Dict[str, torch.Tensor]:
    refs = OrderedDict()
    for name, m in iter_weight_modules(model):
        refs[name] = m.weight.detach().clone().cpu()
    return refs


def _soft_weight_qdq(module: nn.Module, hard: bool = False) -> torch.Tensor:
    bits = int(module._soft_w_bits)
    qmax = float(2 ** (bits - 1) - 1)
    qmin = float(-2 ** (bits - 1))
    base_scale = module._soft_w_base_scale.to(module._soft_w_qfloor.device)
    if base_scale.numel() == 1:
        shape = [1] * module.weight.dim()
    else:
        shape = [module.weight.shape[0]] + [1] * (module.weight.dim() - 1)
    scale = base_scale.view(*shape) * torch.exp(module._soft_w_log_scale_delta.view(*shape))
    temp = float(getattr(module, "_soft_w_temperature", 1.0))
    logits = module._soft_w_round_logits
    h_soft = torch.sigmoid(logits / max(temp, 1e-6))
    h_hard = (logits >= 0).float()

    # hard=True is used for finalization; _soft_w_eval_hard is used for hard-eval.
    # _soft_w_freeze_rounding means no rounding optimization, so use fixed nearest hard rounding.
    if hard or bool(getattr(module, "_soft_w_eval_hard", False)) or bool(getattr(module, "_soft_w_freeze_rounding", False)):
        h = h_hard
    elif str(getattr(module, "_soft_w_round_forward", "soft")) == "hard_ste":
        # Hard forward, soft backward. This avoids the misleading soft-to-hard gap.
        h = h_hard.detach() - h_soft.detach() + h_soft
    else:
        h = h_soft
    q = torch.clamp(module._soft_w_qfloor + h, qmin, qmax)
    return q * scale


def init_soft_weight_quantization(model: nn.Module, fp_weight_refs: Dict[str, torch.Tensor], bit_cfg: Dict[str, int], args) -> List[nn.Parameter]:
    """Patch Conv/Linear forward to use learnable soft-rounded quantized weights.

    This gives Stage 3 more control than activation-scale-only optimization:
      - weight log-scale delta controls weight quantization scale;
      - soft rounding logits control whether each value moves to lower/upper grid point.

    At the end of Stage 3, call finalize_soft_weight_quantization() to write hard
    q*s weights back into module.weight and restore ordinary Conv/Linear forward.
    """
    params: List[nn.Parameter] = []
    for name, m in iter_weight_modules(model):
        if name not in fp_weight_refs:
            continue
        bits = int(bit_cfg.get(name, min(int(v) for v in bit_cfg.values())))
        w_ref = fp_weight_refs[name].to(m.weight.device).float()
        C_out = w_ref.shape[0]
        w_view = w_ref.view(C_out, -1)
        qmax = float(2 ** (bits - 1) - 1)
        qmin = float(-2 ** (bits - 1))
        max_abs = w_view.abs().amax(dim=1, keepdim=True)
        base_scale = max_abs / max(qmax, 1.0)
        base_scale[base_scale < 1e-8] = 1.0
        w_norm = w_view / base_scale
        q_floor = torch.floor(w_norm)
        q_floor = torch.clamp(q_floor, qmin, qmax - 1.0)
        q_nearest = torch.round(w_norm).clamp(qmin, qmax)
        h0_hard = torch.clamp(q_nearest - q_floor, 0.0, 1.0)
        round_init_mode = str(getattr(args, "round_init_mode", "frac"))
        if round_init_mode == "frac":
            # Initialize from the fractional position between floor and ceil.
            # With hard-forward STE, thresholding at 0.5 gives ordinary nearest rounding,
            # but the logits are not saturated, so rounding directions can still be learned.
            h_frac = torch.clamp(w_norm - q_floor, 1e-4, 1.0 - 1e-4)
            logits = torch.log(h_frac / (1.0 - h_frac))
        else:
            init = float(args.weight_round_init_logit)
            logits = torch.where(h0_hard >= 0.5, torch.full_like(h0_hard, init), torch.full_like(h0_hard, -init))

        # Clean possible previous soft params if rerun in the same process.
        for attr in ["_soft_w_round_logits", "_soft_w_log_scale_delta"]:
            if hasattr(m, attr):
                delattr(m, attr)
        for attr in ["_soft_w_qfloor", "_soft_w_base_scale"]:
            if hasattr(m, attr):
                delattr(m, attr)

        m.register_buffer("_soft_w_qfloor", q_floor.view_as(w_ref).detach())
        m.register_buffer("_soft_w_base_scale", base_scale.squeeze(1).detach())
        m.register_parameter("_soft_w_round_logits", nn.Parameter(logits.view_as(w_ref).detach()))
        m.register_parameter("_soft_w_log_scale_delta", nn.Parameter(torch.zeros(C_out, device=w_ref.device)))
        m._soft_w_bits = bits
        m._soft_w_temperature = float(args.weight_round_temperature)
        m._soft_w_freeze_rounding = not bool(getattr(args, "optimize_weight_rounding", False))
        m._soft_w_round_forward = str(getattr(args, "weight_round_forward", "soft"))
        m._soft_w_eval_hard = False
        m._soft_w_is_conv = isinstance(m, nn.Conv2d)
        if not hasattr(m, "_soft_w_orig_forward"):
            m._soft_w_orig_forward = m.forward

        if isinstance(m, nn.Conv2d):
            def _conv_forward(x, module=m):
                return F.conv2d(x, _soft_weight_qdq(module, hard=False), module.bias, module.stride, module.padding, module.dilation, module.groups)
            m.forward = _conv_forward
        elif isinstance(m, nn.Linear):
            def _linear_forward(x, module=m):
                return F.linear(x, _soft_weight_qdq(module, hard=False), module.bias)
            m.forward = _linear_forward

        m.weight.requires_grad_(False)
        params.extend([m._soft_w_round_logits, m._soft_w_log_scale_delta])
    print(f"[Stage3][WeightOpt] enabled soft weight scale/rounding for {len(params)//2} modules, trainable tensors={len(params)}")
    return params


def split_weight_soft_params(model: nn.Module):
    round_params, scale_params = [], []
    for m in model.modules():
        if hasattr(m, "_soft_w_round_logits"):
            round_params.append(m._soft_w_round_logits)
        if hasattr(m, "_soft_w_log_scale_delta"):
            scale_params.append(m._soft_w_log_scale_delta)
    return round_params, scale_params


def weight_soft_regularization_loss(model: nn.Module):
    scale_losses, round_losses = [], []
    for m in model.modules():
        if hasattr(m, "_soft_w_log_scale_delta"):
            scale_losses.append(m._soft_w_log_scale_delta.pow(2).mean())
        if hasattr(m, "_soft_w_round_logits"):
            h = torch.sigmoid(m._soft_w_round_logits / max(float(getattr(m, "_soft_w_temperature", 1.0)), 1e-6))
            round_losses.append((h * (1.0 - h)).mean())
    dev = next(model.parameters()).device
    scale_reg = torch.stack(scale_losses).mean() if scale_losses else torch.tensor(0.0, device=dev)
    round_reg = torch.stack(round_losses).mean() if round_losses else torch.tensor(0.0, device=dev)
    return scale_reg, round_reg


def set_soft_weight_hard_eval(model: nn.Module, enabled: bool) -> None:
    for m in model.modules():
        if hasattr(m, "_soft_w_eval_hard"):
            m._soft_w_eval_hard = bool(enabled)


def anneal_soft_round_temperature(model: nn.Module, anneal: float = 1.0, min_temp: float = 0.25) -> None:
    if anneal >= 0.999999:
        return
    for m in model.modules():
        if hasattr(m, "_soft_w_temperature"):
            m._soft_w_temperature = max(float(min_temp), float(m._soft_w_temperature) * float(anneal))




def parse_stage3_lr_schedule(args):
    """Return a sorted list of (start_epoch, base_lr) for Stage3. Epoch starts are 1-based."""
    raw = str(getattr(args, "scale_lr_schedule", "") or "").strip()
    if not raw:
        return [(1, float(getattr(args, "scale_lr", 3e-5)))]
    vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        return [(1, float(getattr(args, "scale_lr", 3e-5)))]
    raw_ms = str(getattr(args, "scale_lr_milestones", "") or "").strip()
    if raw_ms:
        starts = [int(x.strip()) for x in raw_ms.split(",") if x.strip()]
        if len(starts) != len(vals):
            raise ValueError(
                f"--scale_lr_milestones length ({len(starts)}) must match "
                f"--scale_lr_schedule length ({len(vals)})."
            )
    else:
        # Generic fallback: split the total epoch range into equal contiguous segments.
        total = int(getattr(args, "scale_epochs", 1))
        starts = [1 + int(round(i * total / len(vals))) for i in range(len(vals))]
        starts[0] = 1
    pairs = sorted((max(1, int(st)), float(lr)) for st, lr in zip(starts, vals))
    if pairs[0][0] != 1:
        pairs.insert(0, (1, pairs[0][1]))
    return pairs


def stage3_lr_for_epoch(schedule, epoch):
    lr = schedule[0][1]
    for st, val in schedule:
        if epoch >= st:
            lr = val
        else:
            break
    return float(lr)


def apply_stage3_lr(optimizer, base_lr):
    for pg in optimizer.param_groups:
        mult = float(pg.get("lr_mult", 1.0))
        pg["lr"] = float(base_lr) * mult

def weight_rounding_stats(model: nn.Module) -> Dict[str, float]:
    vals, temps = [], []
    for m in model.modules():
        if hasattr(m, "_soft_w_round_logits"):
            temp = max(float(getattr(m, "_soft_w_temperature", 1.0)), 1e-6)
            h = torch.sigmoid(m._soft_w_round_logits.detach() / temp)
            vals.append(h.reshape(-1).float().cpu())
            temps.append(float(temp))
    if not vals:
        return {"round_mean": 0.0, "round_binary_frac": 0.0, "round_entropy": 0.0, "round_temp": 0.0}
    h = torch.cat(vals, dim=0).clamp(1e-6, 1 - 1e-6)
    binary_frac = ((h < 0.05) | (h > 0.95)).float().mean().item()
    entropy = (-(h * h.log() + (1 - h) * (1 - h).log())).mean().item()
    return {
        "round_mean": float(h.mean().item()),
        "round_binary_frac": float(binary_frac),
        "round_entropy": float(entropy),
        "round_temp": float(sum(temps) / max(len(temps), 1)),
    }


def summarize_stage3_trainables(model: nn.Module) -> None:
    act_names, w_scale_names, round_names = [], [], []
    for name, m in model.named_modules():
        if hasattr(m, "log_scale") and isinstance(getattr(m, "log_scale"), nn.Parameter):
            act_names.append(name)
        if hasattr(m, "_soft_w_log_scale_delta"):
            w_scale_names.append(name)
        if hasattr(m, "_soft_w_round_logits") and getattr(m._soft_w_round_logits, "requires_grad", False):
            round_names.append(name)
    def preview(xs, k=6):
        if len(xs) <= 2 * k:
            return xs
        return xs[:k] + ["..."] + xs[-k:]
    print("[Stage3][TrainableSummary]")
    print(f"  activation scale modules: {len(act_names)} {preview(act_names)}")
    print(f"  weight scale modules:     {len(w_scale_names)} {preview(w_scale_names)}")
    print(f"  weight rounding modules:  {len(round_names)} {preview(round_names)}")


def project_weight_log_scales(model: nn.Module, rho: float) -> None:
    with torch.no_grad():
        for m in model.modules():
            if hasattr(m, "_soft_w_log_scale_delta"):
                m._soft_w_log_scale_delta.data.clamp_(min=-float(rho), max=float(rho))


def finalize_soft_weight_quantization(model: nn.Module, hard: bool = True) -> None:
    """Write hard/soft q*s weights into module.weight and restore original forward.

    After this call, the model state_dict is loadable by the ordinary resnet50_ptq
    architecture; temporary soft-rounding parameters are deleted.
    """
    finalized = 0
    for name, m in iter_weight_modules(model):
        if not hasattr(m, "_soft_w_round_logits"):
            continue
        with torch.no_grad():
            w_q = _soft_weight_qdq(m, hard=hard).detach()
            m.weight.data.copy_(w_q)
            scale = m._soft_w_base_scale.to(m.weight.device) * torch.exp(m._soft_w_log_scale_delta.detach())
            if not hasattr(m, "weight_scale"):
                m.register_buffer("weight_scale", torch.ones_like(scale.detach()))
            try:
                m.weight_scale.copy_(scale.detach().view(-1))
            except Exception:
                pass
        if hasattr(m, "_soft_w_orig_forward"):
            m.forward = m._soft_w_orig_forward
            delattr(m, "_soft_w_orig_forward")
        for attr in ["_soft_w_round_logits", "_soft_w_log_scale_delta", "_soft_w_qfloor", "_soft_w_base_scale"]:
            if hasattr(m, attr):
                delattr(m, attr)
        for attr in ["_soft_w_bits", "_soft_w_temperature", "_soft_w_freeze_rounding", "_soft_w_round_forward", "_soft_w_eval_hard", "_soft_w_is_conv"]:
            if hasattr(m, attr):
                delattr(m, attr)
        finalized += 1
    print(f"[Stage3][WeightOpt] finalized hard soft-rounded weights for {finalized} modules")


@torch.no_grad()
def compute_fp32_output_prototypes(fp_model: nn.Module, loader, device: torch.device, args) -> torch.Tensor:
    print("\n[Stage3] Computing full-train FP32 output confidence prototypes...")
    fp_model.eval()
    sums = torch.zeros(args.nz, args.nz, dtype=torch.float64)
    counts = torch.zeros(args.nz, dtype=torch.float64)
    for x, y in tqdm(loader, desc="[Stage3] output prototypes", leave=False):
        x = x.to(device)
        y = y.to(device).long()
        p = fp_model(x).exp().detach().cpu().double()
        y_cpu = y.detach().cpu().long()
        sums.index_add_(0, y_cpu, p)
        counts.index_add_(0, y_cpu, torch.ones_like(y_cpu, dtype=torch.float64))
    valid = counts > 0
    protos = torch.zeros_like(sums)
    protos[valid] = sums[valid] / counts[valid].unsqueeze(1)
    protos = protos.clamp_min(1e-12)
    protos = protos / protos.sum(dim=1, keepdim=True).clamp_min(1e-12)

    alpha = float(getattr(args, "output_proto_sharpen", 1.0))
    if abs(alpha - 1.0) > 1e-12:
        protos = protos.pow(alpha)
        protos = protos / protos.sum(dim=1, keepdim=True).clamp_min(1e-12)

    eta = float(getattr(args, "output_proto_eta", 0.0))
    if eta > 0:
        eye = torch.eye(args.nz, dtype=protos.dtype)
        protos = (1.0 - eta) * protos + eta * eye
        protos = protos.clamp_min(1e-12)
        protos = protos / protos.sum(dim=1, keepdim=True).clamp_min(1e-12)

    print(f"[Stage3] output prototypes covered_classes={int(valid.sum().item())}/{args.nz}, sharpen={alpha}, onehot_eta={eta}")
    return protos.float().to(device)


def output_prototype_loss(logp_q: torch.Tensor, labels: torch.Tensor, out_prototypes: torch.Tensor) -> torch.Tensor:
    target = out_prototypes[labels.long()]
    return F.kl_div(logp_q, target, reduction="batchmean")


def output_intra_class_loss(logp_q: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    p = logp_q.exp()
    losses = []
    for c in labels.unique():
        mask = labels == c
        if mask.sum() <= 1:
            continue
        pc = p[mask]
        mu = pc.mean(dim=0, keepdim=True)
        losses.append(((pc - mu) ** 2).sum(dim=1).mean())
    if len(losses) == 0:
        return p.sum() * 0.0
    return torch.stack(losses).mean()


def _intra_dispersion_from_vectors(v: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mean class-balanced squared distance to within-batch class center."""
    losses = []
    for c in labels.unique():
        mask = labels == c
        if mask.sum() <= 1:
            continue
        vc = v[mask]
        mu = vc.mean(dim=0, keepdim=True)
        losses.append(((vc - mu) ** 2).sum(dim=1).mean())
    if len(losses) == 0:
        return v.sum() * 0.0
    return torch.stack(losses).mean()


def output_intra_temp_loss(logp_q: torch.Tensor, labels: torch.Tensor, temp: float = 4.0) -> torch.Tensor:
    """Output intra loss on softened probabilities to avoid near-zero saturated gradients."""
    temp = max(float(temp), 1e-6)
    p = torch.softmax(logp_q / temp, dim=1)
    return _intra_dispersion_from_vectors(p, labels)


def output_logp_intra_loss(logp_q: torch.Tensor, labels: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Intra consistency on normalized log-probability vectors."""
    v = logp_q - logp_q.mean(dim=1, keepdim=True)
    v = F.normalize(v, p=2, dim=1, eps=eps)
    return _intra_dispersion_from_vectors(v, labels)


def output_contract_loss(logp_q: torch.Tensor, logp_fp: torch.Tensor, labels: torch.Tensor,
                         rho: float = 0.95, temp: float = 4.0, eps: float = 1e-8):
    """Directly require Q output intra distance to be smaller than FP batch intra distance.

    Uses softened probabilities, because train-set FP outputs are often nearly one-hot and
    raw probability distances can be numerically tiny.
    """
    temp = max(float(temp), 1e-6)
    p_q = torch.softmax(logp_q / temp, dim=1)
    with torch.no_grad():
        p_fp = torch.softmax(logp_fp / temp, dim=1)
        d_fp = _intra_dispersion_from_vectors(p_fp, labels).detach()
    d_q = _intra_dispersion_from_vectors(p_q, labels)
    ratio = d_q / (d_fp + eps)
    return torch.relu(ratio - float(rho)).pow(2), d_q, d_fp, ratio.detach()


def feature_contract_loss(feat_q: torch.Tensor, feat_fp: torch.Tensor, labels: torch.Tensor,
                          rho: float = 0.98, eps: float = 1e-8):
    """Directly require Q target-feature intra distance to be smaller than FP batch baseline."""
    zq = feat_q.view(feat_q.shape[0], -1)
    zq = F.normalize(zq, p=2, dim=1, eps=eps)
    with torch.no_grad():
        zfp = feat_fp.view(feat_fp.shape[0], -1)
        zfp = F.normalize(zfp, p=2, dim=1, eps=eps)
        d_fp = _intra_dispersion_from_vectors(zfp, labels).detach()
    d_q = _intra_dispersion_from_vectors(zq, labels)
    ratio = d_q / (d_fp + eps)
    return torch.relu(ratio - float(rho)).pow(2), d_q, d_fp, ratio.detach()



def _inter_center_dispersion_from_vectors(v: torch.Tensor, labels: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Mean squared pairwise distance between batch class centers.

    This is differentiable w.r.t. v. It is intentionally computed on the current
    class-balanced batch, because Stage-3 scale optimization cannot afford a full
    training-set pass at every step.
    """
    classes = labels.unique()
    centers = []
    for c in classes:
        mask = labels == c
        if mask.sum() <= 0:
            continue
        centers.append(v[mask].mean(dim=0))
    if len(centers) <= 1:
        return v.sum() * 0.0
    centers = torch.stack(centers, dim=0)
    diff = centers[:, None, :] - centers[None, :, :]
    dist2 = (diff ** 2).sum(dim=-1)
    mask = ~torch.eye(dist2.shape[0], dtype=torch.bool, device=dist2.device)
    return dist2[mask].mean()


def _make_inter_objective(ratio: torch.Tensor, mode: str = "ratio", rho: float = 0.90,
                          target: float = 0.80) -> torch.Tensor:
    """Differentiable objective for inter-class contraction.

    - hinge  : old behavior, only active when ratio > rho.
    - ratio  : continuously minimizes the ratio, so it never becomes a zero-gradient diagnostic.
    - target : pulls ratio toward a desired target band center; useful when avoiding excessive collapse.
    """
    if mode == "hinge":
        return torch.relu(ratio - float(rho)).pow(2)
    if mode == "target":
        return (ratio - float(target)).pow(2)
    # default: ratio minimization. Squared ratio gives stronger gradient when ratio is still large.
    return ratio.pow(2)


def output_inter_contract_loss(logp_q: torch.Tensor, logp_fp: torch.Tensor, labels: torch.Tensor,
                               rho: float = 0.90, temp: float = 1.0, eps: float = 1e-8,
                               mode: str = "ratio", target: float = 0.80):
    """Output-level class-ambiguity objective.

    v11 changes this from a mostly diagnostic hinge loss into an active objective. In
    `ratio` mode it continuously shrinks the distance between class-center output
    distributions. CE / margin / KL-hinge losses should prevent trivial uniform collapse.
    """
    temp = max(float(temp), 1e-6)
    p_q = torch.softmax(logp_q / temp, dim=1)
    with torch.no_grad():
        p_fp = torch.softmax(logp_fp / temp, dim=1)
        d_fp = _inter_center_dispersion_from_vectors(p_fp, labels, eps=eps).detach()
    d_q = _inter_center_dispersion_from_vectors(p_q, labels, eps=eps)
    ratio = d_q / (d_fp + eps)
    obj = _make_inter_objective(ratio, mode=mode, rho=rho, target=target)
    return obj, d_q, d_fp, ratio.detach()


def feature_inter_contract_loss(feat_q: torch.Tensor, feat_fp: torch.Tensor, labels: torch.Tensor,
                                rho: float = 0.95, eps: float = 1e-8,
                                mode: str = "ratio", target: float = 0.85):
    """Target-feature class-ambiguity objective.

    Feature-level inter contraction is optional. It is useful for testing whether
    semantic/identity mixing appears in late representations, but should usually be
    weaker than output-level inter contraction to avoid destroying utility.
    """
    zq = feat_q.view(feat_q.shape[0], -1)
    zq = F.normalize(zq, p=2, dim=1, eps=eps)
    with torch.no_grad():
        zfp = feat_fp.view(feat_fp.shape[0], -1)
        zfp = F.normalize(zfp, p=2, dim=1, eps=eps)
        d_fp = _inter_center_dispersion_from_vectors(zfp, labels, eps=eps).detach()
    d_q = _inter_center_dispersion_from_vectors(zq, labels, eps=eps)
    ratio = d_q / (d_fp + eps)
    obj = _make_inter_objective(ratio, mode=mode, rho=rho, target=target)
    return obj, d_q, d_fp, ratio.detach()




def _inter_retention_objective(d_current: torch.Tensor, d_pre: torch.Tensor, eps: float = 1e-8):
    """Adaptive no-threshold retention: penalize only if Stage-3 increases inter distance beyond pre-Stage3 Q.

    This does not impose a dataset-specific target such as 0.88.  It simply says:
    Stage-3 may recover utility, but should not undo the class ambiguity produced
    by the bit-width reduction.  The log form is scale-stable.
    """
    ratio_to_pre = d_current / (d_pre + eps)
    obj = torch.relu(torch.log(ratio_to_pre + eps)).pow(2)
    return obj, ratio_to_pre.detach()


def output_inter_retention_loss(logp_q: torch.Tensor, logp_pre: torch.Tensor, labels: torch.Tensor,
                                temp: float = 1.0, eps: float = 1e-8):
    temp = max(float(temp), 1e-6)
    p_q = torch.softmax(logp_q / temp, dim=1)
    with torch.no_grad():
        p_pre = torch.softmax(logp_pre / temp, dim=1)
        d_pre = _inter_center_dispersion_from_vectors(p_pre, labels, eps=eps).detach()
    d_q = _inter_center_dispersion_from_vectors(p_q, labels, eps=eps)
    obj, ratio_pre = _inter_retention_objective(d_q, d_pre, eps=eps)
    return obj, d_q, d_pre, ratio_pre


def feature_inter_retention_loss(feat_q: torch.Tensor, feat_pre: torch.Tensor, labels: torch.Tensor,
                                 eps: float = 1e-8):
    zq = feat_q.view(feat_q.shape[0], -1)
    zq = F.normalize(zq, p=2, dim=1, eps=eps)
    with torch.no_grad():
        zp = feat_pre.view(feat_pre.shape[0], -1)
        zp = F.normalize(zp, p=2, dim=1, eps=eps)
        d_pre = _inter_center_dispersion_from_vectors(zp, labels, eps=eps).detach()
    d_q = _inter_center_dispersion_from_vectors(zq, labels, eps=eps)
    obj, ratio_pre = _inter_retention_objective(d_q, d_pre, eps=eps)
    return obj, d_q, d_pre, ratio_pre


def label_smoothing_nll_loss(logp: torch.Tensor, labels: torch.Tensor, smoothing: float = 0.0) -> torch.Tensor:
    """
    Generalized label-smoothing CE.

    smoothing > 0: standard uniform label smoothing.
    smoothing = 0: ordinary NLL / CE.
    smoothing < 0: negative label smoothing / label sharpening.

    For negative smoothing this is no longer a proper probability-target CE,
    but is intentionally supported as an experimental anti-smoothing loss.
    It follows the same algebraic form as common label smoothing:
        (1 - eps) * NLL + eps * mean(-log p_j).
    """
    smoothing = float(smoothing)
    y = labels.long()
    nll = -logp.gather(1, y.view(-1, 1)).squeeze(1)
    if abs(smoothing) < 1e-12:
        return nll.mean()
    smooth = -logp.mean(dim=1)
    return ((1.0 - smoothing) * nll + smoothing * smooth).mean()

def classification_margin_loss(logp_q: torch.Tensor, labels: torch.Tensor, margin_target: float = 2.0) -> torch.Tensor:
    """Keep a minimum true-class log-probability margin to preserve utility."""
    y = labels.long()
    true_logp = logp_q.gather(1, y.view(-1, 1)).squeeze(1)
    masked = logp_q.clone()
    masked.scatter_(1, y.view(-1, 1), -1e9)
    max_other = masked.max(dim=1).values
    margin = true_logp - max_other
    return torch.relu(float(margin_target) - margin).pow(2).mean()


def inter_class_geometry_loss(features: torch.Tensor, labels: torch.Tensor, fp_prototypes: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Preserve target-layer class-center geometry in the current balanced batch."""
    z = features.view(features.shape[0], -1)
    z = F.normalize(z, p=2, dim=1, eps=eps)
    classes = labels.unique()
    if classes.numel() <= 1:
        return z.sum() * 0.0
    q_centers, fp_centers = [], []
    for c in classes:
        mask = labels == c
        if mask.sum() <= 1:
            continue
        qc = z[mask].mean(dim=0)
        qc = F.normalize(qc, p=2, dim=0, eps=eps)
        q_centers.append(qc)
        fp_centers.append(fp_prototypes[c.long()])
    if len(q_centers) <= 1:
        return z.sum() * 0.0
    q_centers = torch.stack(q_centers, dim=0)
    fp_centers = F.normalize(torch.stack(fp_centers, dim=0), p=2, dim=1, eps=eps)
    sim_q = q_centers @ q_centers.t()
    sim_fp = fp_centers @ fp_centers.t()
    mask = ~torch.eye(sim_q.shape[0], dtype=torch.bool, device=sim_q.device)
    return ((sim_q[mask] - sim_fp[mask]) ** 2).mean()

def class_aware_scale_optimization(
    q_model: nn.Module,
    fp_model: nn.Module,
    calib_loader,
    scale_loader,
    test_loader,
    device: torch.device,
    args,
    bit_cfg: Optional[Dict[str, int]] = None,
    fp_weight_refs: Optional[Dict[str, torch.Tensor]] = None,
):
    if hasattr(q_model, "_set_all_act_quant_mode"):
        q_model._set_all_act_quant_mode("fake")
    if hasattr(q_model, "set_ptq_scale_optim_mode"):
        q_model.set_ptq_scale_optim_mode()

    # Freeze all ordinary weights; scale params should be enabled by set_ptq_scale_optim_mode.
    for p in q_model.parameters():
        p.requires_grad_(False)
    scale_params = get_scale_params(q_model)
    for p in scale_params:
        p.requires_grad_(True)
    if len(scale_params) == 0:
        n_log_scale = sum(1 for m in q_model.modules() if hasattr(m, "log_scale"))
        n_act_like = sum(1 for m in q_model.modules() if "ActQuant" in type(m).__name__)
        raise RuntimeError(
            "No activation scale parameters found. "
            f"modules_with_log_scale={n_log_scale}, act_quant_like_modules={n_act_like}. "
            "Check whether resnet50_ptq inserts ActQuantLayerwise modules and whether "
            "act_scale_parameters() returns their log_scale parameters."
        )

    target_name = find_feature_module_name(q_model, args.target_feature_module)
    if args.target_feature_idx >= 0:
        print(f"\n[Stage3] Target returned feature index for losses: {args.target_feature_idx}")
        feat_hook = None
    else:
        print(f"\n[Stage3] Target feature module for losses: {target_name}")
        feat_hook = FeatureHook(q_model, target_name)

    fp_model.to(device).eval()
    q_model.to(device)

    # Pre-Stage3 quantized reference.  v13 uses this adaptive reference to retain
    # the class ambiguity produced by lowering bit-width.  This avoids dataset-specific
    # hand thresholds: Stage-3 is only penalized when it increases inter-class distance
    # beyond the pure quantized model on the same batch.
    q_pre_model = copy.deepcopy(q_model).to(device).eval()
    for p in q_pre_model.parameters():
        p.requires_grad_(False)

    prototypes = None
    if args.lambda_proto > 0 or args.lambda_inter > 0:
        prototypes = compute_fp32_class_prototypes(fp_model, calib_loader, device, args, target_name)
    out_prototypes = None
    if args.lambda_out_proto > 0:
        out_prototypes = compute_fp32_output_prototypes(fp_model, calib_loader, device, args)

    round_params, weight_scale_params = [], []
    if args.optimize_weight_scale or args.optimize_weight_rounding:
        if bit_cfg is None or fp_weight_refs is None:
            print("[Stage3][WeightOpt][WARN] bit_cfg or fp_weight_refs missing; skip weight scale/round optimization.")
        else:
            init_soft_weight_quantization(q_model, fp_weight_refs, bit_cfg, args)
            round_params, weight_scale_params = split_weight_soft_params(q_model)
            if not args.optimize_weight_rounding:
                for p in round_params:
                    p.requires_grad_(False)
                round_params = []

    summarize_stage3_trainables(q_model)

    lr_schedule = parse_stage3_lr_schedule(args)
    print(f"[Stage3][LR] schedule={lr_schedule}; weight_scale_lr_mult={float(args.weight_scale_lr_mult):.4g}")
    init_base_lr = stage3_lr_for_epoch(lr_schedule, 1)
    param_groups = [{"params": scale_params, "lr": init_base_lr, "lr_mult": 1.0, "name": "activation_scale"}]
    if round_params:
        param_groups.append({"params": round_params, "lr": init_base_lr, "lr_mult": 1.0, "name": "weight_rounding"})
    if weight_scale_params:
        param_groups.append({"params": weight_scale_params, "lr": init_base_lr * args.weight_scale_lr_mult, "lr_mult": args.weight_scale_lr_mult, "name": "weight_scale"})
    optimizer = torch.optim.Adam(param_groups)

    best_hard_acc = -1.0
    best_hard_state = None
    best_hard_epoch = -1

    try:
        for epoch in range(1, args.scale_epochs + 1):
            current_base_lr = stage3_lr_for_epoch(lr_schedule, epoch)
            apply_stage3_lr(optimizer, current_base_lr)
            q_model.train()
            running = collections.defaultdict(float)
            nb = 0
            in_phase1 = epoch <= int(getattr(args, "phase1_epochs", 0))

            for b_idx, (x, y) in enumerate(tqdm(scale_loader, desc=f"[Stage3] epoch {epoch}", leave=False)):
                if args.calib_batches_scale and b_idx >= args.calib_batches_scale:
                    break
                x = x.to(device)
                y = y.to(device).long()

                with torch.no_grad():
                    feat_fp, logp_fp = forward_target_feature(fp_model, x, args.target_feature_idx, None)
                    p_fp = logp_fp.exp()
                    feat_pre, logp_pre = forward_target_feature(q_pre_model, x, args.target_feature_idx, None)

                feat_q, logp_q = forward_target_feature(q_model, x, args.target_feature_idx, feat_hook)
                intra = intra_class_feature_loss(feat_q, y, eps=args.eps)
                proto = prototype_feature_loss(feat_q, y, prototypes, eps=args.eps) if prototypes is not None else (intra * 0.0)
                feat_contract, feat_intra_q, feat_intra_fp, feat_ratio = feature_contract_loss(
                    feat_q, feat_fp, y, rho=args.rho_feat, eps=args.eps
                ) if args.lambda_feat_contract > 0 else (intra * 0.0, intra.detach(), intra.detach(), torch.tensor(0.0, device=device))

                kl = F.kl_div(logp_q, p_fp, reduction="batchmean")
                if str(getattr(args, "kl_loss_mode", "raw")) == "raw":
                    kl_obj = kl
                else:
                    kl_obj = torch.relu(kl - args.out_kl_eps) ** 2
                nll = F.nll_loss(logp_q, y)
                task_ce = label_smoothing_nll_loss(logp_q, y, smoothing=float(getattr(args, "label_smoothing", 0.0)))
                # v13 default: raw label-smoothed CE for accuracy recovery. Hinge remains
                # available for ablation, but is not the main generic setting.
                if in_phase1 or str(getattr(args, "nll_loss_mode", "raw")) == "raw":
                    nll_obj = task_ce
                else:
                    nll_obj = torch.relu(task_ce - float(getattr(args, "ce_eps", 0.25))).pow(2)
                out_proto = output_prototype_loss(logp_q, y, out_prototypes) if out_prototypes is not None else (nll * 0.0)
                out_intra = output_intra_temp_loss(logp_q, y, temp=args.output_temp) if args.lambda_out_intra > 0 else (nll * 0.0)
                out_logp_intra = output_logp_intra_loss(logp_q, y, eps=args.eps) if args.lambda_logp_intra > 0 else (nll * 0.0)
                out_contract, out_intra_q, out_intra_fp, out_ratio = output_contract_loss(
                    logp_q, logp_fp, y, rho=args.rho_out, temp=args.output_temp, eps=args.eps
                ) if args.lambda_out_contract > 0 else (nll * 0.0, out_intra.detach(), out_intra.detach(), torch.tensor(0.0, device=device))

                # Always compute active model inter-distance metrics against FP32 for diagnostics.
                # The corresponding loss only contributes when lambda_*_inter_contract > 0.
                out_inter_contract, out_inter_q, out_inter_fp, out_inter_ratio = output_inter_contract_loss(
                    logp_q, logp_fp, y, rho=args.rho_out_inter, temp=args.inter_temp, eps=args.eps,
                    mode=args.inter_loss_mode, target=args.out_inter_target
                )

                feat_inter_contract, feat_inter_q, feat_inter_fp, feat_inter_ratio = feature_inter_contract_loss(
                    feat_q, feat_fp, y, rho=args.rho_feat_inter, eps=args.eps,
                    mode=args.inter_loss_mode, target=args.feat_inter_target
                )

                out_inter_retain, out_inter_cur_for_retain, out_inter_pre, out_inter_ratio_pre = output_inter_retention_loss(
                    logp_q, logp_pre, y, temp=args.inter_temp, eps=args.eps
                ) if getattr(args, "lambda_out_inter_retain", 0.0) > 0 else (nll * 0.0, out_inter_q.detach(), out_inter_q.detach(), torch.tensor(0.0, device=device))

                feat_inter_retain, feat_inter_cur_for_retain, feat_inter_pre, feat_inter_ratio_pre = feature_inter_retention_loss(
                    feat_q, feat_pre, y, eps=args.eps
                ) if getattr(args, "lambda_feat_inter_retain", 0.0) > 0 else (intra * 0.0, feat_inter_q.detach(), feat_inter_q.detach(), torch.tensor(0.0, device=device))

                margin = classification_margin_loss(logp_q, y, margin_target=args.margin_target) if args.lambda_margin > 0 else (nll * 0.0)

                inter = inter_class_geometry_loss(feat_q, y, prototypes, eps=args.eps) if (prototypes is not None and args.lambda_inter > 0) else (intra * 0.0)
                scale_reg = scale_regularization_loss(q_model)
                w_scale_reg, round_reg = weight_soft_regularization_loss(q_model)

                # Two-stage optimization:
                #   Phase A: recover utility and calibration; disable active inter contraction.
                #   Phase B: keep utility by hinge/constraint terms while actively shrinking inter distance.
                lambda_out_inter_eff = 0.0 if in_phase1 else float(args.lambda_out_inter_contract)
                lambda_feat_inter_eff = 0.0 if in_phase1 else float(args.lambda_feat_inter_contract)

                loss = (
                    args.lambda_intra * intra
                    + args.lambda_proto * proto
                    + args.lambda_feat_contract * feat_contract
                    + args.lambda_out_proto * out_proto
                    + args.lambda_out_intra * out_intra
                    + args.lambda_logp_intra * out_logp_intra
                    + args.lambda_out_contract * out_contract
                    + lambda_out_inter_eff * out_inter_contract
                    + lambda_feat_inter_eff * feat_inter_contract
                    + getattr(args, "lambda_out_inter_retain", 0.0) * out_inter_retain
                    + getattr(args, "lambda_feat_inter_retain", 0.0) * feat_inter_retain
                    + args.lambda_inter * inter
                    + args.lambda_kl * kl_obj
                    + args.lambda_nll * nll_obj
                    + args.lambda_margin * margin
                    + args.lambda_scale_reg * scale_reg
                    + args.lambda_weight_scale_reg * w_scale_reg
                    + args.lambda_round_reg * round_reg
                )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                project_log_scales(q_model, args.rho)
                project_weight_log_scales(q_model, args.weight_rho)

                running["loss"] += float(loss.detach().cpu())
                running["intra"] += float(intra.detach().cpu())
                running["proto"] += float(proto.detach().cpu())
                running["kl"] += float(kl.detach().cpu())
                running["kl_obj"] += float(kl_obj.detach().cpu())
                running["nll"] += float(nll.detach().cpu())
                running["task_ce"] += float(task_ce.detach().cpu())
                running["nll_obj"] += float(nll_obj.detach().cpu())
                running["phase"] += 1.0 if in_phase1 else 2.0
                running["out_proto"] += float(out_proto.detach().cpu())
                running["out_intra"] += float(out_intra.detach().cpu())
                running["out_logp_intra"] += float(out_logp_intra.detach().cpu())
                running["out_contract"] += float(out_contract.detach().cpu())
                running["out_ratio"] += float(out_ratio.detach().cpu())
                running["out_intra_q"] += float(out_intra_q.detach().cpu())
                running["out_intra_fp"] += float(out_intra_fp.detach().cpu())
                running["out_inter_contract"] += float(out_inter_contract.detach().cpu())
                running["out_inter_ratio"] += float(out_inter_ratio.detach().cpu())
                running["out_inter_q"] += float(out_inter_q.detach().cpu())
                running["out_inter_fp"] += float(out_inter_fp.detach().cpu())
                running["feat_inter_contract"] += float(feat_inter_contract.detach().cpu())
                running["feat_inter_ratio"] += float(feat_inter_ratio.detach().cpu())
                running["feat_inter_q"] += float(feat_inter_q.detach().cpu())
                running["feat_inter_fp"] += float(feat_inter_fp.detach().cpu())
                running["out_inter_retain"] += float(out_inter_retain.detach().cpu())
                running["out_inter_ratio_pre"] += float(out_inter_ratio_pre.detach().cpu())
                running["out_inter_pre"] += float(out_inter_pre.detach().cpu())
                running["out_inter_cur_retain"] += float(out_inter_cur_for_retain.detach().cpu())
                running["feat_inter_retain"] += float(feat_inter_retain.detach().cpu())
                running["feat_inter_ratio_pre"] += float(feat_inter_ratio_pre.detach().cpu())
                running["feat_inter_pre"] += float(feat_inter_pre.detach().cpu())
                running["feat_inter_cur_retain"] += float(feat_inter_cur_for_retain.detach().cpu())
                running["margin"] += float(margin.detach().cpu())
                running["feat_contract"] += float(feat_contract.detach().cpu())
                running["feat_ratio"] += float(feat_ratio.detach().cpu())
                running["feat_intra_q"] += float(feat_intra_q.detach().cpu())
                running["feat_intra_fp"] += float(feat_intra_fp.detach().cpu())
                running["inter"] += float(inter.detach().cpu())
                running["scale_reg"] += float(scale_reg.detach().cpu())
                running["w_scale_reg"] += float(w_scale_reg.detach().cpu())
                running["round_reg"] += float(round_reg.detach().cpu())
                nb += 1

            q_model.eval()
            set_soft_weight_hard_eval(q_model, False)
            acc = eval_classifier(q_model, test_loader, device)
            hard_acc = float('nan')
            if round_params and bool(getattr(args, "hard_eval_each_epoch", True)):
                set_soft_weight_hard_eval(q_model, True)
                hard_acc = eval_classifier(q_model, test_loader, device)
                if hard_acc > best_hard_acc:
                    best_hard_acc = hard_acc
                    best_hard_epoch = epoch
                    best_hard_state = {k: v.detach().cpu().clone() for k, v in q_model.state_dict().items()}
                set_soft_weight_hard_eval(q_model, False)
            rstats = weight_rounding_stats(q_model)
            phase_name = "A-utility" if in_phase1 else "B-inter"
            print(
                f"[Stage3][Epoch {epoch}/{args.scale_epochs}][{phase_name}] "
                f"lr={current_base_lr:.2e} "
                f"loss={running['loss']/max(nb,1):.4f} "
                f"intra={running['intra']/max(nb,1):.4f} "
                f"proto={running['proto']/max(nb,1):.4f} "
                f"KL={running['kl']/max(nb,1):.4f} "
                f"KLobj={running['kl_obj']/max(nb,1):.4f} "
                f"NLL={running['nll']/max(nb,1):.4f} "
                f"CEls={running['task_ce']/max(nb,1):.4f} "
                f"NLLobj={running['nll_obj']/max(nb,1):.4f} "
                f"out_proto={running['out_proto']/max(nb,1):.4f} "
                f"out_intraT={running['out_intra']/max(nb,1):.4f} "
                f"out_logp={running['out_logp_intra']/max(nb,1):.4f} "
                f"out_contract={running['out_contract']/max(nb,1):.4f} "
                f"out_ratio={running['out_ratio']/max(nb,1):.4f} "
                f"out_inter_metric={running['out_inter_contract']/max(nb,1):.4f} "
                f"out_ratio_fp={running['out_inter_ratio']/max(nb,1):.4f} "
                f"out_inter_q={running['out_inter_q']/max(nb,1):.4e} "
                f"out_inter_fp={running['out_inter_fp']/max(nb,1):.4e} "
                f"feat_contract={running['feat_contract']/max(nb,1):.4f} "
                f"feat_ratio={running['feat_ratio']/max(nb,1):.4f} "
                f"feat_inter_metric={running['feat_inter_contract']/max(nb,1):.4f} "
                f"feat_ratio_fp={running['feat_inter_ratio']/max(nb,1):.4f} "
                f"feat_inter_q={running['feat_inter_q']/max(nb,1):.4e} "
                f"feat_inter_fp={running['feat_inter_fp']/max(nb,1):.4e} "
                f"out_retain={running['out_inter_retain']/max(nb,1):.4e} "
                f"out_ratio_pre={running['out_inter_ratio_pre']/max(nb,1):.4f} "
                f"out_cur={running['out_inter_cur_retain']/max(nb,1):.4e} "
                f"out_pre={running['out_inter_pre']/max(nb,1):.4e} "
                f"feat_retain={running['feat_inter_retain']/max(nb,1):.4e} "
                f"feat_ratio_pre={running['feat_inter_ratio_pre']/max(nb,1):.4f} "
                f"feat_cur={running['feat_inter_cur_retain']/max(nb,1):.4e} "
                f"feat_pre={running['feat_inter_pre']/max(nb,1):.4e} "
                f"margin={running['margin']/max(nb,1):.4f} "
                f"inter={running['inter']/max(nb,1):.4f} "
                f"scale_reg={running['scale_reg']/max(nb,1):.4e} "
                f"w_scale={running['w_scale_reg']/max(nb,1):.4e} "
                f"round={running['round_reg']/max(nb,1):.4e} "
                f"round_bin={rstats['round_binary_frac']:.3f} "
                f"round_H={rstats['round_entropy']:.4f} "
                f"round_T={rstats['round_temp']:.3f} "
                f"test_acc={acc:.4f} "
                f"hard_acc={hard_acc:.4f}"
            )
            anneal_soft_round_temperature(q_model, getattr(args, "round_temp_anneal", 1.0), getattr(args, "min_round_temp", 0.25))
    finally:
        if feat_hook is not None:
            feat_hook.close()

    if best_hard_state is not None and bool(getattr(args, "optimize_weight_rounding", False)):
        print(f"[Stage3][BestHard] loading best hard-eval soft state from epoch {best_hard_epoch}, hard_acc={best_hard_acc:.4f}")
        q_model.load_state_dict({k: v.to(device) for k, v in best_hard_state.items()}, strict=False)
        set_soft_weight_hard_eval(q_model, False)

    if args.optimize_weight_scale or args.optimize_weight_rounding:
        finalize_soft_weight_quantization(q_model, hard=True)

    return q_model


# -----------------------------
# Full Type-9 pipeline
# -----------------------------
def run_quantization_pipeline(args):
    set_seed(args.seed)
    device = get_device(args)
    print(f"Using device: {device}")

    bits = [int(x.strip()) for x in args.bits.split(",") if x.strip()]
    bits = sorted(bits, reverse=True)
    if len(bits) == 0:
        raise ValueError("--bits must contain at least one integer bit-width")
    if args.ref_bit not in bits:
        raise ValueError(f"--ref_bit {args.ref_bit} must be in --bits {bits}")
    if len(bits) == 1:
        print(f"[V20] single-candidate bit mode detected: all quantized weight layers use {bits[0]} bit before Stage-3 scale/rounding optimization.")
    else:
        print(f"[V20] mixed-bit candidate mode detected: bits={bits}; allocation mode={args.alloc_mode}.")
    resolve_bit_budgets(args)

    calib_loader, test_loader, calib_dataset = build_loaders(args)
    scale_loader = build_scale_loader(args, calib_dataset)
    fp_model = load_resnet50_ptq(args, device)

    base_acc = eval_classifier(fp_model, test_loader, device)
    print(f"\n[Baseline] FP32 test acc = {base_acc:.4f}")

    # Keep an unfused FP32 reference model for KD/reference losses.
    # IMPORTANT: weight-scale/rounding refinement must use the *fused* weights
    # of q_model, not the unfused FP32 conv weights. Otherwise soft-rounded
    # weights bypass BN fusion and accuracy collapses at the start of Stage 3.
    fp_ref_model = copy.deepcopy(fp_model).to(device).eval()
    fp_weight_refs = None

    # -------- Stage 1 --------
    print("\n========== Stage 1: Fisher-weighted quantization sensitivity ==========")
    fisher = compute_weight_fisher_tensors(fp_model, calib_loader, device, max_batches=args.fisher_batches)
    U, R, n_params = compute_weight_sensitivity_tables(
        fp_model,
        fisher=fisher,
        bits=bits,
        granularity=args.weight_quant_granularity,
        eps=args.eps,
    )

    print(f"\n[Stage1] Layer sensitivity table, ranking by R@{args.ref_bit}:")
    ranked_names = sorted(R.keys(), key=lambda n: R[n][args.ref_bit], reverse=True)
    for n in ranked_names:
        vals = " ".join([f"R@{b}={R[n][b]:.2e},U@{b}={U[n][b]:.2e}" for b in bits])
        print(f"  {n:50s} {vals}")

    if args.dry_run:
        print("[DryRun] Stop after Stage 1.")
        return

    # -------- Stage 2 --------
    print("\n========== Stage 2: Sensitivity-guided bit allocation ==========")
    if args.alloc_mode == "budget_greedy":
        bit_cfg, alloc_info = budget_greedy_bit_allocation(
            U=U,
            R=R,
            n_params=n_params,
            bits=bits,
            ref_bit=args.ref_bit,
            target_avg_bit=args.target_avg_w_bit,
        )
    elif args.alloc_mode == "budget_greedy_v2":
        bit_cfg, alloc_info = budget_greedy_v2_bit_allocation(
            U=U,
            R=R,
            n_params=n_params,
            bits=bits,
            ref_bit=args.ref_bit,
            target_avg_bit=args.target_avg_w_bit,
            rel_alpha=args.alloc_rel_sensitivity_alpha,
            saved_power=args.alloc_score_saved_power,
            eps=args.eps,
        )
    elif args.alloc_mode == "fixed_ratio":
        bit_cfg, alloc_info = fixed_ratio_bit_allocation(
            U=U,
            R=R,
            n_params=n_params,
            bits=bits,
            ref_bit=args.ref_bit,
            bit_param_ratios=args.bit_param_ratios,
            target_avg_bit=args.target_avg_w_bit,
            max_lowest_bit_param_ratio=args.max_lowest_bit_param_ratio,
        )
    else:
        bit_cfg, alloc_info = enumerate_monotonic_bucket_allocation(
            U=U,
            R=R,
            n_params=n_params,
            bits=bits,
            ref_bit=args.ref_bit,
            target_avg_bit=args.target_avg_w_bit,
            force_nonempty_bucket=args.force_nonempty_bucket,
        )

    # -------- Apply weight PTQ --------
    print("\n[Apply] Fuse model and apply mixed-precision weight PTQ")
    q_model = fp_model
    q_model.eval()
    if hasattr(q_model, "fuse_model"):
        q_model.fuse_model()
    # Collect references after BN fusion and before PTQ. These are the correct
    # weights that soft rounding should refine.
    fp_weight_refs = collect_fp_weight_refs(q_model) if (getattr(args, "optimize_weight_scale", False) or getattr(args, "optimize_weight_rounding", False)) else None
    if hasattr(q_model, "apply_weight_ptq"):
        q_model.apply_weight_ptq(bit_cfg=bit_cfg, default_bits=min(bits))
    else:
        raise RuntimeError("model has no apply_weight_ptq method. Check resnet50_mp.py")

    if hasattr(q_model, "set_fp32_train_mode"):
        q_model.set_fp32_train_mode()
    q_model.eval()
    weight_only_acc = eval_classifier(q_model, test_loader, device)
    print(f"[Apply] Weight-only mixed PTQ acc = {weight_only_acc:.4f}")

    # -------- Activation calibration --------
    act_bit_cfg = assign_activation_bits(q_model, bit_cfg, args, calib_loader=calib_loader, device=device)

    # W/A/overall bit diagnostics.  This is printed before calibration and saved
    # into the checkpoint so later proxy analysis can report the exact budgets used.
    allowed_bits_for_summary = sorted([int(x.strip()) for x in str(args.bits).split(",") if x.strip()], reverse=True)
    bit_summary = compute_global_bit_summary(
        weight_bit_cfg=bit_cfg,
        weight_size_weights=n_params,
        act_bit_cfg=act_bit_cfg,
        act_size_weights=getattr(args, "_act_size_weights", {}),
        bits=allowed_bits_for_summary,
    )
    print_global_bit_summary(bit_summary)
    setattr(args, "_bit_summary", dict(bit_summary))

    calibrate_activations(q_model, calib_loader, device)

    # pre-scale optimization accuracy
    q_model.eval()
    if hasattr(q_model, "_set_all_act_quant_mode"):
        q_model._set_all_act_quant_mode("fake")
    pre_scale_acc = eval_classifier(q_model, test_loader, device)
    print(f"[Apply] Weight+Act PTQ before scale optimization acc = {pre_scale_acc:.4f}")

    # Save a pre-Stage3 quantized state for later three-way diagnostics.
    # This lets the analysis script compare FP32 vs pure-PTQ vs Stage3-optimized.
    pre_stage3_state = {k: v.detach().cpu().clone() for k, v in q_model.state_dict().items()}

    # -------- Stage 3 --------
    print("\n========== Stage 3: Class-aware scale optimization ==========")
    q_model = class_aware_scale_optimization(q_model, fp_ref_model, calib_loader, scale_loader, test_loader, device, args, bit_cfg=bit_cfg, fp_weight_refs=fp_weight_refs)

    # inference mode if supported
    if hasattr(q_model, "set_inference_mode"):
        try:
            q_model.set_inference_mode(default_bits=args.act_bits)
        except TypeError:
            q_model.set_inference_mode()
    q_model.eval()
    final_acc = eval_classifier(q_model, test_loader, device)
    print(f"\n[Final] Type-9 quantized model acc = {final_acc:.4f}")

    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = os.path.join(args.save_dir, args.save_name)
    torch.save(
        {
            "model": q_model.state_dict(),
            "model_pre_stage3": pre_stage3_state,
            "bit_cfg": bit_cfg,
            "act_bit_cfg": act_bit_cfg,
            "act_size_weights": getattr(args, "_act_size_weights", {}),
            "act_allocation_info": getattr(args, "_act_alloc_info", {}),
            "bit_summary": getattr(args, "_bit_summary", {}),
            "actual_avg_a_bit": getattr(args, "_actual_avg_a_bit", None),
            "allocation_info": alloc_info,
            "actual_avg_w_bit": alloc_info.get("avg_bit") if isinstance(alloc_info, dict) else None,
            "actual_avg_total_bit_count_weighted": getattr(args, "_bit_summary", {}).get("actual_avg_total_bit_count_weighted", None),
            "actual_avg_total_bit_balanced": getattr(args, "_bit_summary", {}).get("actual_avg_total_bit_balanced", None),
            "U_table": U,
            "R_table": R,
            "n_params": n_params,
            "acc_fp32": base_acc,
            "acc_weight_only": weight_only_acc,
            "acc_pre_scale": pre_scale_acc,
            "acc_final": final_acc,
            "args": vars(args),
        },
        ckpt_path,
    )
    print(f"[SAVE] {ckpt_path}")
    print("[SAVE] checkpoint includes model_pre_stage3 for FP32 / pre-Q / post-Q analysis")


if __name__ == "__main__":
    args = build_parser().parse_args()
    run_quantization_pipeline(args)
