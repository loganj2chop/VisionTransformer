#!/usr/bin/env python3
"""
Evaluate a saved MViT kidney checkpoint with TTA (N random-slice passes).

Usage:
    python evaluate_kidney.py \
        --csv data.csv \
        --checkpoint checkpoints/best.pt \
        --tta_passes 5
"""
import os
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import nibabel as nib

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms.functional as TF

from scipy.ndimage import label as cc_label
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, accuracy_score, confusion_matrix,
)


# -----------------------------
# Config (must match training)
# -----------------------------
LABEL_COLS  = ["kidney_stone", "ureteral_stone", "hydronephrosis", "urinary_tract_dilation"]
NUM_FRAMES  = 16
IMG_SIZE    = 224
BATCH_SIZE  = 8
NUM_WORKERS = 4
SEED        = 123
TEST_FRAC   = 0.2
TTA_PASSES  = 5

MIN_SLICE_GAP        = 2
MIN_BILATERAL_FRAMES = 4

# Epoch tag offset used during TTA — must not overlap with training epochs.
# If you trained for e.g. 50 epochs, this starts at 150, giving plenty of room.
TTA_EPOCH_OFFSET = 150


# -----------------------------
# Utilities  (identical to training)
# -----------------------------
def _safe_float32(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float32, copy=False) if x.dtype != np.float32 else x


def intensity_normalize(img: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    img = _safe_float32(img)
    p1, p99 = np.percentile(img, [1, 99])
    img = np.clip(img, p1, p99)
    return (img - img.mean()) / (img.std() + eps)


def load_nifti(path: str) -> np.ndarray:
    return _safe_float32(nib.load(path).get_fdata())


def get_axial_slices_xyz(img_xyz: np.ndarray) -> np.ndarray:
    return np.transpose(img_xyz, (2, 1, 0))


def count_kidney_components(mask_slice: np.ndarray, min_pixels: int = 50) -> int:
    ms = mask_slice
    u  = np.unique(ms)
    u  = u[u != 0]
    if len(u) >= 2:
        return sum(1 for lab in u[:10] if np.count_nonzero(ms == lab) >= min_pixels)
    bin_mask = (ms > 0).astype(np.uint8)
    if bin_mask.sum() < min_pixels:
        return 0
    cc, ncc = cc_label(bin_mask)
    sizes   = sorted(
        [np.count_nonzero(cc == k) for k in range(1, ncc + 1)], reverse=True
    )
    return sum(1 for s in sizes if s >= min_pixels)


def compute_valid_slice_indices(mask_zyx: np.ndarray) -> Tuple[List[int], List[int]]:
    bilateral, unilateral = [], []
    for z in range(mask_zyx.shape[0]):
        n = count_kidney_components(mask_zyx[z])
        if n >= 2:
            bilateral.append(z)
        elif n == 1:
            unilateral.append(z)
    return sorted(bilateral), sorted(unilateral)


def spaced_sample(indices: List[int], n: int, rng: np.random.Generator,
                  min_gap: int = MIN_SLICE_GAP) -> List[int]:
    if len(indices) == 0:
        return []
    if len(indices) <= n:
        return list(indices)
    arr = np.array(indices)
    for _ in range(50):
        chosen = [arr[rng.integers(0, max(1, len(arr) // n))]]
        for _ in range(n - 1):
            candidates = arr[arr >= chosen[-1] + min_gap]
            if len(candidates) == 0:
                break
            chosen.append(candidates[rng.integers(0, max(1, len(candidates) // 2 + 1))])
        if len(chosen) == n:
            return sorted(chosen)
    step = len(arr) // n
    return sorted([arr[i * step] for i in range(n)])


def sample_slices_each_epoch(
    bilateral: List[int],
    unilateral: List[int],
    num_frames: int,
    rng: np.random.Generator,
    min_bilateral: int = MIN_BILATERAL_FRAMES,
    min_gap: int = MIN_SLICE_GAP,
) -> List[int]:
    all_valid = sorted(set(bilateral + unilateral))
    if len(all_valid) == 0:
        return []
    n_bilateral      = min(min_bilateral, len(bilateral), num_frames)
    bilateral_chosen = spaced_sample(bilateral, n_bilateral, rng, min_gap)
    remaining_pool   = sorted(set(all_valid) - set(bilateral_chosen))
    n_remaining      = num_frames - len(bilateral_chosen)
    extra_chosen     = spaced_sample(remaining_pool, n_remaining, rng, min_gap)
    chosen = sorted(set(bilateral_chosen + extra_chosen))
    while len(chosen) < num_frames:
        chosen = chosen + [chosen[-1]]
    chosen = chosen[:num_frames]
    return sorted(chosen, reverse=True)


def resize_frame(frame_2d: np.ndarray, out_size: int = IMG_SIZE) -> torch.Tensor:
    t = torch.from_numpy(frame_2d).float().unsqueeze(0)
    return TF.resize(t, [out_size, out_size], antialias=True)


# -----------------------------
# Dataset  (identical to training)
# -----------------------------
class KidneyVideoDataset(Dataset):
    def __init__(self, df: pd.DataFrame, base_seed: int = SEED):
        self.df        = df.reset_index(drop=True).copy()
        self.base_seed = base_seed
        self.epoch     = 0
        self._cache: Dict[int, Tuple[List[int], List[int]]] = {}

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.df)

    def _get_slice_pools(self, idx: int, mask_path: str) -> Tuple[List[int], List[int]]:
        if idx in self._cache:
            return self._cache[idx]
        mask_zyx              = get_axial_slices_xyz(load_nifti(mask_path))
        bilateral, unilateral = compute_valid_slice_indices(mask_zyx)
        self._cache[idx]      = (bilateral, unilateral)
        return bilateral, unilateral

    def __getitem__(self, idx: int):
        row       = self.df.iloc[idx]
        img_path  = str(row["nifti"])
        mask_path = str(row["mask"])

        seed = (self.base_seed * 1_000_003) + (self.epoch * 10_000_019) + idx
        rng  = np.random.default_rng(seed)

        img_zyx  = get_axial_slices_xyz(load_nifti(img_path))
        mask_zyx = get_axial_slices_xyz(load_nifti(mask_path))

        img_zyx         = intensity_normalize(img_zyx)
        mask_binary_zyx = (mask_zyx > 0).astype(np.float32)

        bilateral, unilateral = self._get_slice_pools(idx, mask_path)
        if len(bilateral) + len(unilateral) == 0:
            bilateral  = list(range(img_zyx.shape[0]))
            unilateral = []

        z_sel = sample_slices_each_epoch(bilateral, unilateral, NUM_FRAMES, rng)

        frame_tensors = []
        for z in z_sel:
            img_frame  = resize_frame(img_zyx[z],         IMG_SIZE)
            mask_frame = resize_frame(mask_binary_zyx[z], IMG_SIZE)
            combined   = torch.cat([img_frame.repeat(3, 1, 1), mask_frame], dim=0)
            frame_tensors.append(combined)

        clip = torch.stack(frame_tensors, dim=0)
        clip = clip.permute(1, 0, 2, 3).contiguous()

        y = torch.tensor([float(row[c]) for c in LABEL_COLS], dtype=torch.float32)
        return clip, y


# -----------------------------
# Model builder  (identical to training)
# -----------------------------
def build_mvit(num_labels: int = len(LABEL_COLS)) -> nn.Module:
    if not hasattr(torchvision.models.video, "mvit_v2_s"):
        raise RuntimeError("torchvision.models.video.mvit_v2_s not found — upgrade torchvision.")

    model = torchvision.models.video.mvit_v2_s(weights=None)   # no pretrained download needed

    old_conv = model.conv_proj
    new_conv  = nn.Conv3d(
        in_channels=4,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )
    model.conv_proj = new_conv

    if hasattr(model, "head") and isinstance(model.head, nn.Sequential):
        last = model.head[-1]
        if isinstance(last, nn.Linear):
            model.head[-1] = nn.Linear(last.in_features, num_labels)
            return model
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        model.fc = nn.Linear(model.fc.in_features, num_labels)
        return model

    raise RuntimeError("Could not replace classifier head — check torchvision version.")


# -----------------------------
# TTA inference
# -----------------------------
@torch.no_grad()
def predict_tta(
    model: nn.Module,
    df_test: pd.DataFrame,
    device: torch.device,
    num_passes: int = TTA_PASSES,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    base_seed: int = SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        avg_probs    – (N, num_labels)  mean probability across all passes
        all_probs    – (P, N, num_labels) per-pass probabilities
    """
    model.eval()
    all_pass_probs: List[np.ndarray] = []

    for tta_pass in range(num_passes):
        epoch_tag = TTA_EPOCH_OFFSET + tta_pass

        ds = KidneyVideoDataset(df_test, base_seed=base_seed)
        ds.set_epoch(epoch_tag)

        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )

        pass_probs: List[np.ndarray] = []
        for x, _ in loader:
            logits = model(x.to(device, non_blocking=True))
            pass_probs.append(torch.sigmoid(logits).cpu().numpy())

        pass_probs_np = np.concatenate(pass_probs, axis=0)
        all_pass_probs.append(pass_probs_np)
        print(f"  TTA pass {tta_pass + 1}/{num_passes} — mean prob {pass_probs_np.mean():.4f}")

    all_probs_arr = np.stack(all_pass_probs, axis=0)   # (P, N, L)
    avg_probs     = all_probs_arr.mean(axis=0)          # (N, L)
    return avg_probs, all_probs_arr


# -----------------------------
# Metrics
# -----------------------------
def compute_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    label_cols: List[str],
    threshold: float = 0.5,
) -> pd.DataFrame:
    preds = (probs >= threshold).astype(int)
    rows  = []

    for i, col in enumerate(label_cols):
        yt    = y_true[:, i].astype(int)
        yp    = preds[:, i]
        yprob = probs[:, i]

        n_pos = int(yt.sum())
        n_neg = len(yt) - n_pos

        auc = roc_auc_score(yt, yprob)           if len(np.unique(yt)) > 1 else float("nan")
        ap  = average_precision_score(yt, yprob) if len(np.unique(yt)) > 1 else float("nan")
        f1  = f1_score(yt, yp, zero_division=0)
        acc = accuracy_score(yt, yp)

        if len(np.unique(yt)) > 1:
            tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        else:
            tn, fp, fn, tp = (n_neg, 0, 0, n_pos) if yt[0] == 1 else (n_neg, 0, 0, 0)

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

        rows.append(dict(
            label=col,
            auc_roc=round(auc, 4),
            avg_precision=round(ap, 4),
            f1=round(f1, 4),
            accuracy=round(acc, 4),
            sensitivity=round(sensitivity, 4),
            specificity=round(specificity, 4),
            tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn),
            support=n_pos,
            n_total=len(yt),
        ))

    df_rows = pd.DataFrame(rows)
    macro   = df_rows[["auc_roc", "avg_precision", "f1",
                        "accuracy", "sensitivity", "specificity"]].apply(
        lambda c: round(float(np.nanmean(c)), 4)
    ).to_dict()
    macro.update(dict(
        label="MACRO_AVG",
        tp=int(df_rows["tp"].sum()), tn=int(df_rows["tn"].sum()),
        fp=int(df_rows["fp"].sum()), fn=int(df_rows["fn"].sum()),
        support=int(df_rows["support"].sum()),
        n_total=int(df_rows["n_total"].sum()),
    ))
    rows.append(macro)
    return pd.DataFrame(rows)


# -----------------------------
# Multilabel stratified split  (must reproduce training split exactly)
# -----------------------------
def multilabel_train_test_split(df: pd.DataFrame, label_cols: List[str],
                                 test_frac: float, seed: int):
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
        y_mat = df[label_cols].values.astype(int)
        msss  = MultilabelStratifiedShuffleSplit(
            n_splits=1, test_size=test_frac, random_state=seed
        )
        train_idx, test_idx = next(msss.split(np.arange(len(df)), y_mat))
        return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)
    except ImportError:
        print("  [WARN] iterative-stratification not installed — using random split.")
        train_df = df.sample(frac=1 - test_frac, random_state=seed)
        test_df  = df.drop(train_df.index)
        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


# -----------------------------
# Main
# -----------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",         type=str, required=True,
                        help="Full dataset CSV (same file used during training). "
                             "The test split is re-derived with seed=123, test_frac=0.2.")
    parser.add_argument("--checkpoint",  type=str, required=True,
                        help="Path to saved best.pt checkpoint")
    parser.add_argument("--tta_passes",  type=int, default=TTA_PASSES)
    parser.add_argument("--batch_size",  type=int, default=BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--threshold",   type=float, default=0.5)
    parser.add_argument("--out_csv",     type=str, default="test_predictions.csv")
    parser.add_argument("--out_metrics", type=str, default="test_metrics.csv")
    args = parser.parse_args()

    # These are fixed — must match what was used during training exactly.
    SPLIT_SEED      = 123
    SPLIT_TEST_FRAC = 0.2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:     {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"TTA passes: {args.tta_passes}")
    print(f"Split seed: {SPLIT_SEED}  |  test_frac: {SPLIT_TEST_FRAC}  (hardcoded to match training)")

    # ── Derive the exact same test split as training ───────────────────────
    df = pd.read_csv(args.csv)
    for c in ["nifti", "mask"] + LABEL_COLS:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    _, df_test = multilabel_train_test_split(df, LABEL_COLS, SPLIT_TEST_FRAC, SPLIT_SEED)
    print(f"\nTest set size: {len(df_test)}")
    print(f"Label prevalence:")
    print((df_test[LABEL_COLS].mean() * 100).round(1).to_string())

    # ── Load model ────────────────────────────────────────────────────────
    model = build_mvit(num_labels=len(LABEL_COLS)).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"\nLoaded checkpoint from epoch {ckpt.get('epoch', '?')} "
          f"(val_loss={ckpt.get('val_loss', float('nan')):.4f})")

    # ── TTA inference ─────────────────────────────────────────────────────
    print(f"\nRunning TTA inference ({args.tta_passes} passes) …")
    avg_probs, all_pass_probs = predict_tta(
        model, df_test, device,
        num_passes=args.tta_passes,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        base_seed=SPLIT_SEED,
    )

    preds_np = (avg_probs >= args.threshold).astype(np.int8)

    # ── Prediction CSV ────────────────────────────────────────────────────
    df_out = df_test.copy()
    for i, col in enumerate(LABEL_COLS):
        df_out[f"prob_{col}"] = avg_probs[:, i]
        df_out[f"pred_{col}"] = preds_np[:, i]
        for p in range(args.tta_passes):
            df_out[f"prob_{col}_pass{p + 1}"] = all_pass_probs[p, :, i]

    df_out.to_csv(args.out_csv, index=False)
    print(f"\nPredictions saved to: {args.out_csv}")

    # ── Metrics CSV ───────────────────────────────────────────────────────
    y_test_np  = df_test[LABEL_COLS].values.astype(np.float32)
    df_metrics = compute_metrics(y_test_np, avg_probs, LABEL_COLS, threshold=args.threshold)
    df_metrics.to_csv(args.out_metrics, index=False)
    print(f"Metrics saved to:     {args.out_metrics}")

    print(f"\nTest set metrics (TTA={args.tta_passes} passes, threshold={args.threshold}):")
    print(df_metrics.to_string(index=False))

    # Per-pass AUC variance
    print(f"\nPer-TTA-pass macro AUC (variance check):")
    for p in range(args.tta_passes):
        aucs = []
        for i, col in enumerate(LABEL_COLS):
            yt = y_test_np[:, i]
            if len(np.unique(yt)) > 1:
                aucs.append(roc_auc_score(yt, all_pass_probs[p, :, i]))
        mean_auc = np.mean(aucs) if aucs else float("nan")
        print(f"  Pass {p + 1}: macro AUC = {mean_auc:.4f}  "
              f"per-label: {[round(a, 3) for a in aucs]}")


if __name__ == "__main__":
    main()