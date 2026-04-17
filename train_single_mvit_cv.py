#!/usr/bin/env python3
import os
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import nibabel as nib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms.functional as TF

from scipy.ndimage import label as cc_label
from sklearn.metrics import roc_auc_score


# -----------------------------
# Config
# -----------------------------
LABEL_COLS  = ["kidney_stone", "ureteral_stone", "hydronephrosis", "urinary_tract_dilation"]
NUM_FRAMES  = 16
IMG_SIZE    = 224
EPOCHS      = 50
BATCH_SIZE  = 8
LR          = 2e-4
NUM_WORKERS = 4
SEED        = 123
GRAD_CLIP   = 1.0
TEST_FRAC   = 0.2
OUT_DIR     = "checkpoints"

MIN_SLICE_GAP      = 2
MIN_BILATERAL_FRAMES = 4


# -----------------------------
# Utilities
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
    """(X,Y,Z) -> (Z,Y,X) axial slices."""
    return np.transpose(img_xyz, (2, 1, 0))


def count_kidney_components(mask_slice: np.ndarray, min_pixels: int = 50) -> int:
    """Returns number of distinct kidney regions in this slice (0, 1, or 2)."""
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
    """
    Returns:
        bilateral:  slices where BOTH kidneys are visible (sorted ascending)
        unilateral: slices where exactly ONE kidney is visible (sorted ascending)
    """
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
    """Sample up to n indices with at least min_gap between consecutive picks."""
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
    step   = len(arr) // n
    return sorted([arr[i * step] for i in range(n)])


def sample_slices_each_epoch(
    bilateral: List[int],
    unilateral: List[int],
    num_frames: int,
    rng: np.random.Generator,
    min_bilateral: int = MIN_BILATERAL_FRAMES,
    min_gap: int = MIN_SLICE_GAP,
) -> List[int]:
    """
    Reserve up to min_bilateral frames from bilateral slices (spaced),
    fill remainder from all valid slices (spaced), return top->bottom (desc z).
    """
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
    return sorted(chosen, reverse=True)   # top -> bottom


def resize_frame(frame_2d: np.ndarray, out_size: int = IMG_SIZE) -> torch.Tensor:
    t = torch.from_numpy(frame_2d).float().unsqueeze(0)
    return TF.resize(t, [out_size, out_size], antialias=True)


# -----------------------------
# Dataset  (4-channel: RGB image + kidney mask)
# -----------------------------
class KidneyVideoDataset(Dataset):
    def __init__(self, df: pd.DataFrame, base_seed: int = SEED):
        self.df        = df.reset_index(drop=True).copy()
        self.base_seed = base_seed
        self.epoch     = 0
        # cache: idx -> (bilateral_zs, unilateral_zs)
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

        # Load image and mask
        img_zyx  = get_axial_slices_xyz(load_nifti(img_path))
        mask_xyz = load_nifti(mask_path)
        mask_zyx = get_axial_slices_xyz(mask_xyz)

        img_zyx = intensity_normalize(img_zyx)

        # Binary kidney mask normalised to [0, 1]
        mask_binary_zyx = (mask_zyx > 0).astype(np.float32)

        bilateral, unilateral = self._get_slice_pools(idx, mask_path)
        if len(bilateral) + len(unilateral) == 0:
            bilateral  = list(range(img_zyx.shape[0]))
            unilateral = []

        z_sel = sample_slices_each_epoch(bilateral, unilateral, NUM_FRAMES, rng)

        # Build 4-channel frames: channels 0-2 = grayscale image (replicated),
        #                         channel  3   = kidney mask
        frame_tensors = []
        for z in z_sel:
            img_frame  = resize_frame(img_zyx[z],          IMG_SIZE)   # (1, H, W)
            mask_frame = resize_frame(mask_binary_zyx[z],  IMG_SIZE)   # (1, H, W)
            # replicate grayscale to 3 channels, then append mask as channel 4
            combined   = torch.cat([img_frame.repeat(3, 1, 1), mask_frame], dim=0)  # (4, H, W)
            frame_tensors.append(combined)

        clip = torch.stack(frame_tensors, dim=0)          # (T, 4, H, W)
        clip = clip.permute(1, 0, 2, 3).contiguous()      # (4, T, H, W)  — C first for MViT

        y = torch.tensor([float(row[c]) for c in LABEL_COLS], dtype=torch.float32)
        return clip, y


# -----------------------------
# Model  (patched for 4-channel input)
# -----------------------------
def build_mvit(num_labels: int = len(LABEL_COLS)) -> nn.Module:
    if not hasattr(torchvision.models.video, "mvit_v2_s"):
        raise RuntimeError("torchvision.models.video.mvit_v2_s not found — upgrade torchvision.")

    model = torchvision.models.video.mvit_v2_s(weights="DEFAULT")

    # ── Patch first conv to accept 4 channels instead of 3 ──────────────
    # MViT-v2-S uses model.conv_proj as the patch embedding conv
    old_conv = model.conv_proj
    new_conv  = nn.Conv3d(
        in_channels=4,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )
    with torch.no_grad():
        # Copy pretrained RGB weights into first 3 channels
        new_conv.weight[:, :3] = old_conv.weight
        # Initialise mask channel to zero — model learns its weight from scratch
        new_conv.weight[:, 3]  = 0.0
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)
    model.conv_proj = new_conv

    # ── Replace classifier head ──────────────────────────────────────────
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
# Train / Eval
# -----------------------------
def train_one_epoch(model, loader, optimizer, device, epoch: int) -> float:
    model.train()
    bce = nn.BCEWithLogitsLoss()
    total_loss, n = 0.0, 0
    for step, (x, y) in enumerate(loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = bce(model(x), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        n          += x.size(0)
        if step % 20 == 0:
            print(f"  epoch {epoch} step {step}/{len(loader)} loss {loss.item():.4f}")
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device) -> Tuple[float, torch.Tensor, torch.Tensor]:
    model.eval()
    bce = nn.BCEWithLogitsLoss()
    total_loss, n = 0.0, 0
    all_logits, all_y = [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        total_loss += bce(logits, y).item() * x.size(0)
        n          += x.size(0)
        all_logits.append(logits.cpu())
        all_y.append(y.cpu())
    logits = torch.cat(all_logits, dim=0) if all_logits else torch.zeros(0, len(LABEL_COLS))
    ytrue  = torch.cat(all_y,     dim=0) if all_y     else torch.zeros(0, len(LABEL_COLS))
    return total_loss / max(n, 1), logits, ytrue


@torch.no_grad()
def predict(model, loader, device) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    all_logits = []
    for x, _ in loader:
        all_logits.append(model(x.to(device, non_blocking=True)).cpu())
    logits = torch.cat(all_logits, dim=0)
    return logits, torch.sigmoid(logits)


# -----------------------------
# Multilabel stratified split
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
        print("         Install with: pip install iterative-stratification")
        train_df = df.sample(frac=1 - test_frac, random_state=seed)
        test_df  = df.drop(train_df.index)
        return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


# -----------------------------
# Main
# -----------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",         type=str,   required=True)
    parser.add_argument("--batch_size",  type=int,   default=BATCH_SIZE)
    parser.add_argument("--epochs",      type=int,   default=EPOCHS)
    parser.add_argument("--lr",          type=float, default=LR)
    parser.add_argument("--num_workers", type=int,   default=NUM_WORKERS)
    parser.add_argument("--seed",        type=int,   default=SEED)
    parser.add_argument("--test_frac",   type=float, default=TEST_FRAC)
    parser.add_argument("--out_csv",     type=str,   default="test_predictions.csv")
    parser.add_argument("--out_dir",     type=str,   default=OUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.csv)
    for c in ["nifti", "mask"] + LABEL_COLS:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Total samples: {len(df)}")
    print(f"Labels: {LABEL_COLS}")

    # ── Multilabel stratified train/test split ────────────────────────────
    df_train, df_test = multilabel_train_test_split(
        df, LABEL_COLS, args.test_frac, args.seed
    )
    print(f"\nTrain: {len(df_train)}  |  Test: {len(df_test)}")
    print(f"\nLabel prevalence — train:")
    print((df_train[LABEL_COLS].mean() * 100).round(1).to_string())
    print(f"\nLabel prevalence — test:")
    print((df_test[LABEL_COLS].mean() * 100).round(1).to_string())

    train_ds = KidneyVideoDataset(df_train, base_seed=args.seed)
    test_ds  = KidneyVideoDataset(df_test,  base_seed=args.seed)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
    )

    model     = build_mvit(num_labels=len(LABEL_COLS)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")
    best_ckpt     = os.path.join(args.out_dir, "best.pt")

    # ── Training loop ─────────────────────────────────────────────────────
    for epoch in range(args.epochs):
        train_ds.set_epoch(epoch)
        tr_loss        = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_loss, _, _ = evaluate(model, test_loader, device)
        scheduler.step()
        print(f"Epoch {epoch:02d} | train {tr_loss:.4f} | val {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "val_loss": val_loss}, best_ckpt)
            print(f"  ✓ Saved best checkpoint (val_loss={val_loss:.4f})")

    # ── Test set predictions using best checkpoint ────────────────────────
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"\nLoaded best weights from epoch {ckpt['epoch']}")

    logits, probs = predict(model, test_loader, device)
    probs_np  = probs.numpy()
    preds_np  = (probs_np >= 0.5).astype(np.int8)

    for i, col in enumerate(LABEL_COLS):
        df_test[f"prob_{col}"]  = probs_np[:, i]
        df_test[f"pred_{col}"]  = preds_np[:, i]
        df_test[f"logit_{col}"] = logits.numpy()[:, i]

    df_test.to_csv(args.out_csv, index=False)
    print(f"\nTest predictions saved to: {args.out_csv}")

    # ── Per-label AUC on test set ─────────────────────────────────────────
    y_test_np = df_test[LABEL_COLS].values.astype(np.float32)
    print(f"\nTest set per-label AUC:")
    for i, col in enumerate(LABEL_COLS):
        if len(np.unique(y_test_np[:, i])) > 1:
            auc = roc_auc_score(y_test_np[:, i], probs_np[:, i])
            print(f"  {col:<30s} AUC={auc:.3f}")
        else:
            print(f"  {col:<30s} AUC=N/A (single class in test set)")


if __name__ == "__main__":
    main()