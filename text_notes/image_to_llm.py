#!/usr/bin/env python3
# ============================================================
# MViT → Adapter → LLM (MedGemma) → Renal Note Generation
# FULL FIXED VERSION:
#   - Guarantees float32 into MViT (no float64/double)
#   - Casts visual tokens to LLM embed dtype (bf16) before concat
#   - Uses dtype= instead of torch_dtype=
# ============================================================

import numpy as np
import pandas as pd
import nibabel as nib

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms.functional as TF

from transformers import AutoTokenizer, AutoModelForCausalLM
from scipy.ndimage import label as cc_label


# -----------------------------
# CONFIG
# -----------------------------
NUM_FRAMES = 16
IMG_SIZE = 224
NUM_VIS_TOKENS = 8
MAX_TEXT_LEN = 512

LLM_NAME = "google/medgemma-27b-text-it"
LLM_DTYPE = torch.bfloat16  # LLM weights dtype


# -----------------------------
# NIFTI HELPERS
# -----------------------------
def load_nifti_float32(path: str) -> np.ndarray:
    # get_fdata() defaults to float64; force float32 immediately
    return nib.load(path).get_fdata().astype(np.float32, copy=False)

def xyz_to_zyx(x: np.ndarray) -> np.ndarray:
    # (X,Y,Z) -> (Z,Y,X)
    return np.transpose(x, (2, 1, 0)).astype(np.float32, copy=False)

def intensity_normalize(vol_zyx: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    vol = vol_zyx.astype(np.float32, copy=False)
    p1, p99 = np.percentile(vol, [1, 99])
    vol = np.clip(vol, p1, p99)
    return (vol - vol.mean()) / (vol.std() + eps)


# -----------------------------
# MASK LOGIC
# -----------------------------
def kidneys_present_both(mask_slice: np.ndarray, min_pixels: int = 50) -> bool:
    u = np.unique(mask_slice)
    u = u[u != 0]

    # labeled mask case
    if len(u) >= 2:
        ok = 0
        for lab in u[:10]:
            if np.count_nonzero(mask_slice == lab) >= min_pixels:
                ok += 1
        return ok >= 2

    # binary mask → 2 CCs
    binm = (mask_slice > 0).astype(np.uint8)
    if binm.sum() < 2 * min_pixels:
        return False

    cc, ncc = cc_label(binm)
    if ncc < 2:
        return False

    sizes = sorted([np.count_nonzero(cc == k) for k in range(1, ncc + 1)], reverse=True)
    return len(sizes) >= 2 and sizes[0] >= min_pixels and sizes[1] >= min_pixels

def valid_slices(mask_zyx: np.ndarray) -> list[int]:
    return sorted([z for z in range(mask_zyx.shape[0]) if kidneys_present_both(mask_zyx[z])])

def sample_slices(valid_bottom_to_top: list[int], rng: np.random.Generator) -> list[int]:
    # sample 16 randomly if possible; then keep TOP->BOTTOM order
    if len(valid_bottom_to_top) >= NUM_FRAMES:
        z = rng.choice(valid_bottom_to_top, NUM_FRAMES, replace=False)
        return sorted(z, reverse=True)  # top->bottom

    # pad by repeating if not enough
    base = sorted(valid_bottom_to_top, reverse=True)  # top->bottom
    out = []
    i = 0
    while len(out) < NUM_FRAMES:
        out.append(base[i % len(base)])
        i += 1
    return out


# -----------------------------
# DATASET
# -----------------------------
class KidneyNoteDataset(Dataset):
    """
    Returns:
      clip: float32 tensor (C=3,T,H,W)
      note: string
    """
    def __init__(self, df: pd.DataFrame, seed: int = 123):
        self.df = df.reset_index(drop=True)
        self.seed = seed
        self.epoch = 0
        self.cache_valid = {}  # idx -> valid slice list

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        rng = np.random.default_rng(self.seed + self.epoch * 10_000 + idx)

        img = intensity_normalize(xyz_to_zyx(load_nifti_float32(row["nifti"])))  # (Z,Y,X) float32
        mask = xyz_to_zyx(load_nifti_float32(row["mask"]))                       # (Z,Y,X) float32

        if idx not in self.cache_valid:
            self.cache_valid[idx] = valid_slices(mask)

        valid = self.cache_valid[idx]
        if len(valid) == 0:
            valid = list(range(img.shape[0]))

        z_sel = sample_slices(valid, rng)

        frames = []
        for z in z_sel:
            # FORCE float32 here
            t = torch.from_numpy(img[z]).to(dtype=torch.float32).unsqueeze(0)  # (1,H,W)
            t = TF.resize(t, [IMG_SIZE, IMG_SIZE], antialias=True)
            frames.append(t)

        clip = torch.stack(frames, dim=0)          # (T,1,H,W) float32
        clip = clip.repeat(1, 3, 1, 1)             # (T,3,H,W) float32
        clip = clip.permute(1, 0, 2, 3).contiguous()  # (C,T,H,W) float32

        # EXTRA safety: guarantee float32 output
        clip = clip.to(dtype=torch.float32)

        note = str(row["note_clean"]).strip()
        return clip, note


def collate_fn(batch):
    videos, notes = zip(*batch)
    videos = torch.stack(videos, dim=0).to(dtype=torch.float32)  # (B,C,T,H,W) float32
    return videos, list(notes)


# -----------------------------
# MViT BACKBONE
# -----------------------------
def build_mvit():
    mvit = torchvision.models.video.mvit_v2_s(weights="DEFAULT")
    dim = mvit.head[-1].in_features
    mvit.head[-1] = nn.Identity()
    return mvit, dim


# -----------------------------
# VISUAL TOKEN ADAPTER
# -----------------------------
class VisualAdapter(nn.Module):
    def __init__(self, img_dim: int, llm_dim: int, num_tokens: int):
        super().__init__()
        self.num_tokens = num_tokens
        self.fc = nn.Sequential(
            nn.Linear(img_dim, img_dim),
            nn.ReLU(),
            nn.Linear(img_dim, num_tokens * llm_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        return x.view(x.size(0), self.num_tokens, -1)


# -----------------------------
# MULTIMODAL GENERATOR
# -----------------------------
class RenalNoteGenerator(nn.Module):
    def __init__(self):
        super().__init__()

        # Vision stays float32
        self.vision, img_dim = build_mvit()

        # LLM in bf16, possibly sharded if device_map="auto"
        self.tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)

        self.llm = AutoModelForCausalLM.from_pretrained(
            LLM_NAME,
            dtype=LLM_DTYPE,         # <- deprecation-safe
            device_map="auto",
        )

        # Freeze LLM for now (recommended)
        for p in self.llm.parameters():
            p.requires_grad = False

        llm_dim = self.llm.config.hidden_size
        self.adapter = VisualAdapter(img_dim, llm_dim, NUM_VIS_TOKENS)

    def forward(self, video: torch.Tensor, notes: list[str]) -> torch.Tensor:
        # HARD GUARANTEE: vision input float32
        video = video.to(dtype=torch.float32)

        # Vision encode
        img_feat = self.vision(video)        # (B, img_dim) float32
        vis_tokens = self.adapter(img_feat)  # (B, K, H) float32

        prompts = ["Write a renal-focused radiology note:\n" + n for n in notes]

        tok = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=MAX_TEXT_LEN,
            return_tensors="pt",
        )

        # Move token tensors to the SAME device as LLM input embeddings
        # (works with device_map="auto" because embedding layer is on the "input device")
        embed_layer = self.llm.get_input_embeddings()
        embed_device = embed_layer.weight.device
        tok = {k: v.to(embed_device) for k, v in tok.items()}

        # Text embeddings inherit LLM dtype (bf16)
        text_embeds = embed_layer(tok["input_ids"])  # (B, L, H) bf16

        # 🔑 CRITICAL FIX: cast visual tokens to match text embed dtype + device
        vis_tokens = vis_tokens.to(device=embed_device, dtype=text_embeds.dtype)

        # Concatenate (now bf16 overall, not float32)
        inputs_embeds = torch.cat([vis_tokens, text_embeds], dim=1)  # (B, K+L, H) bf16

        # Labels: ignore visual prefix tokens
        ignore = torch.full(
            (tok["input_ids"].size(0), NUM_VIS_TOKENS),
            -100,
            device=embed_device,
            dtype=torch.long,
        )
        labels = torch.cat([ignore, tok["input_ids"]], dim=1)

        # Attention mask: include visual tokens as "attended"
        vis_attn = torch.ones(
            (tok["attention_mask"].size(0), NUM_VIS_TOKENS),
            device=embed_device,
            dtype=tok["attention_mask"].dtype,
        )
        attention_mask = torch.cat([vis_attn, tok["attention_mask"]], dim=1)

        out = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        return out.loss


# -----------------------------
# TRAIN LOOP
# -----------------------------
def train_one_epoch(model, loader, opt, device, epoch):
    model.train()
    total = 0.0

    for step, (video, notes) in enumerate(loader):
        # Keep vision on your chosen device
        video = video.to(device=device, dtype=torch.float32)

        opt.zero_grad(set_to_none=True)
        loss = model(video, notes)
        loss.backward()
        opt.step()

        total += loss.item()
        if step % 5 == 0:
            print(f"epoch {epoch} step {step} loss {loss.item():.4f}")

    return total / max(len(loader), 1)


# -----------------------------
# MAIN
# -----------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--out", default="renal_note_generator.pt")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    dataset = KidneyNoteDataset(df)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Vision device (choose 0 if you're using one GPU for MViT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RenalNoteGenerator().to(device)

    # Only train adapter + vision (LLM frozen)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-4,
    )

    for epoch in range(args.epochs):
        dataset.set_epoch(epoch)
        loss = train_one_epoch(model, loader, opt, device, epoch)
        print(f"Epoch {epoch} avg loss {loss:.4f}")
        torch.save(model.state_dict(), args.out)


if __name__ == "__main__":
    main()
