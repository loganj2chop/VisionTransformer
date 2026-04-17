# Kidney Imaging NLP & Video Classification Pipeline

A two-stage pipeline for automated detection of urinary tract findings from radiology reports and CT imaging.

**Stage 1** uses [MedGemma](https://huggingface.co/google/medgemma-27b-text-it) to extract structured labels from free-text radiology notes. **Stage 2** trains and evaluates a [MViT-v2](https://arxiv.org/abs/2112.01526) video transformer on axial CT slices, guided by kidney segmentation masks.

---

## Pipeline Overview

```
Radiology Notes (CSV)
        │
        ▼
 concepts.py  ──►  Ground-truth labels (CSV)
                          │
                          ▼
              + NIfTI volumes + kidney masks
                          │
                          ▼
  train_single_mvit_cv.py  ──►  Trained checkpoint (best.pt)
                                        │
                                        ▼
                          evaluate_kidney.py  ──►  Metrics + predictions
```

---

## Scripts

### `concepts.py` — Radiology Note Label Extraction

Runs MedGemma-27B over a CSV of free-text radiology reports and outputs binary flags for six urinary tract findings. These flags serve as ground-truth labels for model training.

**Extracted flags:**

| Flag | Description |
|---|---|
| `kidney_stone` | Stone within the kidney (renal parenchyma or collecting system) |
| `ureteral_stone` | Stone within the ureter (proximal, mid, or distal) |
| `hydronephrosis` | Dilation of the renal pelvis/calyces due to obstruction |
| `urinary_tract_dilation` | Hydroureter or ureteral dilation not caused by a stone |
| `stent` | Ureteral stent present (DJ/double-J or similar) |
| `tube` | Drainage tube present (nephrostomy, PCN, or Foley) |

> **Note:** `kidney_stone` and `ureteral_stone` are mutually exclusive by location. If both locations are mentioned in the same note, both flags are set to 1.

**Key config (edit at top of script):**

```python
MODEL_ID       = "google/medgemma-27b-text-it"
INPUT_CSV      = "vtnotes2.csv"        # must contain a "note" column
OUTPUT_CSV     = "notes_medgemma27_concepts_redo.csv"
CHECKPOINT_N   = 500                   # saves progress every N rows
MAX_NOTE_CHARS = 6000                  # notes are truncated to this length
```

**Run:**
```bash
python concepts.py
```

---

### `train_single_mvit_cv.py` — MViT Training

Trains a MViT-v2-S video transformer on axial CT slices for multilabel classification of four findings. Each sample is a 16-frame "video clip" sampled from the axial stack, prioritizing slices where both kidneys are visible. The kidney segmentation mask is appended as a 4th input channel.

**Predicted labels:**

```
kidney_stone | ureteral_stone | hydronephrosis | urinary_tract_dilation
```

**Key design choices:**

- **4-channel input:** Grayscale CT (×3) + binary kidney mask. The mask channel is zero-initialized; the RGB weights are copied from the ImageNet-pretrained MViT.
- **Bilateral-aware slice sampling:** Each epoch draws a fresh random subset of slices, guaranteeing ≥4 frames from slices where both kidneys are visible.
- **Multilabel stratified split:** Uses `iterative-stratification` for reproducible 80/20 train/test split (falls back to random split if not installed).
- **Checkpointing:** Saves `best.pt` whenever validation loss improves.

**Input CSV columns required:**

| Column | Description |
|---|---|
| `nifti` | Path to CT volume (`.nii` / `.nii.gz`) |
| `mask` | Path to kidney segmentation mask (NIfTI) |
| `kidney_stone` | Binary label (0/1) |
| `ureteral_stone` | Binary label (0/1) |
| `hydronephrosis` | Binary label (0/1) |
| `urinary_tract_dilation` | Binary label (0/1) |

**Run:**
```bash
python train_single_mvit_cv.py \
    --csv data.csv \
    --epochs 50 \
    --batch_size 8 \
    --lr 2e-4 \
    --out_dir checkpoints
```

**All arguments:**

| Argument | Default | Description |
|---|---|---|
| `--csv` | *(required)* | Path to input CSV |
| `--batch_size` | `8` | Training batch size |
| `--epochs` | `50` | Number of training epochs |
| `--lr` | `2e-4` | AdamW learning rate |
| `--num_workers` | `4` | DataLoader workers |
| `--seed` | `123` | Random seed |
| `--test_frac` | `0.2` | Fraction held out for test |
| `--out_csv` | `test_predictions.csv` | Test prediction output path |
| `--out_dir` | `checkpoints/` | Directory for saved checkpoints |

---

### `evaluate_kidney.py` — TTA Evaluation

Loads a saved checkpoint and evaluates it on the held-out test set using **test-time augmentation (TTA)** — running multiple random-slice passes and averaging the predicted probabilities.

**Metrics reported (per label + macro average):**

- AUC-ROC
- Average Precision (AP)
- F1 Score
- Accuracy
- Sensitivity (recall)
- Specificity
- Confusion matrix counts (TP / TN / FP / FN)

> The train/test split is re-derived using the same fixed seed (`123`) and test fraction (`0.2`) as training, ensuring the evaluation set is identical.

**Run:**
```bash
python evaluate_kidney.py \
    --csv data.csv \
    --checkpoint checkpoints/best.pt \
    --tta_passes 5
```

**All arguments:**

| Argument | Default | Description |
|---|---|---|
| `--csv` | *(required)* | Same full CSV used during training |
| `--checkpoint` | *(required)* | Path to `best.pt` |
| `--tta_passes` | `5` | Number of random-slice TTA passes |
| `--batch_size` | `8` | Inference batch size |
| `--num_workers` | `4` | DataLoader workers |
| `--threshold` | `0.5` | Probability threshold for binary predictions |
| `--out_csv` | `test_predictions.csv` | Per-sample predictions output |
| `--out_metrics` | `test_metrics.csv` | Per-label metrics output |

---

## Installation

```bash
pip install torch torchvision
pip install transformers accelerate
pip install nibabel pandas scikit-learn scipy tqdm
pip install iterative-stratification   # recommended for stratified splitting
```

> MedGemma requires a Hugging Face account and acceptance of the model's terms of use. Authenticate with `huggingface-cli login` before running `concepts.py`.

A CUDA-capable GPU is strongly recommended for both scripts. `concepts.py` will load MedGemma-27B in `bfloat16`; plan for ~55 GB GPU memory or use multi-GPU with `device_map="auto"`.

---

## Output Files

| File | Generated by | Contents |
|---|---|---|
| `notes_medgemma27_concepts_redo.csv` | `concepts.py` | Original notes + 6 binary flag columns |
| `checkpoints/best.pt` | `train_single_mvit_cv.py` | Best model weights + epoch + val loss |
| `test_predictions.csv` | Both training and eval scripts | Per-sample probabilities and predictions |
| `test_metrics.csv` | `evaluate_kidney.py` | Per-label and macro-average metrics table |

---

## Notes

- CT volumes and masks should be in NIfTI format (`.nii` or `.nii.gz`), oriented so that the axial plane is along the Z axis.
- Kidney masks can be generated by any segmentation tool (e.g. [TotalSegmentator](https://github.com/wasserth/TotalSegmentator)); label values > 0 are treated as kidney.
- The `stent` and `tube` flags extracted by `concepts.py` are not used as training labels for the MViT — they can be incorporated by adding them to `LABEL_COLS` in the training script.