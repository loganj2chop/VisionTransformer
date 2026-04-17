import re
import json
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# ============================================================
# CONFIG
# ============================================================
MODEL_ID       = "google/medgemma-27b-text-it"
INPUT_CSV      = "vtnotes2.csv"
OUTPUT_CSV     = "notes_medgemma27_concepts_redo.csv"
CHECKPOINT_N   = 500
MAX_NOTE_CHARS = 6000

# ============================================================
# FLAG DEFINITIONS
# Only 6 flags — keep prompt focused to reduce confusion
# Stone location distinction is explicitly reinforced in prompt
# ============================================================
FLAGS = {
    "kidney_stone":           "Stone located WITHIN the kidney (renal parenchyma or collecting system). Do NOT set this if the stone is in the ureter.",
    "ureteral_stone":         "Stone located in the ureter (proximal, mid, or distal ureter). Do NOT set this if the stone is in the kidney.",
    "hydronephrosis":         "Hydronephrosis — dilation of the renal pelvis and calyces due to obstruction.",
    "urinary_tract_dilation": "Dilation of the urinary tract including hydroureter or ureteral dilation, not caused by a stone.",
    "stent":                  "Ureteral stent present (DJ stent, double-J stent, or similar indwelling ureteral device).",
    "tube":                   "Drainage tube present — nephrostomy tube, percutaneous nephrostomy, or Foley catheter.",
}

FLAG_COLS = list(FLAGS.keys())

# ============================================================
# LOAD MODEL
# ============================================================
print("Loading tokenizer and model …")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()

# ============================================================
# PROMPT BUILDER
# ============================================================
def build_messages(note: str) -> list[dict]:
    findings_block = "\n".join(
        f'  "{k}": "{desc}"' for k, desc in FLAGS.items()
    )
    json_template = "{\n" + ",\n".join(f'  "{k}": 0' for k in FLAG_COLS) + "\n}"

    system = (
        "You are a medical NLP system that extracts structured findings from radiology reports.\n"
        "Rules:\n"
        "  1. Set a flag to 1 if the finding is PRESENT (including post-operative or historical mention).\n"
        "  2. Set a flag to 0 if ABSENT, explicitly negated, or not mentioned.\n"
        "  3. IMPORTANT — stone location: 'kidney_stone' and 'ureteral_stone' are mutually exclusive.\n"
        "     - kidney_stone = 1 ONLY if the stone is described as within the kidney.\n"
        "     - ureteral_stone = 1 ONLY if the stone is described as within the ureter.\n"
        "     - If both locations are mentioned, set both to 1.\n"
        "     - If location is ambiguous, set kidney_stone = 1 and ureteral_stone = 0.\n"
        "  4. Return ONLY valid JSON — no markdown, no backticks, no explanation."
    )

    user = (
        f"Findings to extract:\n{{\n{findings_block}\n}}\n\n"
        f"Radiology note:\n\"\"\"\n{note}\n\"\"\"\n\n"
        f"Return ONLY this JSON (set 0 or 1 for each key):\n{json_template}"
    )

    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]


# ============================================================
# INFERENCE
# ============================================================
@torch.inference_mode()
def classify_note(note: str) -> dict:
    messages = build_messages(note[:MAX_NOTE_CHARS])

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    attention_mask = (input_ids != tokenizer.eos_token_id).long()

    output_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=150,        # 6 flags needs ~80 tokens — 150 is plenty
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    gen_ids  = output_ids[0][input_ids.shape[-1]:]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    print(gen_text)

    return extract_json(gen_text)


# ============================================================
# ROBUST JSON EXTRACTION WITH PARTIAL RECOVERY
# ============================================================
def extract_json(text: str) -> dict:
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip()

    obj = None

    # Try full JSON first
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError as e:
            print(f"  [WARN] JSON decode error ({e}): {m.group(0)[:200]!r}")

    # Partial recovery — model cut off mid-JSON
    if obj is None:
        m = re.search(r"\{.*", text, flags=re.DOTALL)
        if m:
            partial = m.group(0).rstrip(",\n ") + "\n}"
            try:
                obj = json.loads(partial)
                print("  [WARN] Truncated JSON — partial recovery succeeded")
            except json.JSONDecodeError:
                print(f"  [WARN] Partial recovery failed: {partial[:200]!r}")

    if obj is None:
        print(f"  [WARN] No JSON found in: {text[:200]!r}")
        return {k: 0 for k in FLAG_COLS}

    out = {}
    for k in FLAG_COLS:
        v = obj.get(k, 0)
        if isinstance(v, bool):
            out[k] = int(v)
        elif isinstance(v, (int, float)):
            out[k] = int(bool(v))
        elif isinstance(v, str):
            out[k] = 1 if v.strip().lower() in ("1", "true", "yes", "present") else 0
        else:
            out[k] = 0

    return out


# ============================================================
# LOAD CSV
# ============================================================
print(f"Loading {INPUT_CSV} …")
merged_df = pd.read_csv(
    INPUT_CSV,
    encoding="cp1252",
    dtype=str,
    low_memory=False,
)
merged_df["note"] = (
    merged_df["note"]
    .fillna("")
    .str.replace(r"[\x00-\x1f\x7f-\x9f]", " ", regex=True)
)

for c in FLAG_COLS:
    merged_df[c] = 0

# ============================================================
# RUN
# ============================================================
print(f"Classifying {len(merged_df)} rows with {len(FLAG_COLS)} flags each …")

for idx, row in tqdm(merged_df.iterrows(), total=len(merged_df)):
    preds = classify_note(row["note"])
    for k, v in preds.items():
        merged_df.at[idx, k] = v

    if idx < 3:
        print(f"\n--- Row {idx} ---")
        print({k: v for k, v in preds.items() if v == 1} or "(all zeros)")

    if (idx + 1) % CHECKPOINT_N == 0:
        merged_df.to_csv(OUTPUT_CSV, index=False)
        print(f"  [checkpoint] saved at row {idx + 1}")

# ============================================================
# SAVE FINAL
# ============================================================
merged_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nDone. Saved to: {OUTPUT_CSV}")
print(f"Shape: {merged_df.shape}")
print(f"\nFlag prevalence (% positive):")
print(
    (merged_df[FLAG_COLS].astype(int).mean() * 100)
    .sort_values(ascending=False)
    .apply(lambda x: f"{x:.1f}%")
    .to_string()
)