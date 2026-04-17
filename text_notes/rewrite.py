import re
import torch
import pandas as pd
from transformers import pipeline

merged_df = pd.read_csv("notesouttolook.csv")

# new column for rewritten note
merged_df["renal_focused_note"] = ""

pipe = pipeline(
    "text-generation",
    model="google/medgemma-27b-text-it",
    torch_dtype=torch.bfloat16,
    device=0,
)

def build_prompt(note: str) -> str:
    return (
        "You are a medical language model specializing in radiology report rewriting.\n\n"
        "TASK:\n"
        "Rewrite the radiology note so that it contains ONLY clinically relevant findings "
        "related to the kidneys, ureters, and bladder.\n\n"
        "RULES:\n"
        "- Remove any mention of prior imaging studies or comparisons\n"
        "- Remove dates, times, and temporal references\n"
        "- Remove non-genitourinary findings (e.g., liver, spleen, lungs, bowel)\n"
        "- Keep positive AND explicitly stated negative renal/urinary findings\n"
        "- Use concise, professional radiology language\n"
        "- Do NOT add new information\n"
        "- Do NOT mention that content was removed\n"
        "- Do NOT reference prior exams\n\n"
        "OUTPUT FORMAT:\n"
        "Return ONLY the rewritten note text. No JSON. No bullet points. No explanations.\n\n"
        "ORIGINAL NOTE:\n"
        f"\"\"\"{note}\"\"\"\n\n"
        "REWRITTEN NOTE:"
    )
def rewrite_note(note: str) -> str:
    prompt = build_prompt(note[:8000])  # safety cap

    outputs = pipe(
        prompt,
        max_new_tokens=256,
        do_sample=False,
        temperature=0.0,
        return_full_text=False,
    )

    text = outputs[0]["generated_text"].strip()

    # light cleanup: collapse whitespace
    text = re.sub(r"\n{2,}", "\n", text).strip()
    return text
for idx, row in merged_df.iterrows():
    try:
        rewritten = rewrite_note(str(row["note"]))
        merged_df.at[idx, "renal_focused_note"] = rewritten

        if idx < 3:
            print(f"\n--- Row {idx} acc_num={row.get('acc_num', 'NA')} ---")
            print("REWRITTEN NOTE:")
            print(rewritten)

    except Exception as e:
        print(f"[WARN] Row {idx} failed: {e}")
merged_df.to_csv("notes_medgemma27_renal_rewrite.csv", index=False)
print("Saved: notes_medgemma27_renal_rewrite.csv")
