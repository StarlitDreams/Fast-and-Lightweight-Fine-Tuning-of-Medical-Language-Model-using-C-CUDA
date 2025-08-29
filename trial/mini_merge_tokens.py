#!/usr/bin/env python3
# Tiny one-dataset pipeline → train/val txt + uint16 .bin
import os, csv, time, argparse, random
from pathlib import Path
import numpy as np

# --- tokenizer ---
try:
    from transformers import GPT2TokenizerFast
except ImportError:
    raise SystemExit("Please: pip install transformers tokenizers")

# --- args ---
ap = argparse.ArgumentParser(description="Mini trial: one small Kaggle dataset to txt + bin")
ap.add_argument("--out-dir", default="dataset/minor", help="Output folder for trial files")
ap.add_argument("--val-ratio", type=float, default=0.10, help="Validation split ratio")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--max-examples", type=int, default=10000, help="Cap examples to keep it fast")
ap.add_argument("--cache-dir", default="data_cache/minor_anxiety", help="Where to cache the dataset")
ap.add_argument("--slug", default="michellevp/predicting-anxiety-in-mental-health-data",
                help="Kaggle dataset slug (kept small by default)")
args = ap.parse_args()
random.seed(args.seed)

out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
train_txt = out_dir / "trial_train.txt"
val_txt   = out_dir / "trial_val.txt"
train_bin = out_dir / "trial_train.bin"
val_bin   = out_dir / "trial_val.bin"

cache_dir = Path(args.cache_dir)

def kaggle_download(slug: str, dest: Path):
    if dest.exists() and any(dest.iterdir()):
        print(f"[+] {slug} already in {dest}")
        return
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[-] Downloading {slug} ...")
    code = os.system(f'kaggle datasets download -d "{slug}" -p "{dest}" --quiet --unzip')
    if code != 0:
        raise SystemExit(f"Failed to download {slug}. Ensure Kaggle CLI & credentials are set.")

def read_csv_any(path: Path):
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        sample = f.read(4096); f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
        except Exception:
            dialect = csv.excel
        return list(csv.DictReader(f, dialect=dialect))

def build_example(text: str, raw_label: str):
    text = (text or "").strip().replace("\n", " ")
    if not text or len(text) < 10:
        return None
    val = (raw_label or "").strip().lower()
    ans = "Yes" if val in ("1","true","yes","anxiety") else "No"
    return f"Post: {text}\nTask: Does the following post indicate an anxiety disorder? Answer Yes or No.\nAnswer: {ans}<|endoftext|>"

def stream_tokenize_txt_to_bin(txt_path: Path, bin_path: Path, report_every=2000):
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    if tok.vocab_size > 65535:
        raise ValueError("Vocab exceeds uint16; change dtype if you swap tokenizers.")
    print(f"[tokenize] {txt_path} -> {bin_path}")
    total_lines = 0
    total_tokens = 0
    t0 = time.time()
    with txt_path.open("r", encoding="utf-8") as fin, open(bin_path, "wb") as fout:
        for line in fin:
            ids = tok.encode(line)
            np.asarray(ids, dtype=np.uint16).tofile(fout)
            total_lines += 1
            total_tokens += len(ids)
            if total_lines % report_every == 0:
                dt = time.time() - t0
                rate = total_tokens / dt if dt else 0
                print(f"  > lines={total_lines:,} tokens={total_tokens:,} ~{rate:,.0f} tok/s")
    dt = time.time() - t0
    print(f"[done] lines={total_lines:,} tokens={total_tokens:,} in {dt:,.1f}s")

# --- 1) get one small dataset ---
kaggle_download(args.slug, cache_dir)
candidate = next((p for p in cache_dir.glob("*.csv")), None)
if candidate is None:
    raise SystemExit(f"No CSV found in {cache_dir}")

print(f"[+] Using {candidate.name}")

# --- 2) build examples (capped) ---
rows = read_csv_any(candidate)
examples = []
for r in rows:
    text = (r.get("text") or r.get("post") or r.get("content") or "")
    label = (r.get("anxiety") or r.get("label") or r.get("class") or "")
    ex = build_example(text, label)
    if ex:
        examples.append(ex)
    if len(examples) >= args.max_examples:
        break

if not examples:
    raise SystemExit("No usable examples produced.")

random.shuffle(examples)
split = int(len(examples) * (1.0 - args.val_ratio))
train, val = examples[:split], examples[split:]

# --- 3) write txt ---
train_txt.write_text("\n".join(train) + "\n", encoding="utf-8")
val_txt.write_text("\n".join(val) + "\n", encoding="utf-8")
print(f"[write] train={train_txt} ({len(train):,} lines) | val={val_txt} ({len(val):,} lines)")

# --- 4) tokenize to .bin ---
stream_tokenize_txt_to_bin(train_txt, train_bin)
stream_tokenize_txt_to_bin(val_txt, val_bin)

print("[all good] mini trial finished.")
