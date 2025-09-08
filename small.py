#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ADHD-only trial for llm.c
- Download jerseyneo/reddit-adhd-dataset (Kaggle)
- Build "Post/Task/Answer" examples (RAM)
- Shuffle + split
- Tokenize for GPT-2 exactly like llm.c docs (EOT + encode_ordinary + '\n\n' quirk)
- Write *.gpt2.bin using data_common.write_datafile (adds the required header)

Outputs:
  dataset/data/training_data.gpt2.bin
  dataset/data/validation_data.gpt2.bin
"""

import os, sys, re, csv, math, argparse, hashlib, random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import tiktoken

# --------------------------------------------------------------------------------------
# Robust import of write_datafile from llm.c repo
# --------------------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
CANDIDATES = [
    HERE,                     # repo root
    HERE / "data",            # llm.c/data/data_common.py (usual location)
]
for cand in CANDIDATES:
    if (cand / "data_common.py").exists():
        sys.path.insert(0, str(cand))
        break

try:
    from data_common import write_datafile  # provided by llm.c repo
except Exception as e:
    raise SystemExit(
        "Could not import write_datafile from data_common.py.\n"
        "Put this script in the llm.c repo root and make sure data/data_common.py exists.\n"
        f"Details: {e}"
    )

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def build_argparser():
    p = argparse.ArgumentParser(
        description="ADHD-only trial → GPT-2 tokenized .bin for llm.c (with header)."
    )
    p.add_argument("--train-out", type=str, default="dataset/data/training_data.gpt2.bin")
    p.add_argument("--val-out",   type=str, default="dataset/data/validation_data.gpt2.bin")
    p.add_argument("--val-ratio", type=float, default=0.10)
    p.add_argument("--seed",      type=int, default=42)

    cpu = os.cpu_count() or 8
    p.add_argument("--workers",     type=int, default=min(16, cpu), help="Threads for CSV parsing / row processing.")
    p.add_argument("--tok-workers", type=int, default=min(24, max(2, cpu - 4)), help="Threads for tokenization.")
    p.add_argument("--max-examples", type=int, default=200_000,
                   help="Cap examples to stay within RAM. Set 0 for all.")
    return p

# --------------------------------------------------------------------------------------
# Light cleaning (keep semantics)
# --------------------------------------------------------------------------------------
_URL_RE      = re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE)
_EMAIL_RE    = re.compile(r'\b[\w\.-]+@[\w\.-]+\.\w+\b')
_MENTION_RE  = re.compile(r'@\w+')
_HASHTAG_RE  = re.compile(r'#\w+')
_MULTI_WS_RE = re.compile(r'\s+')
_EMOJI_RE    = re.compile(
    "["  # emoji / pictographs
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002700-\U000027BF"
    "\U000024C2-\U0001F251"
    "]+", flags=re.UNICODE
)

def clean_text(x: str) -> str:
    if not x:
        return ""
    x = _URL_RE.sub(" ", x)
    x = _EMAIL_RE.sub(" ", x)
    x = _MENTION_RE.sub(" ", x)
    x = _HASHTAG_RE.sub(" ", x)
    x = _EMOJI_RE.sub(" ", x)
    x = _MULTI_WS_RE.sub(" ", x).strip()
    return x

# --------------------------------------------------------------------------------------
# Kaggle helper
# --------------------------------------------------------------------------------------
def kaggle_download(slug: str, dest: Path):
    if dest.exists() and any(dest.iterdir()):
        print(f"[+] {slug} already in {dest}")
        return
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[-] Downloading {slug} ...")
    code = os.system(f'kaggle datasets download -d "{slug}" -p "{dest}" --quiet --unzip')
    if code != 0:
        raise RuntimeError(f"Failed to download {slug}. Check Kaggle CLI & credentials.")

# --------------------------------------------------------------------------------------
# CSV reader
# --------------------------------------------------------------------------------------
def read_csv_any(path: Path):
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            sample = f.read(4096); f.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample)
            except Exception:
                dialect = csv.excel
            return list(csv.DictReader(f, dialect=dialect))
    except Exception:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            return list(csv.DictReader(f))

# --------------------------------------------------------------------------------------
# Build ADHD supervision examples
# --------------------------------------------------------------------------------------
def build_adhd_examples(workers: int, max_examples: int) -> list[str]:
    slug = "jerseyneo/reddit-adhd-dataset"
    root = Path("data_cache/reddit_adhd_dataset")
    kaggle_download(slug, root)
    csvs = sorted(root.glob("*.csv"))
    if not csvs:
        raise RuntimeError("No CSVs found in ADHD dataset.")

    print(f"[build] ADHD (jerseyneo) | files={len(csvs)} | workers={workers}")
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(read_csv_any, p): p for p in csvs}
        for fut in as_completed(futs):
            part = fut.result() or []
            rows.extend(part)

    print(f"[build] raw rows: {len(rows):,}")

    cand_fields = ("selftext","body","comment","text","content","message","post","description")

    def proc(row):
        raw = None
        for k in cand_fields:
            v = row.get(k)
            if v and str(v).strip():
                raw = str(v); break
        if not raw:
            joined = " ".join(str(row.get(k,"")).strip()
                              for k in ("title","selftext","body")
                              if str(row.get(k,"")).strip()).strip()
            raw = joined if joined else None
        if not raw:
            return None
        title = (row.get("title") or "")
        text = clean_text(f"{title}\n{raw}" if title and title not in raw else raw)
        if not text:
            return None
        post = text
        task = "Identify which mental health condition this post is about."
        answer = "ADHD"
        return f"Post: {post}\nTask: {task}\nAnswer: {answer}"

    out = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(proc, r) for r in rows]
        for fut in as_completed(futs):
            s = fut.result()
            if s:
                out.append(s)
            if max_examples and len(out) >= max_examples:
                break

    print(f"[build] ADHD examples: {len(out):,}")
    return out

# --------------------------------------------------------------------------------------
# GPT-2 tokenization (exactly like docs) and write with header
# --------------------------------------------------------------------------------------
def tokenize_gpt2_docs_style(sections: list[str], workers: int) -> list[int]:
    """
    Docs behavior:
        enc = tiktoken.get_encoding("gpt2")
        eot = enc._special_tokens['<|endoftext|>']
        tokens = []
        for i, s in enumerate(sections):
            tokens.append(eot)
            spad = s + "\\n\\n" if i != len(sections) - 1 else s
            tokens.extend(enc.encode_ordinary(spad))
    """
    enc = tiktoken.get_encoding("gpt2")
    eot_id = enc._special_tokens['<|endoftext|>']

    n = len(sections)
    if n == 0:
        return []

    parts = max(1, workers)
    step = math.ceil(n / parts)
    ranges = [(i, min(i+step, n)) for i in range(0, n, step)]

    def encode_range(a, b):
        local = []
        for i in range(a, b):
            s = sections[i]
            local.append(eot_id)
            spad = s + "\n\n" if i != (n - 1) else s
            local.extend(enc.encode_ordinary(spad))
        return local

    out_parts = [None] * len(ranges)
    with ThreadPoolExecutor(max_workers=parts) as ex:
        futs = []
        for idx, (a, b) in enumerate(ranges):
            futs.append((idx, ex.submit(encode_range, a, b)))
        for idx, fut in futs:
            out_parts[idx] = fut.result()

    tokens = []
    for part in out_parts:
        tokens.extend(part)
    return tokens

# --------------------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------------------
def main():
    ap = build_argparser()
    args = ap.parse_args()
    random.seed(args.seed)

    out_train = Path(args.train_out); out_train.parent.mkdir(parents=True, exist_ok=True)
    out_val   = Path(args.val_out);   out_val.parent.mkdir(parents=True, exist_ok=True)

    sections = build_adhd_examples(args.workers, args.max_examples)
    if not sections:
        print("[fatal] no examples produced."); return

    # dedupe by sha1
    seen = set(); uniq = []
    for s in sections:
        h = hashlib.sha1(s.encode("utf-8")).hexdigest()
        if h in seen: continue
        seen.add(h); uniq.append(s)
    sections = uniq
    print(f"[dedupe] kept={len(sections):,}")

    # shuffle + split
    rng = random.Random(args.seed)
    rng.shuffle(sections)
    split = int(len(sections) * (1.0 - args.val_ratio))
    train_secs = sections[:split]
    val_secs   = sections[split:]
    print(f"[split] train={len(train_secs):,}  val={len(val_secs):,}  (val-ratio={args.val_ratio})")

    # tokenize (GPT-2 docs-accurate) and write with header
    print("[tok] encoding train...")
    train_tokens = tokenize_gpt2_docs_style(train_secs, args.tok_workers)
    print(f"[tok] train tokens: {len(train_tokens):,}")
    write_datafile(str(out_train), train_tokens, "gpt-2")
    print(f"[write] {out_train}")

    print("[tok] encoding val...")
    val_tokens = tokenize_gpt2_docs_style(val_secs, max(2, args.tok_workers // 2))
    print(f"[tok] val tokens: {len(val_tokens):,}")
    write_datafile(str(out_val), val_tokens, "gpt-2")
    print(f"[write] {out_val}")

    print("[done] ADHD GPT-2 data ready for llm.c")

if __name__ == "__main__":
    main()