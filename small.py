#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ADHD-only trial pipeline (Kaggle: jerseyneo/reddit-adhd-dataset)
- Download + parse CSVs (multi-threaded)
- Light cleaning (URLs/emails/mentions/hashtags/emojis, whitespace)
- Build "Post/Task/Answer" supervision lines entirely in RAM
- Shuffle + split into train/val (by examples)
- Tokenize for GPT-2 **matching the docs behavior**:
    tokens = [EOT] + enc.encode_ordinary(sample + ("\n\n" unless last))
  then pack into fixed ctx=1024 blocks for llm.c

Outputs:
  <train-out>.txt  + <train-out>.gpt2.bin
  <val-out>.txt    + <val-out>.gpt2.bin
"""

import os, re, csv, argparse, hashlib, random, math
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from threading import Thread

import numpy as np

# --- external deps you need installed ---
# pip install kaggle tiktoken pandas

try:
    import pandas as pd
except Exception:
    pd = None

import tiktoken

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ---------------------------
# CLI
# ---------------------------
def build_argparser():
    p = argparse.ArgumentParser(
        description="ADHD-only trial: RAM-first build + GPT-2 tokenization (docs-accurate) with fixed ctx packing."
    )
    p.add_argument("--train-out", type=str, default="dataset/data/training_data.txt")
    p.add_argument("--val-out",   type=str, default="dataset/data/validation_data.txt")
    p.add_argument("--val-ratio", type=float, default=0.10)
    p.add_argument("--seed",      type=int, default=42)

    # Multithreading
    cpu = os.cpu_count() or 8
    p.add_argument("--workers",       type=int, default=min(16, cpu), help="Threads for CSV parsing / row processing.")
    p.add_argument("--tok-workers",   type=int, default=min(24, max(2, cpu-4)), help="Threads for tokenization segments.")
    p.add_argument("--segments",      type=int, default=0, help="Override number of tokenization segments (0=auto=w*4).")
    p.add_argument("--queue-size",    type=int, default=1024, help="Writer queue size (token arrays).")

    # Safety/volume
    p.add_argument("--max-examples",  type=int, default=300_000,
                   help="Optional global cap for examples to keep RAM/bandwidth in check (0 = no cap).")

    # GPT-2 packing
    p.add_argument("--ctx", type=int, default=1024, help="Fixed context window for packing.")
    p.add_argument("--drop-last", action="store_true", help="Drop final partial block (<ctx tokens).")
    return p


# ---------------------------
# Light text cleaning
# ---------------------------
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


# ---------------------------
# Kaggle helper
# ---------------------------
def kaggle_download(slug: str, dest: Path):
    if dest.exists() and any(dest.iterdir()):
        print(f"[+] {slug} already in {dest}")
        return
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[-] Downloading {slug} ...")
    code = os.system(f'kaggle datasets download -d "{slug}" -p "{dest}" --quiet --unzip')
    if code != 0:
        raise RuntimeError(f"Failed to download {slug}. Check Kaggle CLI & credentials.")


# ---------------------------
# ADHD dataset (jerseyneo)
# ---------------------------
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

def build_adhd_examples(workers: int, max_examples: int) -> list[str]:
    slug = "jerseyneo/reddit-adhd-dataset"
    root = Path("data_cache/reddit_adhd_dataset")
    kaggle_download(slug, root)
    csvs = sorted(root.glob("*.csv"))
    if not csvs:
        raise RuntimeError("No CSVs found in ADHD dataset.")

    print(f"[build] ADHD (jerseyneo) | files={len(csvs)} | workers={workers}")
    # read all rows (parallel over files)
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(read_csv_any, p): p for p in csvs}
        for fut in as_completed(futs):
            part = fut.result() or []
            rows.extend(part)

    print(f"[build] raw rows: {len(rows):,}")

    # process rows to supervision lines (parallel)
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
        # Supervision triple
        post = text
        task = "Identify which mental health condition this post is about."
        answer = "ADHD"
        return f"Post: {post}\nTask: {task}\nAnswer: {answer}<|endoftext|>"

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


# ---------------------------
# GPT-2 tokenization (docs-accurate) + fixed-ctx packing
# ---------------------------
EOT_STR = "<|endoftext|>"

def _split_ranges(n, parts):
    if parts <= 1: return [(0, n)]
    step = math.ceil(n / parts)
    return [(i, min(i+step, n)) for i in range(0, n, step)]

def _writer_ctx(out_path, ctx, drop_last, q, n_sentinels):
    buf = np.empty(ctx, dtype=np.uint16)
    buf_len = 0
    chunks = 0
    with open(out_path, "wb") as fout:
        done = 0
        while True:
            arr = q.get()
            if arr is None:
                done += 1
                if done == n_sentinels:
                    break
                continue
            i = 0
            n = arr.size
            while i < n:
                take = min(ctx - buf_len, n - i)
                if take:
                    buf[buf_len:buf_len+take] = arr[i:i+take]
                    buf_len += take
                    i += take
                if buf_len == ctx:
                    buf.tofile(fout)
                    buf_len = 0
                    chunks += 1
    if buf_len and not drop_last:
        with open(out_path, "ab") as fout:
            buf[:buf_len].tofile(fout)
            chunks += 1
    print(f"[gpt2-write] {out_path} | chunks={chunks} | ctx={ctx}")

def tokenize_lines_gpt2_match_docs(lines, out_bin, ctx=1024, workers=8, segments=0, queue_size=1024, drop_last=False):
    """
    EXACT docs behavior for GPT-2:
        eot = enc._special_tokens['<|endoftext|>']
        for each sample_i:
            tokens = [eot] + enc.encode_ordinary(sample_i + ('\\n\\n' unless last))
    Then pack sequentially into fixed ctx windows.
    """
    if not lines:
        open(out_bin, "wb").close(); return
    enc = tiktoken.get_encoding("gpt2")
    eot_id = enc._special_tokens[EOT_STR]

    w = max(1, int(workers))
    segs = segments if segments and segments > 0 else w * 4
    ranges = _split_ranges(len(lines), segs)
    print(f"[gpt2] -> {out_bin} | ctx={ctx} | workers={w} | segments={len(ranges)}")

    q = Queue(maxsize=queue_size)
    wt = Thread(target=_writer_ctx, args=(out_bin, ctx, drop_last, q, len(ranges)), daemon=True)
    wt.start()

    def encode_slice(a, b):
        # note: we must know if 'i' is the global last index to apply the '\n\n' quirk
        n_total = len(lines)
        for i in range(a, b):
            s = lines[i]
            spad = s + "\n\n" if i != (n_total - 1) else s
            ids = [eot_id] + enc.encode_ordinary(spad)
            q.put(np.asarray(ids, dtype=np.uint16))
        q.put(None)

    with ThreadPoolExecutor(max_workers=w) as ex:
        for a, b in ranges:
            ex.submit(encode_slice, a, b)

    wt.join()


# ---------------------------
# MAIN
# ---------------------------
def main():
    ap = build_argparser()
    args = ap.parse_args()
    random.seed(args.seed)

    train_path = Path(args.train_out); train_path.parent.mkdir(parents=True, exist_ok=True)
    val_path   = Path(args.val_out);   val_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) Build ADHD examples in RAM (capped for trial by --max-examples)
    examples = build_adhd_examples(args.workers, args.max_examples)

    if not examples:
        print("[fatal] no examples produced."); return

    # 2) Dedupe (sha1)
    seen = set(); deduped = []
    for s in examples:
        h = hashlib.sha1(s.encode("utf-8")).hexdigest()
        if h in seen: continue
        seen.add(h); deduped.append(s)
    examples = deduped
    print(f"[dedupe] kept={len(examples):,}")

    # 3) Shuffle + split (by examples)
    rng = random.Random(args.seed)
    rng.shuffle(examples)
    split = int(len(examples) * (1.0 - args.val_ratio))
    train_lines = examples[:split]
    val_lines   = examples[split:]
    print(f"[split] train={len(train_lines):,}  val={len(val_lines):,}  (val-ratio={args.val_ratio})")

    # 4) Write text
    train_path.write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    val_path.write_text("\n".join(val_lines) + "\n", encoding="utf-8")
    print(f"[write] {train_path} , {val_path}")

    # 5) GPT-2 tokenization (docs-accurate) + fixed-ctx packing
    train_bin = train_path.with_suffix(".gpt2.bin")
    val_bin   = val_path.with_suffix(".gpt2.bin")

    tokenize_lines_gpt2_match_docs(
        train_lines, train_bin,
        ctx=args.ctx, workers=args.tok_workers, segments=args.segments,
        queue_size=args.queue_size, drop_last=args.drop_last
    )
    tokenize_lines_gpt2_match_docs(
        val_lines, val_bin,
        ctx=args.ctx, workers=max(2, args.tok_workers//2), segments=args.segments//2 if args.segments else 0,
        queue_size=max(256, args.queue_size//2), drop_last=args.drop_last
    )

    print("[done] ADHD trial finished.")

if __name__ == "__main__":
    main()