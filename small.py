#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Process ONLY jerseyneo/reddit-adhd-dataset and tokenize to llm.c GPT-2 .bin shards.

Pipeline
  1) Kaggle download (once) -> data_cache/reddit_adhd_dataset
  2) Parse all CSVs (threaded per-row), extract (title + text)
  3) Light clean: strip URLs/emails/mentions/hashtags, collapse whitespace
     (no lemmatization, no stopword removal, keep natural text)
  4) On-the-fly split to train/val and SHA1 dedupe
  5) Tokenize with tiktoken(gpt2) EXACTLY like the docs: prepend EOT (id=50256)
     to every example, then encode_ordinary(text)
  6) Write sharded .bin with llm.c header via dev/data/data_common.write_datafile
  7) (Optional) write .txt mirrors for inspection

Run:
  python3 adhd_only_tokenize.py \
    --out-dir dataset/data \
    --val-ratio 0.10 \
    --shard-tokens 1500000 \
    --workers 16
"""

import os, sys, csv, re, argparse, hashlib, random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------- llm.c helper (dev/data/data_common.py) ----------
DEV_DATA_DIR = Path(__file__).resolve().parent / "dev" / "data"
if (DEV_DATA_DIR / "data_common.py").exists() and str(DEV_DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DEV_DATA_DIR))
try:
    from data_common import write_datafile  # provided by llm.c
except Exception as e:
    print("Could not import write_datafile from dev/data/data_common.py.\n"
          "Place this script in the llm.c repo root so dev/data/data_common.py is importable.\n"
          f"Details: {e}")
    sys.exit(1)

# ---------- external tokenization ----------
try:
    import tiktoken
except Exception as e:
    print("Missing dependency: tiktoken (pip install tiktoken)\n", e)
    sys.exit(1)

# ---------- config ----------
cache_dir = Path("data_cache")
ds_dir = cache_dir / "reddit_adhd_dataset"
slug = "jerseyneo/reddit-adhd-dataset"

# text cleaning (very light)
_URL_RE      = re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE)
_EMAIL_RE    = re.compile(r'\b[\w\.-]+@[\w\.-]+\.\w+\b')
_MENTION_RE  = re.compile(r'@\w+')
_HASHTAG_RE  = re.compile(r'#\w+')
_MULTI_WS_RE = re.compile(r'\s+')

CAND_TEXT_FIELDS = ("selftext","body","comment","text","content","message","post","description")

# ---------- helpers ----------
def kaggle_download():
    if ds_dir.exists() and any(ds_dir.iterdir()):
        print(f"[+] {slug} already in {ds_dir}")
        return
    ds_dir.mkdir(parents=True, exist_ok=True)
    print(f"[-] Downloading {slug} ...")
    code = os.system(f'kaggle datasets download -d "{slug}" -p "{ds_dir}" --quiet --unzip')
    if code != 0:
        raise RuntimeError("Kaggle download failed. Ensure kaggle CLI and credentials are set.")

def sniff_reader(path: Path):
    with path.open('r', encoding='utf-8', errors='ignore') as f:
        sample = f.read(8192); f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample)
        except Exception:
            dialect = csv.excel
        for row in csv.DictReader(f, dialect=dialect):
            yield row

def clean_join_title_body(row) -> str | None:
    """Return a single natural text string (title + body) or None."""
    title = (row.get("title") or "").strip()
    raw = None
    for k in CAND_TEXT_FIELDS:
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
    text = (f"{title}\n{raw}" if title and title not in raw else raw)
    # light normalization only
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    text = _HASHTAG_RE.sub(" ", text)
    text = _MULTI_WS_RE.sub(" ", text).strip()
    return text if text else None

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode('utf-8')).hexdigest()

# ---------- token writer (sharded) ----------
class ShardedWriter:
    """Accumulates tokens and writes to .bin shards with llm.c header."""
    def __init__(self, out_prefix: Path, model_desc: str, shard_tokens: int):
        self.out_prefix = out_prefix
        self.model_desc = model_desc
        self.shard_tokens = shard_tokens
        self.buf = []
        self.total = 0
        self.idx = 1
        self.written_files = []

    def add(self, token_ids):
        self.buf.extend(token_ids)
        if len(self.buf) >= self.shard_tokens:
            self.flush()

    def flush(self):
        if not self.buf: return
        out_path = Path(f"{self.out_prefix}.gpt2-{self.idx:05d}.bin")
        write_datafile(str(out_path), self.buf, self.model_desc)
        self.written_files.append(out_path)
        self.total += len(self.buf)
        print(f"[write] {out_path.name}  | tokens={len(self.buf):,}  | total={self.total:,}")
        self.idx += 1
        self.buf = []

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser("ADHD-only: process + tokenize for llm.c (GPT-2).")
    ap.add_argument("--out-dir", type=str, default="dataset/data", help="Output directory for .bin and optional .txt")
    ap.add_argument("--val-ratio", type=float, default=0.10, help="Fraction to route to validation")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for split")
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 8), help="Threads for row processing")
    ap.add_argument("--dedupe-cap", type=int, default=2_000_000, help="Max texts to track for dedupe")
    ap.add_argument("--max-examples", type=int, default=0, help="Optional cap for quick trials (0=no cap)")
    ap.add_argument("--write-txt", action="store_true", help="Also write .txt mirrors for inspection")
    ap.add_argument("--shard-tokens", type=int, default=1_500_000, help="Max tokens per .bin shard")
    args = ap.parse_args()

    random.seed(args.seed)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    kaggle_download()
    csvs = sorted(ds_dir.glob("*.csv"))
    if not csvs:
        print("[!] No CSVs found in dataset; abort.")
        sys.exit(1)

    # tokenization setup (docs-correct)
    enc = tiktoken.get_encoding("gpt2")
    EOT = enc._special_tokens['<|endoftext|>']  # 50256

    # sharded writers
    train_prefix = Path(args.out_dir) / "training_data"
    val_prefix   = Path(args.out_dir) / "validation_data"
    train_writer = ShardedWriter(train_prefix, "gpt-2", args.shard_tokens)
    val_writer   = ShardedWriter(val_prefix,   "gpt-2", args.shard_tokens)

    # optional .txt mirrors
    if args.write_txt:
        ftrain = (Path(args.out_dir) / "adhd_training_data.txt").open("w", encoding="utf-8")
        fval   = (Path(args.out_dir) / "adhd_validation_data.txt").open("w", encoding="utf-8")
    else:
        ftrain = fval = None

    # dedupe
    seen = set()
    keep = 0
    total = 0
    target_total = args.max_examples if args.max_examples > 0 else None

    def proc_row(row):
        text = clean_join_title_body(row)
        if not text:
            return None
        h = sha1(text)
        return (h, text)

    print(f"[build] ADHD (jerseyneo) | files={len(csvs)} | workers={args.workers}")
    for csv_path in csvs:
        print(f"    -> {csv_path.name}")
        rows = list(sniff_reader(csv_path))
        # process rows in threads
        out_pairs = []
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(proc_row, r) for r in rows]
            for fut in as_completed(futs):
                pair = fut.result()
                if pair: out_pairs.append(pair)

        # route examples
        for h, text in out_pairs:
            total += 1
            if len(seen) < args.dedupe_cap:
                if h in seen:
                    continue
                seen.add(h)

            # build a simple SFT-style triple (keeps semantics, easy for supervision)
            example = f"Post: {text}\nTask: Identify which mental health condition this post is about.\nAnswer: ADHD"
            # tokenize: prepend true EOT (docs behavior), then encode_ordinary
            toks = [EOT]
            toks.extend(enc.encode_ordinary(example))

            if random.random() < args.val_ratio:
                if fval:   fval.write(example + "\n")
                val_writer.add(toks)
            else:
                if ftrain: ftrain.write(example + "\n")
                train_writer.add(toks)

            keep += 1
            if target_total and keep >= target_total:
                break
        if target_total and keep >= target_total:
            break

    # final flush
    train_writer.flush()
    val_writer.flush()
    if ftrain: ftrain.close()
    if fval:   fval.close()

    print(f"[stats] input_rows≈{total:,}  kept={keep:,}  "
          f"train_shards={len(train_writer.written_files)}  val_shards={len(val_writer.written_files)}")
    print(f"[done] Train bins prefix: {train_prefix}.gpt2-*.bin")
    print(f"[done] Val   bins prefix: {val_prefix}.gpt2-*.bin")
    print("Hint: train with:")
    print(f'  ./train_gpt2cu -i "{train_prefix}.gpt2-*.bin" -j "{val_prefix}.gpt2-*.bin" -T 1024 ...')
    

if __name__ == "__main__":
    main()