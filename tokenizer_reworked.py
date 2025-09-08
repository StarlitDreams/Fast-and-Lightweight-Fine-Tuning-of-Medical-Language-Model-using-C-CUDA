#!/usr/bin/env python3
# RAM-first GPT-2 tokenizer with bounded memory:
# - reads entire .txt into RAM (as you wanted)
# - splits by literal "<|endoftext|>"
# - parallel encodes with tiktoken (threads)
# - hard-caps each sample to ctx<=1024 incl. real EOT
# - single writer packs fixed ctx blocks on the fly (no giant token buffer)

import os, sys, math, argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

import numpy as np

try:
    import tiktoken
except ImportError:
    print("pip install tiktoken"); sys.exit(1)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

def writer_thread(out_path: Path, ctx: int, drop_last: bool, q: Queue, n_sentinels: int):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    buf = np.empty(ctx, dtype=np.uint16)
    buf_len = 0
    chunks = 0
    done = 0
    max_seen = 0
    with out_path.open("wb") as fout:
        while True:
            item = q.get()
            if item is None:
                done += 1
                if done == n_sentinels:
                    break
                continue
            arr = item
            if arr.size == 0:
                continue
            if arr.size > ctx:  # safety belt
                arr = arr[:ctx]
            max_seen = max(max_seen, arr.size)
            i = 0
            while i < arr.size:
                take = min(ctx - buf_len, arr.size - i)
                if take:
                    buf[buf_len:buf_len+take] = arr[i:i+take]
                    buf_len += take
                    i += take
                if buf_len == ctx:
                    buf.tofile(fout)
                    buf_len = 0
                    chunks += 1
    if buf_len and not drop_last:
        with out_path.open("ab") as fout:
            buf[:buf_len].tofile(fout)
            chunks += 1
    print(f"[write] {out_path} | chunks={chunks} | max per-sample len={max_seen} (≤ {ctx})")

def encode_worker(sections, start, end, enc, eot_ids, ctx, q: Queue):
    allowed = {"<|endoftext|>"}
    cap = max(0, ctx - len(eot_ids))
    for i in range(start, end):
        s = sections[i].strip()
        if not s:
            continue
        ids = enc.encode(s, allowed_special=allowed)
        if len(ids) > cap:
            ids = ids[:cap]
        ids = ids + eot_ids
        q.put(np.asarray(ids, dtype=np.uint16))
    # signal this worker is done
    q.put(None)

def split_ranges(n_items: int, n_parts: int):
    if n_parts <= 1: return [(0, n_items)]
    step = math.ceil(n_items / n_parts)
    return [(i, min(i+step, n_items)) for i in range(0, n_items, step)]

def tokenize_file(in_txt: Path, ctx=1024, workers=16, segments=0, queue_size=2048, drop_last=False):
    if not in_txt.is_file():
        raise FileNotFoundError(in_txt)
    raw = in_txt.read_text(encoding="utf-8", errors="ignore")
    sections = raw.split("<|endoftext|>")
    n = len(sections)
    if n == 0:
        out_bin = in_txt.with_suffix(".bin")
        out_bin.write_bytes(b"")
        print(f"[empty] {in_txt} -> {out_bin}")
        return

    enc = tiktoken.get_encoding("gpt2")
    eot_ids = enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})

    workers = max(1, workers)
    segs = segments if segments and segments > 0 else workers * 4
    ranges = split_ranges(n, segs)

    out_bin = in_txt.with_suffix(".bin")
    print(f"[gpt2] {in_txt} -> {out_bin} | ctx={ctx} | workers={workers} | segments={len(ranges)} | q={queue_size}")

    q = Queue(maxsize=queue_size)

    # start writer
    from threading import Thread
    wt = Thread(target=writer_thread, args=(out_bin, ctx, drop_last, q, segs), daemon=True)
    wt.start()

    # launch encoders
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for (a, b) in ranges:
            ex.submit(encode_worker, sections, a, b, enc, eot_ids, ctx, q)

    wt.join()
    print(f"[done] {in_txt} -> {out_bin}")

def main():
    ap = argparse.ArgumentParser(description="GPT-2 tokenizer (RAM text, bounded tokens) -> fixed-ctx .bin")
    ap.add_argument("inputs", nargs="+", help="Path(s) to input .txt")
    ap.add_argument("--ctx", type=int, default=1024)
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 8))
    ap.add_argument("--segments", type=int, default=0, help="0 => 4x workers")
    ap.add_argument("--queue-size", type=int, default=2048, help="Max token arrays buffered")
    ap.add_argument("--drop-last", action="store_true")
    args = ap.parse_args()

    segs = args.segments if args.segments > 0 else args.workers * 4
    for p in args.inputs:
        tokenize_file(Path(p), ctx=args.ctx, workers=args.workers,
                      segments=segs, queue_size=args.queue_size,
                      drop_last=args.drop_last)

if __name__ == "__main__":
    main()