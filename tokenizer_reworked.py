#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
High-performance data prep + tokenizer for Kaggle mental-health corpora

What this script does
---------------------
1) Downloads & parses multiple Kaggle datasets (full run by default).
2) Cleans & keywordizes text (URLs/emails/hashtags/emoji stripped, spam heuristics,
   optional profanity removal via better_profanity + custom list, optional spaCy).
3) Dedupe, global shuffle, split into train/val, and write .txt files.
4) Tokenizes the text for BOTH GPT-2 (tiktoken, uint16) and Llama-3 (HF AutoTokenizer, uint32):
   - Strict hard cap to ctx=1024 (default) per sequence
   - Parallel, RAM-first encoding (process pool)
   - Packs tokens into fixed-size blocks of length ctx, suitable for llm.c
   - Writes per-model .bin files next to the .txt:
        <train-out>_gpt2.bin,  <val-out>_gpt2.bin
        <train-out>_llama3.bin,<val-out>_llama3.bin

Examples
--------
# Full run with DSM-5 injection; defaults tuned for a fast desktop (i9 + lots of RAM)
python high_perf_kaggle_tokenizers.py ^
  --include-dsm --dsm-file "dataset/DSM-5.txt" ^
  --workers 16 --token-workers 24 --ctx 1024

# Skip RMHD (largest dataset) to iterate faster:
python high_perf_kaggle_tokenizers.py --skip-rmhd --include-dsm

Requirements
------------
pip install kaggle transformers tiktoken pandas numpy better_profanity spacy
# (spaCy is optional; only used if --keywordize spacy)
# If using --keywordize spacy:
python -m spacy download en_core_web_sm
"""

import os, sys, csv, re, json, argparse, hashlib
from pathlib import Path
import random
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

# Keep HF tokenizer threads under control (we parallelize at process level)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ---------------------------
# Optional libs
# ---------------------------
try:
    import tiktoken  # GPT-2 tokenizer
except Exception:
    tiktoken = None

try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None

try:
    import pandas as pd
except Exception:
    pd = None

try:
    from better_profanity import profanity as _bp
except Exception:
    _bp = None

try:
    import spacy
    _spacy_nlp = None
except Exception:
    _spacy_nlp = None


# ---------------------------
# CLI
# ---------------------------
def build_argparser():
    p = argparse.ArgumentParser(
        description="Kaggle merge/clean → train/val .txt + {gpt2,llama3}.bin (RAM-first + parallel; ctx-capped)."
    )
    # outputs
    p.add_argument('--train-out', type=str, default='dataset/data/training_data.txt')
    p.add_argument('--val-out',   type=str, default='dataset/data/validation_data.txt')
    p.add_argument('--val-ratio', type=float, default=0.10, help='Validation fraction (after dedupe).')
    p.add_argument('--seed',      type=int, default=42)

    # parallelism (safe defaults; on i9-14900K you can keep or raise)
    cpu = os.cpu_count() or 8
    p.add_argument('--workers', type=int, default=min(16, cpu),
                   help='Threads for dataset parsing / row processing.')
    p.add_argument('--token-workers', type=int, default=min(28, max(2, cpu - 4)),
                   help='Processes for parallel tokenization.')

    # DSM
    p.add_argument('--include-dsm', action='store_true', help='Inject DSM-5 Q&A from --dsm-file.')
    p.add_argument('--dsm-file',  type=str, default='dataset/DSM-5.txt')

    # cleaning / keywordization
    p.add_argument('--keywordize', choices=['simple', 'spacy'], default='simple',
                   help="Keywordization strategy.")
    p.add_argument('--strip-profanity', choices=['on','off'], default='on',
                   help="Remove profane tokens (custom list + better_profanity if installed).")
    p.add_argument('--profanity-file', type=str, default=None,
                   help="Path to custom profanity list (one word per line).")
    p.add_argument('--strip-emojis', choices=['on','off'], default='on', help="Remove emoji/pictographs.")
    p.add_argument('--unique-keywords', choices=['on','off'], default='on', help="Unique tokens per post.")
    p.add_argument('--min-chars', type=int, default=20, help='Min raw chars before cleaning.')
    p.add_argument('--min-kw', type=int, default=3, help='Min keyword count after keywordization.')

    # dedupe / volume
    p.add_argument('--dedupe-cap', type=int, default=2_000_000, help='Max distinct items tracked for dedupe.')
    p.add_argument('--rmhd-seen-cap', type=int, default=600_000, help='(Legacy) RMHD light dedupe (global dedupe still applies).')
    p.add_argument('--skip-rmhd', action='store_true', help='(Optional) Skip RMHD to move faster.')
    p.add_argument('--max-per-dataset', type=int, default=0, help='Optional cap per dataset (0 = no cap).')

    # Context length (default 1024 for GPT-2) and packing
    p.add_argument('--ctx', type=int, default=1024, help='Hard cap per sequence and pack block size.')
    p.add_argument('--drop-last', action='store_true',
                   help='Drop final partial block (< ctx tokens) when writing .bin.')

    # Llama tokenizer id (needs HF auth/weights access)
    p.add_argument('--llama-tokenizer', type=str, default='meta-llama/Meta-Llama-3.1-8B',
                   help='HF tokenizer id for Llama-3. If unavailable, falls back to Meta-Llama-3-8B.')
    return p


# ---------------------------
# Globals configured at runtime
# ---------------------------
_STOP = {
    'a','an','the','and','or','but','if','while','of','to','in','on','for','from','by',
    'with','about','as','at','into','through','during','before','after','above','below',
    'up','down','out','off','over','under','again','further','then','once','here','there',
    'when','where','why','how','all','any','both','each','few','more','most','other','some',
    'such','no','nor','not','only','own','same','so','than','too','very','can','will','just',
    'don','should','now','i','me','my','we','our','you','your','he','she','it','they','them',
    'is','am','are','was','were','be','been','being','do','does','did','doing','have','has',
    'had','having','this','that','these','those'
}
_URL_RE      = re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE)
_EMAIL_RE    = re.compile(r'\b[\w\.-]+@[\w\.-]+\.\w+\b')
_MENTION_RE  = re.compile(r'@\w+')
_HASHTAG_RE  = re.compile(r'#\w+')
_NONWORD_RE  = re.compile(r'[^a-z0-9\s]+')
_MULTI_WS_RE = re.compile(r'\s+')
_REPEAT_RE   = re.compile(r'(.)\1{5,}')
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

_CUSTOM_PROFANITY = set()
_spacy_nlp = None
args = None
cache_dir = Path("data_cache")


# ---------------------------
# Utils (download, read, clean)
# ---------------------------
def kaggle_download(slug: str, dest: Path):
    if dest.exists() and any(dest.iterdir()):
        print(f"[+] {slug} already in {dest}")
        return
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[-] Downloading {slug} ...")
    code = os.system(f'kaggle datasets download -d "{slug}" -p "{dest}" --quiet --unzip')
    if code != 0:
        raise RuntimeError(f"Failed to download {slug}. Check Kaggle CLI + credentials.")

def read_csv_any(path: Path):
    try:
        with path.open('r', encoding='utf-8', errors='ignore') as f:
            sample = f.read(4096); f.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample)
            except Exception:
                dialect = csv.excel_tab if path.suffix.lower()=='.tsv' else csv.excel
            return list(csv.DictReader(f, dialect=dialect))
    except Exception:
        with path.open('r', encoding='utf-8', errors='ignore') as f:
            return list(csv.DictReader(f))

def init_filters():
    global _spacy_nlp
    if args.profanity_file:
        p = Path(args.profanity_file)
        if p.is_file():
            with p.open('r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    w = line.strip().lower()
                    if w:
                        _CUSTOM_PROFANITY.add(w)
            print(f"[info] custom profanity loaded: {len(_CUSTOM_PROFANITY)} tokens")

    if args.strip_profanity == 'on' and _bp is not None:
        try:
            _bp.load_censor_words()
        except Exception:
            pass

    if args.keywordize == 'spacy':
        try:
            _spacy_nlp = spacy.load('en_core_web_sm', disable=['ner','textcat'])
        except Exception:
            print("[warn] spaCy model not available; falling back to 'simple'")
            args.keywordize = 'simple'

def is_spammy_raw(x: str) -> bool:
    if not x or len(x) < args.min_chars:
        return True
    if _REPEAT_RE.search(x):
        return True
    low = x.lower()
    for kw in ("buy now","promo code","discount","follow me","subscribe","click link","http://bit.ly","free followers"):
        if kw in low:
            return True
    return False

def strip_profanity_tokens(tokens):
    if args.strip_profanity != 'on':
        return tokens
    out = []
    for t in tokens:
        tl = t.lower()
        if tl in _CUSTOM_PROFANITY:
            continue
        if _bp is not None and _bp.contains_profanity(t):
            continue
        out.append(t)
    return out

def keywordize_text(text: str) -> str:
    if not text:
        return ""
    x = str(text)
    x = _URL_RE.sub(' ', x)
    x = _EMAIL_RE.sub(' ', x)
    x = _MENTION_RE.sub(' ', x)
    x = _HASHTAG_RE.sub(' ', x)
    if args.strip_emojis == 'on':
        x = _EMOJI_RE.sub(' ', x)
    x = x.lower()
    x = _NONWORD_RE.sub(' ', x)
    x = _MULTI_WS_RE.sub(' ', x).strip()
    if not x:
        return ""

    if args.keywordize == 'spacy' and _spacy_nlp is not None:
        doc = _spacy_nlp(x)
        toks = []
        for tok in doc:
            if tok.is_space or tok.is_punct or tok.is_stop: continue
            if tok.pos_ not in ('NOUN','PROPN','ADJ','VERB'): continue
            lemma = (tok.lemma_ or tok.text).lower()
            if lemma in _STOP or len(lemma) <= 1: continue
            toks.append(lemma)
    else:
        toks = [t for t in x.split() if t not in _STOP and len(t) > 1]

    toks = strip_profanity_tokens(toks)
    if args.unique-keywords if False else args.unique_keywords:  # guard against editor replacements
        toks = list(dict.fromkeys(toks))  # per-post uniqueness
    if len(toks) < args.min_kw:
        return ""
    return " ".join(toks)

def prepare_post(raw_text: str, title: str = "") -> str:
    if not raw_text:
        return ""
    full = raw_text
    if title and title.strip() and title not in raw_text:
        full = f"{title.strip()}\n{raw_text}"
    if is_spammy_raw(full):
        return ""
    full = " ".join(full.split())
    kw = keywordize_text(full)
    return kw

def sample_line(post: str, task: str, answer: str) -> str:
    return f"Post: {post}\nTask: {task}\nAnswer: {answer}<|endoftext|>"

def _parallel_rows(rows, fn, limit=0, workers=8):
    """Apply fn(row)->list[str] in parallel and flatten; respect optional cap."""
    out = []
    if not rows:
        return out
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(fn, r) for r in rows]
        for fut in as_completed(futs):
            part = fut.result() or []
            out.extend(part)
            if limit and len(out) >= limit:
                return out[:limit]
    return out if not limit else out[:limit]


# ---------------------------
# Dataset builders (return list[str]) — each row processed in parallel
# ---------------------------
def ds_kamaruladha(maxn=0):
    slug = "kamaruladha/mental-disorders-identification-reddit-nlp"
    d = cache_dir / "mental_disorders_identification"
    kaggle_download(slug, d)
    csvs = sorted(d.glob("*.csv"))
    if not csvs: return []
    rows = read_csv_any(csvs[0])

    def proc(row):
        text = (row.get('text') or row.get('body') or row.get('post') or "")
        title = (row.get('title') or "")
        post = prepare_post(text, title)
        if not post: return []
        label = (row.get('subreddit') or row.get('label') or "").strip()
        if not label: return []
        low = label.lower()
        mapping = {"adhd":"ADHD","ptsd":"PTSD","ocd":"OCD"}
        label2 = mapping.get(low, label.capitalize() if low==label else label)
        return [sample_line(post, "Identify which mental health condition this post is about.", label2)]

    return _parallel_rows(rows, proc, maxn, args.workers)

def ds_bipolar(maxn=0):
    slug = "michellevp/mental-health-dataset-bipolar"
    d = cache_dir / "mental_health_bipolar"
    kaggle_download(slug, d)
    f = next((p for p in d.glob("*") if p.suffix.lower() in (".csv",".xlsx")), None)
    if not f: return []
    rows = read_csv_any(f) if f.suffix.lower()==".csv" else (pd.read_excel(f).to_dict(orient="records") if pd else [])

    def proc(row):
        text = (row.get('text') or row.get('post') or row.get('content') or "")
        post = prepare_post(text)
        if not post: return []
        sent = (row.get('sentiment') or row.get('sentiment_label') or "")
        if not str(sent).strip(): return []
        s = str(sent).lower()
        if s == '1' or 'pos' in s: sent2 = "Positive"
        elif s == '-1' or 'neg' in s: sent2 = "Negative"
        elif s == '0' or 'neu' in s: sent2 = "Neutral"
        else: sent2 = str(sent).capitalize()
        out = [sample_line(post, "Determine the sentiment expressed in the following bipolar discussion post.", sent2)]
        risk = (row.get('risk_factor') or "").strip()
        if risk:
            out.append(sample_line(post, "Identify any risk factor mentioned in the post.", risk))
        return out

    return _parallel_rows(rows, proc, maxn, args.workers)

def ds_anxiety_binary(maxn=0):
    slug = "michellevp/predicting-anxiety-in-mental-health-data"
    d = cache_dir / "mental_health_anxiety"
    kaggle_download(slug, d)
    f = next((p for p in d.glob("*") if p.suffix.lower() in (".csv",".xlsx")), None)
    if not f: return []
    rows = read_csv_any(f) if f.suffix.lower()==".csv" else (pd.read_excel(f).to_dict(orient="records") if pd else [])

    def proc(r):
        text = (r.get('text') or r.get('post') or r.get('content') or "")
        post = prepare_post(text)
        if not post: return []
        val = (r.get('anxiety') or r.get('label') or r.get('class') or "")
        val = str(val).strip().lower()
        if not val: return []
        ans = "Yes" if val in ("1","true","yes","anxiety") else "No"
        return [sample_line(post, "Does the following post indicate an anxiety disorder? Answer Yes or No.", ans)]

    return _parallel_rows(rows, proc, maxn, args.workers)

def ds_dreaddit(maxn=0):
    slug = "shuvojitdas/stress-analysis"
    d = cache_dir / "stress_analysis"
    kaggle_download(slug, d)
    files = list(d.glob("*.csv")) or list(d.glob("*.tsv"))
    if not files: return []
    rows = []
    for f in files:
        rows.extend(read_csv_any(f))

    def proc(row):
        text = (row.get('text') or row.get('post') or row.get('body') or "")
        post = prepare_post(text)
        if not post: return []
        lab = (row.get('label') or row.get('stress') or row.get('y') or "")
        ans = "Yes" if str(lab).strip().lower() in ("1","yes","true","stressed") else "No"
        return [sample_line(post, "Determine if the user is stressed in the following post. Answer Yes or No.", ans)]

    return _parallel_rows(rows, proc, maxn, args.workers)

def ds_neel(maxn=0):
    slug = "neelghoshal/reddit-mental-health-data"
    d = cache_dir / "reddit_mental_health_data"
    kaggle_download(slug, d)
    f = next((p for p in d.glob("*") if p.suffix.lower() in (".csv",".tsv")), None)
    if not f: return []
    rows = read_csv_any(f)
    mapping = {'0':"Stress",'1':"Depression",'2':"Bipolar",'3':"Anxiety",'4':"PTSD"}

    def proc(row):
        text = (row.get('text') or row.get('post') or row.get('body') or "")
        post = prepare_post(text)
        if not post: return []
        lab = (row.get('target') or row.get('label') or row.get('class') or "")
        lab = str(lab).strip()
        if not lab: return []
        name = mapping.get(lab, lab.capitalize())
        return [sample_line(post,
                            "Identify the mental health issue discussed in this post (Stress, Depression, Bipolar, Anxiety, or PTSD).",
                            name)]
    return _parallel_rows(rows, proc, maxn, args.workers)

def ds_social_anxiety(maxn=0):
    slug = "natezhang123/social-anxiety-dataset"
    d = cache_dir / "social_anxiety_dataset"
    kaggle_download(slug, d)
    preferred = d / "enhanced_anxiety_dataset.csv"
    f = preferred if preferred.exists() else next((p for p in d.glob("*") if p.suffix.lower() in (".csv",".xlsx")), None)
    if not f: return []
    rows = read_csv_any(f) if f.suffix.lower()==".csv" else (pd.read_excel(f).to_dict(orient="records") if pd else [])
    if not rows: return []
    keys0 = list(rows[0].keys())
    def _norm(s): return s.lower().strip().replace(" ", "_")
    pri = ("level","severity","score","class","label","category","status")
    label_col = None
    for k in keys0:
        nk = _norm(k)
        if "anxiety" in nk and any(t in nk for t in pri):
            label_col = k; break
    if not label_col:
        for k in keys0:
            if any(t in _norm(k) for t in pri):
                label_col = k; break
    if not label_col:
        return []

    raw_vals = [str(r.get(label_col,"")).strip() for r in rows if str(r.get(label_col,"")).strip()!=""]
    def _to_float(x):
        try: return float(x)
        except: return None
    numeric_vals = [v for v in (_to_float(x) for x in raw_vals) if v is not None]
    if numeric_vals:
        arr = np.array(numeric_vals, dtype=float)
        if np.allclose(arr.min(), arr.max()):
            q1=q2=q3=arr.min(); mode="degenerate"
        else:
            q1,q2,q3 = np.quantile(arr,[0.25,0.5,0.75]); mode="quartiles"
        def num2sev(x):
            if mode=="degenerate": return "Moderate"
            if x<=q1: return "None"
            elif x<=q2: return "Mild"
            elif x<=q3: return "Moderate"
            else: return "Severe"
    else:
        def cat2sev(s):
            sl=s.lower()
            if sl in ("0","none","no","low","lowest","minimal"): return "None"
            if sl in ("1","mild","slight","light"): return "Mild"
            if sl in ("2","moderate","med","medium"): return "Moderate"
            if sl in ("3","severe","high","very high"): return "Severe"
            if "none" in sl: return "None"
            if "mild" in sl or "low" in sl: return "Mild"
            if "moderate" in sl or "mid" in sl: return "Moderate"
            if "severe" in sl or "high" in sl: return "Severe"
            return "Moderate"

    feat_pool = [k for k in keys0 if k != label_col]
    def score_feat(name):
        n=_norm(name); score=0
        for t in ("anxiety","social","avoid","fear","score","scale","sleep","caffeine","alcohol","exercise"):
            if t in n: score+=1
        return score
    feat_pool.sort(key=score_feat, reverse=True)
    chosen = feat_pool[:3]

    def proc(r):
        raw = r.get(label_col,"")
        if raw is None or str(raw).strip()=="": return []
        sev = num2sev(_to_float(raw)) if numeric_vals and _to_float(raw) is not None else (cat2sev(str(raw)) if not numeric_vals else None)
        if sev is None: return []
        pieces=[]
        for fk in chosen:
            v = str(r.get(fk,"")).strip()
            if v and v.lower()!="nan": pieces.append(f"{fk}: {v}")
        desc = "; ".join(pieces) if pieces else "various lifestyle and screening features available"
        post = prepare_post(f"The individual reports: {desc}.")
        if not post: return []
        return [sample_line(post, "Classify the person's social anxiety level (None, Mild, Moderate, or Severe).", sev)]

    return _parallel_rows(rows, proc, maxn, args.workers)

# ---- RMHD: parallel per file (threads) ----
def _rmhd_process_file(path: Path) -> list:
    def canon_disorder(name: str) -> str:
        if not name: return ""
        low = name.strip().lower()
        if low.startswith("r/"): low = low[2:]
        low = low.replace("_"," ")
        for k,v in {"anxiety":"Anxiety","depression":"Depression","bipolar":"Bipolar","ptsd":"PTSD","adhd":"ADHD","stress":"Stress","ocd":"OCD"}.items():
            if k in low: return v
        return " ".join(w.capitalize() for w in low.split())

    out = []
    try:
        ext = path.suffix.lower()
        rows = []
        if ext in (".csv",".tsv"):
            with path.open('r', encoding='utf-8', errors='ignore') as f:
                sample = f.read(8192); f.seek(0)
                try:
                    dialect = csv.Sniffer().sniff(sample)
                except Exception:
                    dialect = csv.excel_tab if ext==".tsv" else csv.excel
                rows = list(csv.DictReader(f, dialect=dialect))
        elif ext == ".xlsx":
            if pd is None: return []
            rows = pd.read_excel(path).to_dict(orient="records")
        elif ext == ".json":
            with path.open('r', encoding='utf-8', errors='ignore') as f:
                obj = json.load(f)
            rows = obj.get("data", []) if isinstance(obj, dict) else (obj if isinstance(obj, list) else [])
        elif ext == ".jsonl":
            with path.open('r', encoding='utf-8', errors='ignore') as f:
                rows = []
                for line in f:
                    line=line.strip()
                    if not line: continue
                    try:
                        row = json.loads(line)
                        if isinstance(row, dict): rows.append(row)
                    except Exception:
                        continue

        for row in rows:
            title = (row.get("title") or "").strip()
            tfields = ("text","post","content","body","selftext","comment","message","description")
            raw = ""
            for k in tfields:
                v = row.get(k)
                if v and str(v).strip():
                    raw = str(v); break
            if not raw and title:
                raw = title
            if not raw: continue
            post = prepare_post(raw, title)
            if not post: continue
            disorder = None
            for k in ("subreddit","condition","category","label","mental_illness","diagnosis","target","class"):
                v = row.get(k)
                if v and str(v).strip():
                    disorder = canon_disorder(str(v)); break
            if disorder:
                out.append(sample_line(post, "Identify which mental health condition this post is about.", disorder))
    except Exception as e:
        print(f"[warn] RMHD parse failed for {path.name}: {e}")
    return out

def ds_rmhd_parallel(maxn=0):
    slug = "entenam/reddit-mental-health-dataset"
    d = cache_dir / "reddit_mental_health_dataset"
    kaggle_download(slug, d)
    cands = [p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in (".csv",".tsv",".xlsx",".json",".jsonl")]
    if not cands:
        return []
    print(f"[+] RMHD: {len(cands)} files (parallel parse with {args.workers} threads)")
    out = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_rmhd_process_file, p) for p in cands]
        for fut in as_completed(futures):
            chunk = fut.result() or []
            out.extend(chunk)
            if maxn and len(out) >= maxn:
                return out[:maxn]
    return out if not maxn else out[:maxn]


# ---------------------------
# DSM-5 injection
# ---------------------------
def inject_dsm_lines():
    if not args.include_dsm:
        print("[*] DSM-5 injection disabled.")
        return []
    p = Path(args.dsm_file)
    if not p.is_file():
        print(f"[!] DSM-5 file not found at {p}; skipping DSM injection.")
        return []
    print(f"[+] Injecting DSM-5 knowledge from {p}")
    lines = p.read_text(encoding='utf-8', errors='ignore').splitlines()
    disorder = None; bucket=[]; out=[]
    def flush():
        if disorder and bucket:
            crit = " ".join([x.strip() for x in bucket if x.strip()])
            if crit:
                post_kw = keywordize_text("(DSM-5 Reference)")
                out.append(sample_line(post_kw, f"List the DSM-5 diagnostic criteria for {disorder}.", crit))
    for line in lines:
        if line.isupper() and line.strip() and len(line.split()) < 10:
            flush()
            disorder = line.strip().title()
            bucket = []
        else:
            if disorder:
                bucket.append(line)
    flush()
    return out


# ---------------------------
# Tokenization (parallel, hard cap to ctx, packed to fixed ctx blocks)
#   TWO outputs by default:
#     - GPT-2 (tiktoken)  -> *_gpt2.bin  (uint16)
#     - Llama-3 (HF tok)  -> *_llama3.bin (uint32; llama vocab > 65535)
# ---------------------------
def _split_even(lst, parts):
    n = len(lst)
    if parts <= 1 or n == 0:
        return [lst]
    step = (n + parts - 1) // parts
    return [lst[i:i+step] for i in range(0, n, step)]

# ---- GPT-2 / tiktoken (process worker) ----
def _tok_mp_gpt2_init():
    global _ENC_GPT2
    if tiktoken is None:
        raise RuntimeError("tiktoken not installed. pip install tiktoken")
    _ENC_GPT2 = tiktoken.get_encoding("gpt2")

def _tok_mp_gpt2_encode(lines_ctx):
    lines, ctx_len = lines_ctx
    enc = _ENC_GPT2
    out = []
    for s in lines:
        ids = enc.encode_ordinary(s)  # no special tokens inserted
        if len(ids) > ctx_len:
            ids = ids[:ctx_len]
        out.append(ids)
    return out

# ---- Llama-3 / HF AutoTokenizer (process worker) ----
def _tok_mp_llama3_init(tokenizer_id: str):
    global _ENC_LLAMA
    if AutoTokenizer is None:
        raise RuntimeError("transformers not installed. pip install transformers")
    try:
        _ENC_LLAMA = AutoTokenizer.from_pretrained(tokenizer_id)
    except Exception:
        # fallback that commonly exists
        _ENC_LLAMA = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

def _tok_mp_llama3_encode(lines_ctx):
    lines, ctx_len = lines_ctx
    tok = _ENC_LLAMA
    enc = tok(lines, add_special_tokens=False, truncation=True, max_length=ctx_len)
    out = []
    for ids in enc["input_ids"]:
        if len(ids) > ctx_len:
            ids = ids[:ctx_len]
        out.append(ids)
    return out

def _pack_and_write(all_token_seqs, out_bin_path: Path, ctx_len: int, drop_last: bool, dtype):
    # Pack token sequences into fixed ctx_len blocks
    buf = np.empty(ctx_len, dtype=dtype)
    buf_len = 0
    total_chunks = 0
    max_seen = 0

    with open(out_bin_path, 'wb') as fout:
        for seq in all_token_seqs:
            if len(seq) > ctx_len:
                seq = seq[:ctx_len]  # defensive (shouldn’t trigger)
            max_seen = max(max_seen, len(seq))
            i = 0
            while i < len(seq):
                take = min(ctx_len - buf_len, len(seq) - i)
                if take:
                    buf[buf_len:buf_len+take] = np.asarray(seq[i:i+take], dtype=dtype)
                    buf_len += take
                    i += take
                if buf_len == ctx_len:
                    buf.tofile(fout)
                    total_chunks += 1
                    buf_len = 0
        if buf_len and not drop_last:
            buf[:buf_len].tofile(fout)
            total_chunks += 1

    return total_chunks, max_seen

def tokenize_lines_to_bins(lines, out_prefix: Path, ctx_len: int, workers: int, drop_last: bool, llama_tok_id: str):
    """
    Produce two bin files:
      - <out_prefix>_gpt2.bin    (tiktoken, uint16)
      - <out_prefix>_llama3.bin  (HF AutoTokenizer, uint32)
    Parallelized with a process pool; sequences strictly capped to ctx_len.
    """
    n = len(lines)
    if n == 0:
        for suffix in ("_gpt2.bin", "_llama3.bin"):
            open(out_prefix.with_suffix(suffix), 'wb').close()
        print(f"[tokenize] {out_prefix}* | nothing to write")
        return

    splits = _split_even(lines, max(1, workers))

    # --- GPT-2 / tiktoken ---
    out_gpt2 = out_prefix.with_suffix("_gpt2.bin")
    if tiktoken is None:
        print("[tokenize:gpt2] SKIP (tiktoken not installed)")
    else:
        print(f"[tokenize:gpt2] {out_gpt2} | ctx={ctx_len} | workers={len(splits)}")
        all_gpt2 = []
        with ProcessPoolExecutor(max_workers=len(splits), initializer=_tok_mp_gpt2_init) as ex:
            futs = [ex.submit(_tok_mp_gpt2_encode, (chunk, ctx_len)) for chunk in splits]
            for fut in as_completed(futs):
                all_gpt2.extend(fut.result())
        chunks, max_seen = _pack_and_write(all_gpt2, out_gpt2, ctx_len, drop_last, dtype=np.uint16)
        print(f"[tokenize:gpt2] wrote {chunks} chunk(s); max per-seq length observed = {max_seen} (≤ {ctx_len})")

    # --- Llama-3 / HF ---
    out_llama = out_prefix.with_suffix("_llama3.bin")
    if AutoTokenizer is None:
        print("[tokenize:llama3] SKIP (transformers not installed)")
    else:
        print(f"[tokenize:llama3] {out_llama} | ctx={ctx_len} | workers={len(splits)}")
        all_llama = []
        # pass tokenizer id via lambda to initializer (workaround for ProcessPoolExecutor)
        def _init_llama():
            _tok_mp_llama3_init(llama_tok_id)
        with ProcessPoolExecutor(max_workers=len(splits), initializer=_init_llama) as ex:
            futs = [ex.submit(_tok_mp_llama3_encode, (chunk, ctx_len)) for chunk in splits]
            for fut in as_completed(futs):
                all_llama.extend(fut.result())
        chunks, max_seen = _pack_and_write(all_llama, out_llama, ctx_len, drop_last, dtype=np.uint32)
        print(f"[tokenize:llama3] wrote {chunks} chunk(s); max per-seq length observed = {max_seen} (≤ {ctx_len})")


# ---------------------------
# MAIN
# ---------------------------
def main():
    global args, cache_dir
    parser = build_argparser()
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    cache_dir.mkdir(exist_ok=True, parents=True)

    train_path = Path(args.train_out); train_path.parent.mkdir(parents=True, exist_ok=True)
    val_path   = Path(args.val_out);   val_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[ok] train={train_path} | val={val_path}")
    init_filters()

    # --- Build all examples in RAM (full run by default) ---
    examples = []
    per_cap = args.max_per_dataset or 0

    def take(ds_func, name):
        print(f"[build] {name} ...")
        items = ds_func(per_cap)
        print(f"  -> {name}: {len(items):,} examples")
        examples.extend(items)

    take(ds_kamaruladha, "KamarulAdha (reddit-nlp)")
    take(ds_bipolar,     "Bipolar (MichelleVP)")
    take(ds_anxiety_binary, "Anxiety Binary (MichelleVP)")
    take(ds_dreaddit,    "Dreaddit (Stress)")
    take(ds_neel,        "Reddit Mental Health (Neel)")
    take(ds_social_anxiety, "Social Anxiety (binned)")
    if not args.skip_rmhd:
        take(ds_rmhd_parallel, "RMHD (Entenam, parallel)")
    else:
        print("[*] Skipping RMHD as requested.")

    # Optional additional sets (CID007, Panic, ADHD)
    def ds_cid007(maxn=0):
        slug = "cid007/mental-disorder-classification"
        d = cache_dir / "mental_disorder_classification"
        kaggle_download(slug, d)
        pth = None
        for ext in ("*.csv","*.tsv","*.xlsx"):
            L = list(d.rglob(ext))
            if L: pth = L[0]; break
        if not pth: return []
        rows = read_csv_any(pth) if pth.suffix.lower() in (".csv",".tsv") else (pd.read_excel(pth).to_dict(orient="records") if pd else [])
        if not rows: return []
        keys0 = list(rows[0].keys())
        def norm(s): return s.lower().strip().replace(" ","_")
        label_col = None
        pref = ("diagnos","disorder","label","class","target","illness","condition","result")
        for k in keys0:
            if any(w in norm(k) for w in pref):
                label_col = k; break
        if not label_col and pd is not None:
            dfh = pd.DataFrame(rows)
            for k in keys0:
                try:
                    u = dfh[k].nunique(dropna=True)
                    if 2 <= u <= 8 and not pd.api.types.is_numeric_dtype(dfh[k]): label_col = k; break
                except Exception:
                    pass
        if not label_col: return []
        symptom_cols = [k for k in keys0 if k != label_col]
        def clean_diag(x):
            if x is None: return ""
            low = str(x).strip().lower()
            mp = {"normal":"Normal","anxiety":"Anxiety","depression":"Depression","bipolar":"Bipolar","ptsd":"PTSD","ocd":"OCD","adhd":"ADHD"}
            return mp.get(low, str(x).strip().title())
        def include_val(v):
            if v is None: return False
            s = str(v).strip().lower()
            if s in ("","nan","none"): return False
            try:
                return float(s) > 0.0
            except Exception:
                return s in ("yes","true","present","high","mild","moderate","severe","1")
        def proc(r):
            diag = clean_diag(r.get(label_col,""))
            if not diag: return []
            pres = []
            for c in symptom_cols:
                if include_val(r.get(c,"")):
                    pres.append(str(c).replace("_"," ").strip().title())
            if not pres and diag != "Normal": 
                return []
            if not pres and diag == "Normal":
                pres = ["No significant psychiatric symptoms"]
            post = prepare_post(f"Patient exhibits the following symptoms: {', '.join(pres[:6])}.")
            if not post: return []
            return [sample_line(post, "Determine the most likely diagnosis for this patient.", diag)]
        return _parallel_rows(rows, proc, maxn, args.workers)

    def ds_panic(maxn=0):
        slug = "muhammadshahidazeem/panic-disorder-detection-dataset"
        d = cache_dir / "panic_disorder_detection"
        kaggle_download(slug, d)
        files = [p for p in d.glob("*") if p.suffix.lower() in (".csv",".tsv")]
        if not files: return []
        rows = []
        for f in files:
            rows.extend(read_csv_any(f))
        def proc(r):
            k0 = list(r.keys())
            label_key = next((k for k in k0 if ("panic" in k.lower()) and any(x in k.lower() for x in ("label","disorder","target"))), None) or k0[-1]
            lv = str(r.get(label_key,"")).strip().lower()
            ans = "Yes" if lv in ("1","yes","true","panic") else "No"
            parts=[]
            for fk in k0:
                if fk == label_key: continue
                v = str(r.get(fk,"")).strip()
                if not v or v.lower()=="nan": continue
                low = fk.lower()
                if low in ("age","gender","sex"):
                    parts.append(f"{fk}: {v}")
                elif low in ("symptom","symptoms","chest_pain","sweating","palpitations","dizziness"):
                    if v in ("1","yes","true"): parts.append(f"{fk.replace('_',' ')}: yes")
                if len(parts)>=5: break
            if not parts: parts.append("no notable symptoms")
            post = prepare_post(f"Patient data - {'; '.join(parts)}.")
            if not post: return []
            return [sample_line(post, "Is this a case of Panic Disorder? Answer Yes or No.", ans)]
        return _parallel_rows(rows, proc, maxn, args.workers)

    def ds_adhd(maxn=0):
        slug = "jerseyneo/reddit-adhd-dataset"
        d = cache_dir / "reddit_adhd_dataset"
        kaggle_download(slug, d)
        csvs = sorted(d.glob("*.csv"))
        if not csvs: return []
        rows = []
        for pth in csvs:
            with pth.open('r', encoding='utf-8', errors='ignore') as f:
                sample = f.read(8192); f.seek(0)
                try:
                    dialect = csv.Sniffer().sniff(sample)
                except Exception:
                    dialect = csv.excel
                rows.extend(list(csv.DictReader(f, dialect=dialect)))
        candidate_fields = ("selftext","body","comment","text","content","message","post","description")
        def proc(row):
            raw = None
            for k in candidate_fields:
                v = row.get(k)
                if v and str(v).strip():
                    raw = str(v); break
            if not raw:
                joined = " ".join(str(row.get(k,"")).strip()
                                  for k in ("title","selftext","body")
                                  if str(row.get(k,"")).strip()).strip()
                raw = joined if joined else None
            if not raw: return []
            title = (row.get("title") or "")
            post = prepare_post(raw, title)
            if not post: return []
            return [sample_line(post, "Identify which mental health condition this post is about.", "ADHD")]
        return _parallel_rows(rows, proc, maxn, args.workers)

    take(ds_cid007,      "CID007 (symptom→diagnosis)")
    take(ds_panic,       "Panic disorder detection")
    take(ds_adhd,        "Reddit ADHD (jerseyneo)")

    # DSM-5
    dsm = inject_dsm_lines()
    print(f"  -> DSM-5 injected: {len(dsm):,} examples")
    examples.extend(dsm)

    if not examples:
        print("[fatal] no examples produced.")
        sys.exit(1)

    # --- Global dedupe (sha1 on full sample line) ---
    print(f"[dedupe] input={len(examples):,}")
    seen = set(); deduped = []
    cap = args.dedupe_cap
    for s in examples:
        h = hashlib.sha1(s.encode('utf-8')).hexdigest()
        if len(seen) < cap:
            if h in seen: continue
            seen.add(h)
        deduped.append(s)
    examples = deduped
    print(f"[dedupe] kept={len(examples):,}")

    # --- Shuffle + split ---
    rng = random.Random(args.seed)
    rng.shuffle(examples)
    split = int(len(examples) * (1.0 - args.val_ratio))
    train_lines = examples[:split]
    val_lines   = examples[split:]
    print(f"[split] train={len(train_lines):,}  val={len(val_lines):,}  (val-ratio={args.val_ratio})")

    # --- Write text once ---
    train_path.write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    val_path.write_text("\n".join(val_lines) + "\n", encoding="utf-8")
    print(f"[write] {train_path} , {val_path}")

    # --- Tokenize in parallel, **strictly capped to ctx**, packed into fixed ctx chunks ---
    tokenize_lines_to_bins(train_lines, train_path, args.ctx, args.token_workers, args.drop_last, args.llama_tokenizer)
    tokenize_lines_to_bins(val_lines,   val_path,   args.ctx, max(2, args.token_workers//2), args.drop_last, args.llama_tokenizer)

    print("[done] all steps complete.")


if __name__ == "__main__":
    main()
```0