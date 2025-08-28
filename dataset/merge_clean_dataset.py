#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge mental-health datasets, clean & keywordize text, split into train/val,
and write streaming GPT-2 token binaries for llm.c.

Datasets included:
  - kamaruladha/mental-disorders-identification-reddit-nlp
  - michellevp/mental-health-dataset-bipolar
  - michellevp/predicting-anxiety-in-mental-health-data
  - shuvojitdas/stress-analysis
  - neelghoshal/reddit-mental-health-data
  - natezhang123/social-anxiety-dataset
  - entenam/reddit-mental-health-dataset  (recursive loader)
  - cid007/mental-disorder-classification
  - muhammadshahidazeem/panic-disorder-detection-dataset
  - jerseyneo/reddit-adhd-dataset

Outputs:
  --train-out  (text) + .bin
  --val-out    (text) + .bin
"""

import os
import sys
import csv
import re
import math
import json
import argparse
import hashlib
from pathlib import Path
import random
import numpy as np

# Optional libs
try:
    from transformers import GPT2TokenizerFast
except ImportError:
    GPT2TokenizerFast = None

try:
    from better_profanity import profanity as _bp
except ImportError:
    _bp = None

try:
    import pandas as pd
except Exception:
    pd = None

try:
    import spacy
    _spacy_nlp = None
except Exception:
    _spacy_nlp = None


# ---------------------------
# CLI
# ---------------------------
parser = argparse.ArgumentParser(description="Merge & clean mental-health datasets → train/val .txt + .bin")
parser.add_argument('--train-out', type=str, default='dataset/data/training_data.txt')
parser.add_argument('--val-out',   type=str, default='dataset/data/validation_data.txt')
parser.add_argument('--val-ratio', type=float, default=0.10, help='Fraction routed to validation set (approx).')
parser.add_argument('--seed',      type=int, default=42)
parser.add_argument('--include-dsm', action='store_true', help='Inject DSM-5 knowledge Q&A from --dsm-file.')
parser.add_argument('--dsm-file',  type=str, default='dataset/DSM-5.txt')

# cleaning / keywordization
parser.add_argument('--keywordize', choices=['simple', 'spacy'], default='simple',
                    help="Reduce posts to keywords: 'simple' (fast) or 'spacy' (lemma+POS).")
parser.add_argument('--strip-profanity', choices=['on','off'], default='on',
                    help="Remove profane tokens (custom list + better_profanity if installed).")
parser.add_argument('--profanity-file', type=str, default=None, help="Path to custom profanity/stop list (one word per line).")
parser.add_argument('--strip-emojis', choices=['on','off'], default='on', help="Remove emoji/pictographs.")
parser.add_argument('--unique-keywords', choices=['on','off'], default='on', help="Make tokens set-unique per post.")
parser.add_argument('--min-chars', type=int, default=20, help='Min raw characters before cleaning.')
parser.add_argument('--min-kw', type=int, default=3, help='Min keyword count after keywordization.')

# dedupe & spam heuristics
parser.add_argument('--dedupe-cap', type=int, default=1_000_000, help='Max items to track for dedupe (memory cap).')
parser.add_argument('--rmhd-seen-cap', type=int, default=500_000, help='Light dedupe cap for RMHD.')
parser.add_argument('--skip-rmhd', action='store_true', help='Skip Entenam RMHD if you want to limit volume.')

args = parser.parse_args()
random.seed(args.seed)

# ---------------------------
# Paths & setup
# ---------------------------
cache_dir = Path("data_cache"); cache_dir.mkdir(exist_ok=True, parents=True)
train_path = Path(args.train_out); train_path.parent.mkdir(parents=True, exist_ok=True)
val_path   = Path(args.val_out);   val_path.parent.mkdir(parents=True, exist_ok=True)

print(f"[ok] train={train_path} | val={val_path}")

# ---------------------------
# Kaggle download helper
# ---------------------------
def kaggle_download(slug: str, dest: Path):
    if dest.exists() and any(dest.iterdir()):
        print(f"[+] {slug} already downloaded in {dest}")
        return
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[-] Downloading {slug} ...")
    # quiet unzip + show meta in console (kaggle CLI prints URL/license)
    code = os.system(f'kaggle datasets download -d "{slug}" -p "{dest}" --quiet --unzip')
    if code != 0:
        raise RuntimeError(f"Failed to download {slug}. Ensure Kaggle CLI is installed and credentials are set.")

# ---------------------------
# CSV/TSV reader (sniff)
# ---------------------------
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

# ---------------------------
# Cleaning / keywordization
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
_REPEAT_RE   = re.compile(r'(.)\1{5,}')  # e.g., 'aaaaaa'
_EMOJI_RE    = re.compile(
    "["                      # common emoji/ pictographs
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002700-\U000027BF"
    "\U000024C2-\U0001F251"
    "]+", flags=re.UNICODE
)

_CUSTOM_PROFANITY = set()

def _init_filters():
    # custom profanity list
    if args.profanity_file:
        p = Path(args.profanity_file)
        if p.is_file():
            with p.open('r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    w = line.strip().lower()
                    if w:
                        _CUSTOM_PROFANITY.add(w)
            print(f"[info] loaded {len(_CUSTOM_PROFANITY)} custom profanity tokens")

    # better_profanity
    if args.strip_profanity == 'on' and _bp is not None:
        try:
            _bp.load_censor_words()
        except Exception:
            pass

    # spaCy
    global _spacy_nlp
    if args.keywordize == 'spacy':
        if _spacy_nlp is None:
            try:
                _spacy_nlp = spacy.load('en_core_web_sm', disable=['ner','textcat'])
            except Exception:
                print("[warn] spaCy model not available; falling back to simple keywordization.")
                args.keywordize = 'simple'

_init_filters()

def _is_spammy_raw(x: str) -> bool:
    if not x:
        return True
    if len(x) < args.min_chars:
        return True
    if _REPEAT_RE.search(x):
        return True
    # lightweight promo spam cues
    low = x.lower()
    for kw in ("buy now", "promo code", "discount", "follow me", "subscribe", "click link", "http://bit.ly", "free followers"):
        if kw in low:
            return True
    return False

def _strip_profanity_tokens(tokens):
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

def _dedupe_tokens(tokens):
    if args.unique_keywords == 'off':
        return tokens
    seen = set(); out = []
    for t in tokens:
        if t in seen: continue
        seen.add(t); out.append(t)
    return out

def keywordize_text(text: str) -> str:
    if not text:
        return ""
    x = str(text)

    # remove URLs/emails/mentions/hashtags
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

    toks = _strip_profanity_tokens(toks)
    toks = _dedupe_tokens(toks)
    if len(toks) < args.min_kw:
        return ""
    return " ".join(toks)

def prepare_post(raw_text: str, title: str = "") -> str | None:
    if not raw_text:
        return None
    full = raw_text
    if title and title.strip() and title not in raw_text:
        full = f"{title.strip()}\n{raw_text}"
    if _is_spammy_raw(full):
        return None
    full = " ".join(full.split())
    kw = keywordize_text(full)
    if not kw:
        return None
    return kw

# ---------------------------
# Streaming writers (train/val)
# ---------------------------
dedupe_hashes = set()

train_f = train_path.open('w', encoding='utf-8')
val_f   = val_path.open('w',   encoding='utf-8')

def emit_example(sample: str):
    """Route a sample to train/val (approx ratio), with dedupe by sha1 of text."""
    sha = hashlib.sha1(sample.encode('utf-8')).hexdigest()
    if len(dedupe_hashes) < args.dedupe_cap:
        if sha in dedupe_hashes:
            return
        dedupe_hashes.add(sha)
    # add EOT separator
    line = sample + "\n"
    if random.random() < args.val_ratio:
        val_f.write(line)
    else:
        train_f.write(line)

# ---------------------------
# Tokenize to .bin (stream, no token cap)
# ---------------------------
def tokenize_text_file_to_bin(txt_path: Path):
    if GPT2TokenizerFast is None:
        raise RuntimeError("transformers not installed (pip install transformers). Needed for .bin tokenization.")
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    if tok.vocab_size > 65535:
        raise ValueError(f"Vocab {tok.vocab_size} exceeds uint16 range (change dtype to uint32 if needed).")
    bin_path = txt_path.with_suffix('.bin')
    print(f"[tokenize] {txt_path} -> {bin_path} (streaming, no token cap)")
    with txt_path.open('r', encoding='utf-8') as fin, open(bin_path, 'wb') as fout:
        for line in fin:
            ids = tok.encode(line)
            np.asarray(ids, dtype=np.uint16).tofile(fout)

# ---------------------------
# Datasets
# ---------------------------
def run_kamaruladha():
    slug = "kamaruladha/mental-disorders-identification-reddit-nlp"
    d = cache_dir / "mental_disorders_identification"
    kaggle_download(slug, d)
    csvs = sorted(d.glob("*.csv"))
    if not csvs:
        return
    print(f"[+] Processing {csvs[0].name}")
    for row in read_csv_any(csvs[0]):
        text = (row.get('text') or row.get('body') or row.get('post') or "")
        title = (row.get('title') or "")
        post = prepare_post(text, title)
        if not post: continue
        label = (row.get('subreddit') or row.get('label') or "").strip()
        if not label: continue
        low = label.lower()
        mapping = {"adhd":"ADHD","ptsd":"PTSD","ocd":"OCD"}
        label = mapping.get(low, label.capitalize() if low==label else label)
        emit_example(f"Post: {post}\nTask: Identify which mental health condition this post is about.\nAnswer: {label}<|endoftext|>")

def run_bipolar():
    slug = "michellevp/mental-health-dataset-bipolar"
    d = cache_dir / "mental_health_bipolar"
    kaggle_download(slug, d)
    f = next((p for p in d.glob("*") if p.suffix.lower() in (".csv",".xlsx")), None)
    if not f: return
    print(f"[+] Processing {f.name}")
    rows = read_csv_any(f) if f.suffix.lower()==".csv" else (pd.read_excel(f).to_dict(orient="records") if pd else [])
    for row in rows:
        text = (row.get('text') or row.get('post') or row.get('content') or "")
        post = prepare_post(text)
        if not post: continue
        sent = (row.get('sentiment') or row.get('sentiment_label') or "")
        if not str(sent).strip(): continue
        s = str(sent).lower()
        if s == '1' or 'pos' in s: sent = "Positive"
        elif s == '-1' or 'neg' in s: sent = "Negative"
        elif s == '0' or 'neu' in s: sent = "Neutral"
        else: sent = str(sent).capitalize()
        emit_example(f"Post: {post}\nTask: Determine the sentiment expressed in the following bipolar discussion post.\nAnswer: {sent}<|endoftext|>")
        risk = (row.get('risk_factor') or "").strip()
        if risk:
            emit_example(f"Post: {post}\nTask: Identify any risk factor mentioned in the post.\nAnswer: {risk}<|endoftext|>")

def run_anxiety_binary():
    slug = "michellevp/predicting-anxiety-in-mental-health-data"
    d = cache_dir / "mental_health_anxiety"
    kaggle_download(slug, d)
    f = next((p for p in d.glob("*") if p.suffix.lower() in (".csv",".xlsx")), None)
    if not f: return
    print(f"[+] Processing {f.name}")
    rows = read_csv_any(f) if f.suffix.lower()==".csv" else (pd.read_excel(f).to_dict(orient="records") if pd else [])
    for r in rows:
        text = (r.get('text') or r.get('post') or r.get('content') or "")
        post = prepare_post(text)
        if not post: continue
        val = (r.get('anxiety') or r.get('label') or r.get('class') or "")
        val = str(val).strip().lower()
        if not val: continue
        ans = "Yes" if val in ("1","true","yes","anxiety") else "No"
        emit_example(f"Post: {post}\nTask: Does the following post indicate an anxiety disorder? Answer Yes or No.\nAnswer: {ans}<|endoftext|>")

def run_dreaddit():
    slug = "shuvojitdas/stress-analysis"
    d = cache_dir / "stress_analysis"
    kaggle_download(slug, d)
    files = list(d.glob("*.csv")) or list(d.glob("*.tsv"))
    if not files: return
    print("[+] Processing Dreaddit files...")
    for f in files:
        for row in read_csv_any(f):
            text = (row.get('text') or row.get('post') or row.get('body') or "")
            post = prepare_post(text)
            if not post: continue
            lab = (row.get('label') or row.get('stress') or row.get('y') or "")
            ans = "Yes" if str(lab).strip().lower() in ("1","yes","true","stressed") else "No"
            emit_example(f"Post: {post}\nTask: Determine if the user is stressed in the following post. Answer Yes or No.\nAnswer: {ans}<|endoftext|>")

def run_neel():
    slug = "neelghoshal/reddit-mental-health-data"
    d = cache_dir / "reddit_mental_health_data"
    kaggle_download(slug, d)
    f = next((p for p in d.glob("*") if p.suffix.lower() in (".csv",".tsv")), None)
    if not f: return
    print(f"[+] Processing {f.name}")
    mapping = {'0':"Stress",'1':"Depression",'2':"Bipolar",'3':"Anxiety",'4':"PTSD"}
    for row in read_csv_any(f):
        text = (row.get('text') or row.get('post') or row.get('body') or "")
        post = prepare_post(text)
        if not post: continue
        lab = (row.get('target') or row.get('label') or row.get('class') or "")
        lab = str(lab).strip()
        if not lab: continue
        name = mapping.get(lab, lab.capitalize())
        emit_example(f"Post: {post}\nTask: Identify the mental health issue discussed in this post (Stress, Depression, Bipolar, Anxiety, or PTSD).\nAnswer: {name}<|endoftext|>")

def run_social_anxiety():
    slug = "natezhang123/social-anxiety-dataset"
    d = cache_dir / "social_anxiety_dataset"
    kaggle_download(slug, d)
    preferred = d / "enhanced_anxiety_dataset.csv"
    f = preferred if preferred.exists() else next((p for p in d.glob("*") if p.suffix.lower() in (".csv",".xlsx")), None)
    if not f:
        print("[!] Social Anxiety dataset missing; skipping.")
        return
    print(f"[+] Processing {f.name}")
    rows = read_csv_any(f) if f.suffix.lower()==".csv" else (pd.read_excel(f).to_dict(orient="records") if pd else [])
    if not rows:
        print("[!] Social Anxiety parsed empty; skipping.")
        return
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
        print("[!] Could not infer Social Anxiety label; skipping.")
        return
    # numeric?
    raw_vals = [str(r.get(label_col,"")).strip() for r in rows if str(r.get(label_col,"")).strip()!=""]
    def _to_float(x):
        try: return float(x)
        except: return None
    numeric_vals = [v for v in (_to_float(x) for x in raw_vals) if v is not None]
    if numeric_vals:
        arr = np.array(numeric_vals, dtype=float)
        if np.allclose(arr.min(), arr.max()):
            q1=q2=q3=arr.min(); mode = "degenerate"
        else:
            q1,q2,q3 = np.quantile(arr, [0.25,0.5,0.75]); mode = "quartiles"
        def num2sev(x):
            if mode=="degenerate": return "Moderate"
            if x<=q1: return "None"
            elif x<=q2: return "Mild"
            elif x<=q3: return "Moderate"
            else: return "Severe"
    else:
        def cat2sev(s):
            sl = s.lower()
            if sl in ("0","none","no","low","lowest","minimal"): return "None"
            if sl in ("1","mild","slight","light"): return "Mild"
            if sl in ("2","moderate","med","medium"): return "Moderate"
            if sl in ("3","severe","high","very high"): return "Severe"
            if "none" in sl: return "None"
            if "mild" in sl or "low" in sl: return "Mild"
            if "moderate" in sl or "mid" in sl: return "Moderate"
            if "severe" in sl or "high" in sl: return "Severe"
            return "Moderate"
    # pick up to 3 informative features (not the label)
    feat_pool = [k for k in keys0 if k != label_col]
    def score_feat(name):
        n=_norm(name); score=0
        for t in ("anxiety","social","avoid","fear","score","scale","sleep","caffeine","alcohol","exercise"): 
            if t in n: score+=1
        return score
    feat_pool.sort(key=score_feat, reverse=True)
    chosen = feat_pool[:3]
    for r in rows:
        raw = r.get(label_col,"")
        if raw is None or str(raw).strip()=="":
            continue
        sev = num2sev(_to_float(raw)) if numeric_vals and _to_float(raw) is not None else (cat2sev(str(raw)) if not numeric_vals else None)
        if sev is None: 
            continue
        pieces=[]
        for fk in chosen:
            v = str(r.get(fk,"")).strip()
            if v and v.lower()!="nan":
                pieces.append(f"{fk}: {v}")
        desc = "; ".join(pieces) if pieces else "various lifestyle and screening features available"
        # keywordize the constructed sentence too
        post = prepare_post(f"The individual reports: {desc}.")
        if not post: continue
        emit_example(f"Post: {post}\nTask: Classify the person's social anxiety level (None, Mild, Moderate, or Severe).\nAnswer: {sev}<|endoftext|>")

def run_rmhd():
    if args.skip_rmhd:
        print("[*] Skipping RMHD as requested.")
        return
    slug = "entenam/reddit-mental-health-dataset"
    d = cache_dir / "reddit_mental_health_dataset"
    kaggle_download(slug, d)
    cands = [p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in (".csv",".tsv",".xlsx",".json",".jsonl")]
    if not cands:
        print("[!] No RMHD files found; skipping.")
        return
    print(f"[+] Processing Entenam RMHD ({len(cands)} file(s) found; recursive).")
    rmhd_seen_texts = set()
    def canon_disorder(name: str) -> str:
        if not name: return ""
        low = name.strip().lower()
        if low.startswith("r/"): low = low[2:]
        low = low.replace("_"," ")
        for k,v in {"anxiety":"Anxiety","depression":"Depression","bipolar":"Bipolar","ptsd":"PTSD","adhd":"ADHD","stress":"Stress","ocd":"OCD"}.items():
            if k in low: return v
        return " ".join(w.capitalize() for w in low.split())
    def handle(row: dict):
        title = (row.get("title") or "").strip()
        tfields = ("text","post","content","body","selftext","comment","message","description")
        raw = ""
        for k in tfields:
            v = row.get(k)
            if v and str(v).strip():
                raw = str(v); break
        if not raw and title:
            raw = title
        if not raw: return
        post = prepare_post(raw, title)
        if not post: return
        if post in rmhd_seen_texts:
            pass
        elif len(rmhd_seen_texts) < args.rmhd_seen_cap:
            rmhd_seen_texts.add(post)
        disorder = None
        for k in ("subreddit","condition","category","label","mental_illness","diagnosis","target","class"):
            v = row.get(k)
            if v and str(v).strip():
                disorder = canon_disorder(str(v)); break
        if disorder:
            emit_example(f"Post: {post}\nTask: Identify which mental health condition this post is about.\nAnswer: {disorder}<|endoftext|>")
    for p in sorted(cands):
        print(f"    -> {p.relative_to(d)}")
        ext = p.suffix.lower()
        try:
            if ext in (".csv",".tsv"):
                with p.open('r', encoding='utf-8', errors='ignore') as f:
                    sample = f.read(8192); f.seek(0)
                    try:
                        dialect = csv.Sniffer().sniff(sample)
                    except Exception:
                        dialect = csv.excel_tab if ext==".tsv" else csv.excel
                    for row in csv.DictReader(f, dialect=dialect):
                        handle(row)
            elif ext == ".xlsx":
                if pd is None:
                    print("[!] pandas not installed; skipping XLSX:", p.name); continue
                for row in pd.read_excel(p).to_dict(orient="records"):
                    handle(row)
            elif ext == ".json":
                with p.open('r', encoding='utf-8', errors='ignore') as f:
                    obj = json.load(f)
                rows = obj.get("data", []) if isinstance(obj, dict) else (obj if isinstance(obj, list) else [])
                for row in rows:
                    if isinstance(row, dict): handle(row)
            elif ext == ".jsonl":
                with p.open('r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        line=line.strip()
                        if not line: continue
                        try:
                            row = json.loads(line)
                            if isinstance(row, dict): handle(row)
                        except Exception:
                            continue
        except Exception as e:
            print(f"[!] Failed parsing {p.name}: {e}")

def run_cid007():
    slug = "cid007/mental-disorder-classification"
    d = cache_dir / "mental_disorder_classification"
    kaggle_download(slug, d)
    pth = None
    for ext in ("*.csv","*.tsv","*.xlsx"):
        L = list(d.rglob(ext))
        if L: pth = L[0]; break
    if not pth:
        print("[!] CID007 missing; skipping."); return
    print(f"[+] Processing {pth.relative_to(d)}")
    rows = read_csv_any(pth) if pth.suffix.lower() in (".csv",".tsv") else (pd.read_excel(pth).to_dict(orient="records") if pd else [])
    if not rows:
        print("[!] CID007 empty; skipping."); return
    keys0 = list(rows[0].keys())
    def norm(s): return s.lower().strip().replace(" ","_")
    label_col = None
    pref = ("diagnos","disorder","label","class","target","illness","condition","result")
    for k in keys0:
        if any(w in norm(k) for w in pref):
            label_col = k; break
    if not label_col:
        if pd is not None:
            dfh = pd.DataFrame(rows)
            for k in keys0:
                try:
                    u = dfh[k].nunique(dropna=True)
                    if 2 <= u <= 8 and not pd.api.types.is_numeric_dtype(dfh[k]): label_col = k; break
                except Exception:
                    pass
    if not label_col:
        print("[!] CID007 label not found; skipping."); return
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
    added = 0
    for r in rows:
        diag = clean_diag(r.get(label_col,""))
        if not diag: continue
        pres = []
        for c in symptom_cols:
            if include_val(r.get(c,"")):
                pres.append(str(c).replace("_"," ").strip().title())
        if not pres and diag != "Normal": 
            continue
        if not pres and diag == "Normal":
            pres = ["No significant psychiatric symptoms"]
        # keywordize synthetic sentence too
        post = prepare_post(f"Patient exhibits the following symptoms: {', '.join(pres[:6])}.")
        if not post: continue
        emit_example(f"Post: {post}\nTask: Determine the most likely diagnosis for this patient.\nAnswer: {diag}<|endoftext|>")
        added += 1
    print(f"[info] CID007 added {added} examples.")

def run_panic():
    slug = "muhammadshahidazeem/panic-disorder-detection-dataset"
    d = cache_dir / "panic_disorder_detection"
    kaggle_download(slug, d)
    files = [p for p in d.glob("*") if p.suffix.lower() in (".csv",".tsv")]
    if not files: return
    print("[+] Processing Panic Disorder CSVs...")
    for f in files:
        rows = read_csv_any(f)
        if not rows: continue
        k0 = list(rows[0].keys())
        label_key = next((k for k in k0 if ("panic" in k.lower()) and any(x in k.lower() for x in ("label","disorder","target"))), None)
        if not label_key:
            label_key = k0[-1]
        feat_keys = [k for k in k0 if k != label_key]
        for r in rows:
            lv = str(r.get(label_key,"")).strip().lower()
            ans = "Yes" if lv in ("1","yes","true","panic") else "No"
            # build compact case description
            parts=[]
            for fk in feat_keys:
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
            if not post: continue
            emit_example(f"Post: {post}\nTask: Is this a case of Panic Disorder? Answer Yes or No.\nAnswer: {ans}<|endoftext|>")

def run_adhd():
    slug = "jerseyneo/reddit-adhd-dataset"
    d = cache_dir / "reddit_adhd_dataset"
    kaggle_download(slug, d)
    csvs = sorted(d.glob("*.csv"))
    if not csvs:
        print("[!] ADHD dataset CSVs not found; skipping."); return
    print("[+] Processing ADHD CSVs (streaming)...")
    candidate_fields = ("selftext","body","comment","text","content","message","post","description")
    for p in csvs:
        print(f"    -> {p.name}")
        with p.open('r', encoding='utf-8', errors='ignore') as f:
            sample = f.read(8192); f.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample)
            except Exception:
                dialect = csv.excel
            reader = csv.DictReader(f, dialect=dialect)
            for row in reader:
                raw = None
                for k in candidate_fields:
                    v = row.get(k)
                    if v and str(v).strip():
                        raw = str(v); break
                if not raw:
                    joined = " ".join(str(row.get(k,"")).strip() for k in ("title","selftext","body") if str(row.get(k,"")).strip()).strip()
                    raw = joined if joined else None
                if not raw: continue
                title = (row.get("title") or "")
                post = prepare_post(raw, title)
                if not post: continue
                emit_example(f"Post: {post}\nTask: Identify which mental health condition this post is about.\nAnswer: ADHD<|endoftext|>")

# ---------------------------
# DSM-5 injection
# ---------------------------
def inject_dsm():
    p = Path(args.dsm_file)
    if not args.include_dsm:
        print("[*] DSM-5 injection disabled."); return
    if not p.is_file():
        print(f"[!] DSM-5 file not found at {p}; skipping DSM injection."); return
    print(f"[+] Injecting DSM-5 knowledge from {p}")
    with p.open('r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    disorder = None; bucket=[]
    def flush():
        if disorder and bucket:
            crit = " ".join([x.strip() for x in bucket if x.strip()])
            if crit:
                post = "(DSM-5 Reference)"  # also keywordized by pipeline (but keep as-is)
                emit_example(f"Post: {keywordize_text(post)}\nTask: List the DSM-5 diagnostic criteria for {disorder}.\nAnswer: {crit}<|endoftext|>")
    for line in lines:
        if line.isupper() and line.strip() and len(line.split()) < 10:
            flush()
            disorder = line.strip().title()
            bucket = []
        else:
            if disorder:
                bucket.append(line)
    flush()

# ---------------------------
# Run all datasets
# ---------------------------
try:
    run_kamaruladha()
    run_bipolar()
    run_anxiety_binary()
    run_dreaddit()
    run_neel()
    run_social_anxiety()
    run_rmhd()
    run_cid007()
    run_panic()
    run_adhd()
    inject_dsm()
finally:
    train_f.close()
    val_f.close()

# ---------------------------
# Tokenize to .bin (automatic)
# ---------------------------
tokenize_text_file_to_bin(train_path)
tokenize_text_file_to_bin(val_path)

print("[done] all steps complete.")
