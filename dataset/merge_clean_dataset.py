import os, re, csv, argparse, random, unicodedata
import numpy as np
from pathlib import Path

# Optional deps (used if present)
try:
    import pandas as pd
except Exception:
    pd = None

try:
    from transformers import GPT2TokenizerFast
except Exception:
    GPT2TokenizerFast = None

# ----------------------------
# CLI
# ----------------------------
p = argparse.ArgumentParser("Merge & clean mental-health datasets + DSM validation + streaming tokenization")
p.add_argument("--train-out", type=str, required=True)
p.add_argument("--val-out",   type=str, required=True)
p.add_argument("--val-ratio", type=float, default=0.10)
p.add_argument("--include-dsm", action="store_true", help="also inject DSM-5 Q&A from --dsm-file")
p.add_argument("--dsm-file", type=str, default="DSM-5.txt")
p.add_argument("--dsm-validate", type=str, choices=["off","soft","strict"], default="soft",
               help="validate disorder-labeled items via DSM-5 keywords")
p.add_argument("--rejects-out", type=str, default="dataset/data/cleaning_rejects.csv")
p.add_argument("--min-words", type=int, default=5)
p.add_argument("--max-words", type=int, default=400)
p.add_argument("--profanity-file", type=str, default="")
p.add_argument("--seed", type=int, default=42)
p.add_argument("--no-shuffle", action="store_true")
args = p.parse_args()

# ----------------------------
# Paths & Helpers
# ----------------------------
train_out = Path(args.train_out)
val_out   = Path(args.val_out)
train_out.parent.mkdir(parents=True, exist_ok=True)
val_out.parent.mkdir(parents=True, exist_ok=True)
rejects_out = Path(args.rejects_out)
rejects_out.parent.mkdir(parents=True, exist_ok=True)

cache_dir = Path("data_cache"); cache_dir.mkdir(exist_ok=True)

def download_kaggle_dataset(dataset_slug: str, dest_dir: Path):
    if dest_dir.exists() and any(dest_dir.iterdir()):
        print(f"[+] {dataset_slug} already downloaded in {dest_dir}")
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"[-] Downloading {dataset_slug} ...")
    ret = os.system(f'kaggle datasets download -d {dataset_slug} -p "{dest_dir}" --quiet --unzip')
    if ret != 0:
        raise RuntimeError(f"Failed to download {dataset_slug}. Is Kaggle API configured?")

def read_csv_any(path: Path):
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            sample = f.read(4096); f.seek(0)
            try: dialect = csv.Sniffer().sniff(sample)
            except Exception: dialect = csv.excel_tab if path.suffix.lower()==".tsv" else csv.excel
            rdr = csv.DictReader(f, dialect=dialect)
            return [r for r in rdr]
    except Exception:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            rdr = csv.DictReader(f)
            return [r for r in rdr]

# ----------------------------
# Cleaning: regex/NLP-ish heuristics
# ----------------------------
URL_RE   = re.compile(r"https?://\S+|www\.\S+", re.I)
USER_RE  = re.compile(r"u/[A-Za-z0-9_-]+|@[A-Za-z0-9_]+")
SUB_RE   = re.compile(r"r/[A-Za-z0-9_]+")
CODE_RE  = re.compile(r"`{1,3}.*?`{1,3}", re.S)
MULTI_WS = re.compile(r"\s+")
REPEAT_PUNCT = re.compile(r"([!?.,])\1{3,}")
REPEAT_CHAR  = re.compile(r"(.)\1{4,}")  # aaaaa -> aaa
EMOJI_RE = re.compile(
    "[" +
    "\U0001F600-\U0001F64F" +  # emoticons
    "\U0001F300-\U0001F5FF" +  # symbols & pictographs
    "\U0001F680-\U0001F6FF" +  # transport & map
    "\U0001F1E0-\U0001F1FF" +  # flags
    "\U00002700-\U000027BF" +  # dingbats
    "\U00002600-\U000026FF" +  # misc
    "]+", flags=re.UNICODE)

BOILERPLATE_PHRASES = [
    "subscribe", "follow me", "buy now", "promo code", "click here",
    "visit my profile", "like and share", "giveaway", "discount",
    "dm me", "telegram", "whatsapp", "bit.ly", "tinyurl", " t.me/",
]
TRASH_TOKENS = {"[deleted]", "[removed]"}

def load_profanity_list(path: str):
    built_in = {
        # small placeholder list; you can expand via --profanity-file
        "fuck","shit","bitch","asshole","bastard","dumbass","douche","slut","whore"
    }
    if not path: return built_in
    p = Path(path)
    if not p.exists(): return built_in
    words = set(built_in)
    with p.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            w = line.strip().lower()
            if w: words.add(w)
    return words

PROFANITY = load_profanity_list(args.profanity_file)

def _strip_noise(s: str) -> str:
    s = s.replace("\u200b","")              # zero-width
    s = CODE_RE.sub(" ", s)
    s = URL_RE.sub(" ", s)
    s = USER_RE.sub(" ", s)
    s = SUB_RE.sub(" ", s)
    s = EMOJI_RE.sub(" ", s)
    s = REPEAT_PUNCT.sub(r"\1\1", s)
    s = REPEAT_CHAR.sub(r"\1\1\1", s)
    s = MULTI_WS.sub(" ", s).strip()
    return s

def _word_count(s: str) -> int:
    return len([w for w in s.split() if w])

def _is_offtopic(s: str) -> bool:
    low = s.lower()
    if any(t in low for t in TRASH_TOKENS): return True
    if sum(1 for c in s if c.isalpha()) < 0.4 * max(1,len(s)): return True
    if any(ph in low for ph in BOILERPLATE_PHRASES): return True
    return False

def _has_profanity(s: str) -> bool:
    tokens = re.findall(r"[a-zA-Z']{2,}", s.lower())
    return any(t in PROFANITY for t in tokens)

def clean_and_filter(text: str, min_words: int, max_words: int):
    """Returns (ok, cleaned_text, reason_if_rejected)"""
    if not text: return (False, "", "empty")
    s = _strip_noise(str(text))
    if _is_offtopic(s): return (False, "", "offtopic_or_trash")
    wc = _word_count(s)
    if wc < min_words: return (False, "", f"too_short({wc})")
    if wc > max_words: return (False, "", f"too_long({wc})")
    if _has_profanity(s): return (False, "", "profanity")
    return (True, s, "")

# ----------------------------
# DSM-5 keyword map & validation
# ----------------------------
def build_dsm_keyword_map(dsm_path: Path):
    """
    Parse DSM-5 text to a simple keyword map per disorder.
    Assumes sections headed by ALL-CAPS disorder lines. Produces a set of
    de-duplicated, lowercased keywords (minus stopwords & tiny tokens).
    """
    if not dsm_path.exists():
        return {}

    with dsm_path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()

    disorder = None
    bag = {}
    STOP = set("""
        the a an of and or to for with without in on at by from into out over under above below
        as is are was were be been being this that those these it its their them they he she his her
        have has had do does did not no nor but if then while when where who whom whose which what
        i you we us our your my mine yours theirs herself himself itself ourselves themselves
    """.split())

    def feed(d, line):
        tokens = re.findall(r"[a-zA-Z][a-zA-Z\-]{2,}", line.lower())
        toks = [t.strip("-") for t in tokens if len(t.strip("-"))>=3 and t not in STOP]
        if toks:
            bag.setdefault(d, set()).update(toks)

    for line in lines:
        if line.strip() and line.isupper() and len(line.split()) < 10:
            disorder = line.title().strip()
            bag.setdefault(disorder, set())
        else:
            if disorder:
                feed(disorder, line)

    # Canonical keys to map dataset labels onto
    canonical = {
        "Anxiety":"Anxiety",
        "Major Depressive Disorder":"Depression",
        "Depression":"Depression",
        "Bipolar I Disorder":"Bipolar",
        "Bipolar Ii Disorder":"Bipolar",
        "Bipolar":"Bipolar",
        "Posttraumatic Stress Disorder":"PTSD",
        "Ptsd":"PTSD",
        "Attention-Deficit/Hyperactivity Disorder":"ADHD",
        "Adhd":"ADHD",
        "Obsessive-Compulsive Disorder":"OCD",
        "Ocd":"OCD",
    }
    out = {"Anxiety":set(), "Depression":set(), "Bipolar":set(), "PTSD":set(), "ADHD":set(), "OCD":set()}
    for k, v in bag.items():
        key = canonical.get(k, None)
        if key in out:
            out[key].update(v)
    # prune overly common stems
    for k in out:
        out[k] = {w for w in out[k] if len(w)>=4}
    return out

DSM_KW = build_dsm_keyword_map(Path(args.dsm_file))
print(f"[info] DSM map built for: {', '.join([k for k,v in DSM_KW.items() if v]) or 'none'}")

def normalize_label(label: str):
    if not label: return None
    s = label.lower().strip()
    s = s.replace("_"," ").replace("-"," ")
    if "ptsd" in s: return "PTSD"
    if "adhd" in s: return "ADHD"
    if "bipolar" in s: return "Bipolar"
    if "depress" in s: return "Depression"
    if "anx" in s: return "Anxiety"
    if "ocd" in s: return "OCD"
    return None  # only validate known DSM target classes

def dsm_validate(label: str, text: str, mode: str):
    """Return (ok, matched_keywords_set_or_empty)."""
    if mode == "off": return (True, set())
    cls = normalize_label(label)
    if not cls or cls not in DSM_KW or not DSM_KW[cls]:
        return (True, set())  # nothing to validate against

    words = set(re.findall(r"[a-zA-Z][a-zA-Z\-]{2,}", text.lower()))
    hits = {w for w in words if w in DSM_KW[cls]}
    if mode == "soft":
        return (len(hits) >= 1, hits)
    else:  # strict
        return (len(hits) >= 2, hits)

# ----------------------------
# Rejection logging & dedupe
# ----------------------------
REJECT_FIELDS = ["source","reason","label","raw_excerpt"]
rej_fp = rejects_out.open("w", newline="", encoding="utf-8")
rej_csv = csv.DictWriter(rej_fp, fieldnames=REJECT_FIELDS); rej_csv.writeheader()

SEEN = set()  # dedupe on cleaned text (exact match)
def maybe_emit(post_text: str, task: str, answer: str, source: str, label_for_dsm: str|None=None, min_words=None, max_words=None):
    ok, cleaned, reason = clean_and_filter(post_text, min_words or args.min_words, max_words or args.max_words)
    if not ok:
        rej_csv.writerow({"source":source,"reason":reason,"label":label_for_dsm or "", "raw_excerpt":(post_text or "")[:280]})
        return None
    # DSM validation (only disorder tasks)
    v_ok, hits = dsm_validate(label_for_dsm or "", cleaned, args.dsm_validate)
    if not v_ok:
        rej_csv.writerow({"source":source,"reason":f"dsm_validate_fail({args.dsm_validate})", "label":label_for_dsm or "", "raw_excerpt":cleaned[:280]})
        return None
    # dedupe
    key = cleaned.lower()
    if key in SEEN:
        rej_csv.writerow({"source":source,"reason":"duplicate", "label":label_for_dsm or "", "raw_excerpt":cleaned[:280]})
        return None
    SEEN.add(key)
    return f"Post: {cleaned}\nTask: {task}\nAnswer: {answer}<|endoftext|>"

# ----------------------------
# DATASETS
# ----------------------------

all_lines = []

# 1) KamarulAdha: mental-disorders-identification-reddit-nlp
slug = "kamaruladha/mental-disorders-identification-reddit-nlp"
d = cache_dir/"mental_disorders_identification"; download_kaggle_dataset(slug, d)
csvs = sorted(d.glob("*.csv"))
if not csvs: raise FileNotFoundError(f"No CSV for {slug}")
print(f"[+] Processing {csvs[0].name}")
for row in read_csv_any(csvs[0]):
    text = (row.get("text") or row.get("body") or row.get("post") or "").strip()
    title = (row.get("title") or "").strip()
    if title and title not in text: text = f"{title}\n{text}".strip()
    if not text: continue
    label = (row.get("subreddit") or row.get("label") or "").strip()
    low = label.lower()
    if low == "adhd": label = "ADHD"
    elif low == "ptsd": label = "PTSD"
    elif low == "ocd": label = "OCD"
    elif label.islower(): label = label.capitalize()
    sample = maybe_emit(text, "Identify which mental health condition this post is about.", label, "kamaruladha", label_for_dsm=label)
    if sample: all_lines.append(sample)

# 2) MichelleVP Bipolar (sentiment)
slug = "michellevp/mental-health-dataset-bipolar"
d = cache_dir/"mental_health_bipolar"; download_kaggle_dataset(slug, d)
files = list(d.glob("*"))
f = next((x for x in files if x.suffix.lower() in (".csv",".xlsx")), None)
if not f: raise FileNotFoundError(f"No CSV/XLSX for {slug}")
print(f"[+] Processing {f.name}")
rows = read_csv_any(f) if f.suffix.lower()==".csv" else (pd.read_excel(f).to_dict(orient="records") if pd else [])
for row in rows:
    text = (row.get("text") or row.get("post") or row.get("content") or "").strip()
    if not text: continue
    sent = (row.get("sentiment") or row.get("sentiment_label") or "").strip().lower()
    if   sent=="1" or "pos" in sent: sent = "Positive"
    elif sent=="-1" or "neg" in sent: sent = "Negative"
    elif sent=="0" or "neu" in sent: sent = "Neutral"
    else: sent = sent.capitalize() if sent else None
    if not sent: continue
    sample = maybe_emit(text, "Determine the sentiment expressed in the following bipolar discussion post.", sent, "michellevp_bipolar")
    if sample: all_lines.append(sample)
    risk = (row.get("risk_factor") or "").strip()
    if risk:
        sample = maybe_emit(text, "Identify any risk factor mentioned in the post.", risk, "michellevp_bipolar")
        if sample: all_lines.append(sample)

# 3) MichelleVP Anxiety (binary)
slug = "michellevp/predicting-anxiety-in-mental-health-data"
d = cache_dir/"mental_health_anxiety"; download_kaggle_dataset(slug, d)
f = next((x for x in d.glob("*") if x.suffix.lower() in (".csv",".xlsx")), None)
if not f: raise FileNotFoundError(f"No CSV/XLSX for {slug}")
print(f"[+] Processing {f.name}")
rows = read_csv_any(f) if f.suffix.lower()==".csv" else (pd.read_excel(f).to_dict(orient="records") if pd else [])
for row in rows:
    text = (row.get("text") or row.get("post") or row.get("content") or "").strip()
    if not text: continue
    val = (row.get("anxiety") or row.get("label") or row.get("class") or "").strip().lower()
    if not val: continue
    ans = "Yes" if val in ("1","true","yes","anxiety") else "No"
    sample = maybe_emit(text, "Does the following post indicate an anxiety disorder? Answer Yes or No.", ans, "michellevp_anxiety",
                        label_for_dsm="Anxiety" if ans=="Yes" else None)
    if sample: all_lines.append(sample)

# 4) Dreaddit Stress (Yes/No) – skip DSM validation (stress != DSM diagnosis per se)
slug = "shuvojitdas/stress-analysis"
d = cache_dir/"stress_analysis"; download_kaggle_dataset(slug, d)
files = list(d.glob("*.csv")) or list(d.glob("*.tsv"))
if not files: raise FileNotFoundError(f"No CSV/TSV for {slug}")
print("[+] Processing Dreaddit files...")
for f in files:
    for row in read_csv_any(f):
        text = (row.get("text") or row.get("post") or row.get("body") or "").strip()
        if not text: continue
        lab = (row.get("label") or row.get("stress") or row.get("y") or "").strip().lower()
        ans = "Yes" if lab in ("1","yes","true","stressed") else "No"
        sample = maybe_emit(text, "Determine if the user is stressed in the following post. Answer Yes or No.", ans, "dreaddit")
        if sample: all_lines.append(sample)

# 5) NeelGhoshal Reddit mental health (multi-class)
slug = "neelghoshal/reddit-mental-health-data"
d = cache_dir/"reddit_mental_health_data"; download_kaggle_dataset(slug, d)
f = next((x for x in d.glob("*") if x.suffix.lower() in (".csv",".tsv")), None)
if not f: raise FileNotFoundError(f"No CSV/TSV for {slug}")
print(f"[+] Processing {f.name}")
mapping = {'0':"Stress",'1':"Depression",'2':"Bipolar",'3':"Anxiety",'4':"PTSD"}
for row in read_csv_any(f):
    text = (row.get("text") or row.get("post") or row.get("body") or "").strip()
    if not text: continue
    lab = (row.get("target") or row.get("label") or row.get("class") or "").strip()
    if not lab: continue
    name = mapping.get(lab, lab.capitalize())
    sample = maybe_emit(text, "Identify the mental health issue discussed in this post (Stress, Depression, Bipolar, Anxiety, or PTSD).", name,
                        "neelghoshal", label_for_dsm=name if name in {"Depression","Bipolar","Anxiety","PTSD","ADHD","OCD"} else None)
    if sample: all_lines.append(sample)

# 6) Social Anxiety severity (bin numeric/categorical)
slug = "natezhang123/social-anxiety-dataset"
d = cache_dir/"social_anxiety_dataset"; download_kaggle_dataset(slug, d)
preferred = d/"enhanced_anxiety_dataset.csv"
f = preferred if preferred.exists() else next((x for x in d.glob("*") if x.suffix.lower() in (".csv",".xlsx")), None)
if f:
    print(f"[+] Processing {f.name}")
    rows = read_csv_any(f) if f.suffix.lower()==".csv" else (pd.read_excel(f).to_dict(orient="records") if pd else [])
    if rows:
        keys0 = list(rows[0].keys())
        nk = {k: k.lower().strip().replace(" ","_") for k in keys0}
        pri = ("level","severity","score","class","label","category","status")
        cand = [k for k, n in nk.items() if "anxiety" in n and any(t in n for t in pri)] or \
               [k for k, n in nk.items() if any(t in n for t in pri)]
        label_col = cand[0] if cand else None
        if label_col:
            # choose a few features to surface
            feat_pool = [k for k in keys0 if k != label_col]
            def feat_score(name: str):
                n = nk[name]
                return sum(t in n for t in ("anxiety","social","avoid","fear","score","scale","sleep","caffeine","exercise","alcohol"))
            feat_pool.sort(key=feat_score, reverse=True)
            feats = feat_pool[:3]

            # detect numeric distribution
            def f2(s):
                try: return float(str(s).strip())
                except: return None
            raw = [str(r.get(label_col,"")).strip() for r in rows if str(r.get(label_col,"")).strip()!=""]
            nums = [x for x in (f2(v) for v in raw) if x is not None]
            if len(nums) >= max(50, int(0.2*len(raw))):
                arr = np.array(nums, dtype=float)
                if np.allclose(arr.min(), arr.max()):
                    def sev(x): return "Moderate"
                else:
                    q1,q2,q3 = np.quantile(arr, [0.25,0.5,0.75])
                    def sev(x):
                        x=float(x)
                        if x<=q1: return "None"
                        elif x<=q2: return "Mild"
                        elif x<=q3: return "Moderate"
                        else: return "Severe"
            else:
                def sev(s):
                    sl=str(s).lower()
                    if sl in ("0","none","no","low","lowest","minimal"): return "None"
                    if sl in ("1","mild","slight","light"): return "Mild"
                    if sl in ("2","moderate","med","medium"): return "Moderate"
                    if sl in ("3","severe","high","very high"): return "Severe"
                    if "none" in sl: return "None"
                    if "mild" in sl or "low" in sl: return "Mild"
                    if "moderate" in sl or "mid" in sl: return "Moderate"
                    if "severe" in sl or "high" in sl: return "Severe"
                    return "Moderate"

            for r in rows:
                lbl = r.get(label_col, None)
                if lbl is None or str(lbl).strip()=="": continue
                level = sev(lbl)
                desc = []
                for fk in feats:
                    v = str(r.get(fk,"")).strip()
                    if v and v.lower()!="nan":
                        desc.append(f"{fk}: {v}")
                text = f"The individual reports: {'; '.join(desc) if desc else 'various lifestyle and screening features available'}."
                sample = maybe_emit(text, "Classify the person's social anxiety level (None, Mild, Moderate, or Severe).", level, "social_anxiety")
                if sample: all_lines.append(sample)

# 7) Entenam RMHD (recursive + canonical labels)
slug = "entenam/reddit-mental-health-dataset"
d = cache_dir/"reddit_mental_health_dataset"; download_kaggle_dataset(slug, d)
cands = [p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in (".csv",".tsv",".xlsx",".json",".jsonl")]
if cands:
    print(f"[+] Processing Entenam RMHD ({len(cands)} file(s) found; recursive).")

    def canon_disorder(name: str):
        if not name: return None
        s = name.lower().strip().replace("_"," ")
        if s.startswith("r/"): s = s[2:]
        for k,v in {
            "anxiety":"Anxiety","depression":"Depression","bipolar":"Bipolar",
            "ptsd":"PTSD","adhd":"ADHD","ocd":"OCD","stress":"Stress",
        }.items():
            if k in s: return v
        return None

    def handle_row(row: dict):
        title = (row.get("title") or "").strip()
        text = ""
        for k in ("text","post","content","body","selftext","comment","message","description"):
            v = row.get(k)
            if v and str(v).strip():
                text = str(v).strip(); break
        if not text and title: text = title
        if not text or len(text) < 20: return
        if title and title not in text: text = f"{title}\n{text}"
        dis = None
        for k in ("subreddit","condition","category","label","mental_illness","diagnosis","target","class"):
            v = row.get(k)
            if v and str(v).strip():
                dis = canon_disorder(str(v)); break
        if not dis: dis = None
        sample = maybe_emit(text, "Identify which mental health condition this post is about.", dis or "Unknown", "entenam",
                            label_for_dsm=dis if dis in {"Anxiety","Depression","Bipolar","PTSD","ADHD","OCD"} else None)
        if sample and dis: all_lines.append(sample)

    for pth in sorted(cands):
        print(f"    -> {pth.relative_to(d)}")
        ext = pth.suffix.lower()
        try:
            if ext in (".csv",".tsv"):
                for r in read_csv_any(pth): handle_row(r)
            elif ext==".xlsx" and pd:
                df = pd.read_excel(pth); 
                for r in df.to_dict(orient="records"): handle_row(r)
            elif ext==".json":
                import json
                with pth.open("r", encoding="utf-8") as f:
                    try:
                        obj = json.load(f)
                        rows = obj["data"] if isinstance(obj,dict) and isinstance(obj.get("data"),list) else obj if isinstance(obj,list) else []
                    except Exception:
                        rows = []
                for r in rows:
                    if isinstance(r,dict): handle_row(r)
            elif ext==".jsonl":
                import json
                with pth.open("r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        line=line.strip()
                        if not line: continue
                        try:
                            r=json.loads(line)
                            if isinstance(r,dict): handle_row(r)
                        except Exception:
                            pass
        except Exception as e:
            print(f"[!] Failed parsing {pth.name}: {e}")

# 8) CID007 symptom table → diagnosis
slug = "cid007/mental-disorder-classification"
d = cache_dir/"mental_disorder_classification"; download_kaggle_dataset(slug, d)
sym_path = None
for pat in ("*.csv","*.tsv","*.xlsx"):
    found = list(d.rglob(pat))
    if found: sym_path = found[0]; break
if sym_path:
    print(f"[+] Processing {sym_path.relative_to(d)}")
    rows = read_csv_any(sym_path) if sym_path.suffix.lower() in (".csv",".tsv") else (pd.read_excel(sym_path).to_dict(orient="records") if pd else [])
    if rows:
        keys0 = list(rows[0].keys())
        def norm(s): return s.lower().strip().replace(" ","_")
        label_col = None
        for k in keys0:
            n = norm(k)
            if any(t in n for t in ("diagnos","disorder","label","class","target","illness","condition","result","category")):
                label_col = k; break
        if not label_col and pd:
            df = pd.DataFrame(rows)
            for k in keys0:
                u = df[k].nunique(dropna=True)
                if 2 <= u <= 8 and not pd.api.types.is_numeric_dtype(df[k]): label_col = k; break
        if label_col:
            symptom_cols = [k for k in keys0 if k != label_col]
            def clean_diag(x):
                s = str(x or "").strip().lower()
                for k,v in {"normal":"Normal","anxiety":"Anxiety","depression":"Depression","bipolar":"Bipolar","ptsd":"PTSD","ocd":"OCD","adhd":"ADHD"}.items():
                    if k==s: return v
                return s.title() if s else ""
            def include_sym(v):
                s = str(v or "").strip().lower()
                if s in ("","nan","none"): return False
                try:
                    f=float(s); return f>0.0
                except: return s in ("yes","true","present","high","mild","moderate","severe","1")
            for r in rows:
                diag = clean_diag(r.get(label_col,""))
                if not diag: continue
                feats=[]
                for c in symptom_cols:
                    if include_sym(r.get(c,"")):
                        feats.append(str(c).replace("_"," ").title())
                if not feats and diag!="Normal": continue
                if not feats and diag=="Normal": feats=["No significant psychiatric symptoms"]
                text = f"Patient exhibits the following symptoms: {', '.join(feats[:6])}."
                sample = maybe_emit(text, "Determine the most likely diagnosis for this patient.", diag, "cid007",
                                    label_for_dsm=diag if diag in {"Anxiety","Depression","Bipolar","PTSD","ADHD","OCD"} else None)
                if sample: all_lines.append(sample)

# 9) Panic disorder detection
slug = "muhammadshahidazeem/panic-disorder-detection-dataset"
d = cache_dir/"panic_disorder_detection"; download_kaggle_dataset(slug, d)
csvs = [x for x in d.glob("*") if x.suffix.lower() in (".csv",".tsv")]
if csvs:
    print("[+] Processing Panic Disorder CSVs...")
    for f in csvs:
        rows = read_csv_any(f)
        if not rows: continue
        keys0 = list(rows[0].keys())
        label_key = next((k for k in keys0 if ("panic" in k.lower()) and any(x in k.lower() for x in ("label","disorder","target"))), None) or keys0[-1]
        feat_keys = [k for k in keys0 if k != label_key]
        for r in rows:
            lv = str(r.get(label_key,"")).strip().lower()
            ans = "Yes" if lv in ("1","yes","true","panic") else "No"
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
            text = f"Patient data - {'; '.join(parts)}."
            sample = maybe_emit(text, "Is this a case of Panic Disorder? Answer Yes or No.", ans, "panic_disorder")
            if sample: all_lines.append(sample)

# 10) Reddit ADHD (jerseyneo)
slug = "jerseyneo/reddit-adhd-dataset"
d = cache_dir/"reddit_adhd_dataset"; download_kaggle_dataset(slug, d)
adhd_csvs = sorted(d.glob("*.csv"))
if not adhd_csvs: raise FileNotFoundError(f"No CSV files for {slug}")
print("[+] Processing ADHD CSVs (streaming)...")
for pth in adhd_csvs:
    print(f"    -> {pth.name}")
    with pth.open("r", encoding="utf-8", errors="ignore") as f:
        try:
            sample = f.read(8192); f.seek(0)
            dialect = csv.Sniffer().sniff(sample)
        except Exception:
            dialect = csv.excel
        rdr = csv.DictReader(f, dialect=dialect)
        for row in rdr:
            raw = None
            for key in ("selftext","body","comment","text","content","message","post","description"):
                val = row.get(key)
                if val and str(val).strip():
                    raw = str(val); break
            if not raw:
                joined = " ".join(str(row.get(k,"")).strip() for k in ("title","selftext","body") if str(row.get(k,"")).strip())
                raw = joined if joined else None
            if not raw: continue
            sample = maybe_emit(raw, "Identify which mental health condition this post is about.", "ADHD", "adhd", label_for_dsm="ADHD")
            if sample: all_lines.append(sample)

# ----------------------------
# DSM-5 knowledge injection (optional)
# ----------------------------
if args.include_dsm and Path(args.dsm_file).exists():
    print(f"[+] Injecting DSM-5 knowledge from {args.dsm_file}")
    lines = Path(args.dsm_file).read_text(encoding="utf-8", errors="ignore").splitlines()
    disorder, bucket = None, []
    def flush():
        if disorder and bucket:
            crit = " ".join([x for x in bucket if x]).strip()
            if crit:
                q = f"List the DSM-5 diagnostic criteria for {disorder}."
                sample = maybe_emit("(DSM-5 Reference)", q, crit, "DSM5")
                if sample: all_lines.append(sample)
    for line in lines:
        if line.strip() and line.isupper() and len(line.split())<10:
            flush(); disorder = line.strip().title(); bucket=[]
        else:
            if disorder: bucket.append(line.strip())
    flush()
else:
    if args.include_dsm:
        print(f"[!] DSM file not found at {args.dsm_file}; skipping injection.")

# ----------------------------
# Shuffle, split, write
# ----------------------------
if not args.no_shuffle:
    random.seed(args.seed); random.shuffle(all_lines)

n_total = len(all_lines)
n_val = max(1, int(n_total * args.val_ratio))
val_lines = all_lines[:n_val]
train_lines = all_lines[n_val:]

with train_out.open("w", encoding="utf-8") as ft:
    for s in train_lines: ft.write(s + "\n")
with val_out.open("w", encoding="utf-8") as fv:
    for s in val_lines: fv.write(s + "\n")

rej_fp.close()
print(f"[ok] train={train_out} | val={val_out}")

# ----------------------------
# Streaming tokenization to .bin (no token cap, low RAM)
# ----------------------------
def stream_tokenize_txt_to_bin(txt_path: Path, bin_path: Path):
    if GPT2TokenizerFast is None:
        raise RuntimeError("transformers not installed; required for tokenization")
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    # silence context warnings; we only emit ids, no forward pass
    tok.model_max_length = int(1e30)
    tok.init_kwargs["model_max_length"] = tok.model_max_length

    # write in streaming manner
    with txt_path.open("r", encoding="utf-8") as fin, open(bin_path, "wb") as fout:
        count = 0
        for line in fin:
            ids = tok.encode(line, add_special_tokens=False)
            if ids:
                np.asarray(ids, dtype=np.uint16).tofile(fout)
                count += len(ids)
                if count and count % 5_000_000 == 0:
                    print(f"  ... wrote ~{count:,} tokens to {bin_path.name}")
    print(f"[ok] wrote {bin_path} (tokens ≈ {count:,})")

train_bin = train_out.with_suffix(".bin")
val_bin   = val_out.with_suffix(".bin")
print(f"[tokenize] {train_out} -> {train_bin} (streaming, no token cap)")
stream_tokenize_txt_to_bin(train_out, train_bin)
print(f"[tokenize] {val_out} -> {val_bin} (streaming, no token cap)")
stream_tokenize_txt_to_bin(val_out, val_bin)
