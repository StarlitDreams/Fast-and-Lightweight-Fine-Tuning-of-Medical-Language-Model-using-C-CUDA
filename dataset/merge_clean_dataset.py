import os
import argparse
import csv
import random
import numpy as np
from pathlib import Path

try:
    # Optional: only needed when --emit-bin
    from transformers import GPT2TokenizerFast
except ImportError:
    GPT2TokenizerFast = None

def download_kaggle_dataset(dataset_slug: str, dest_dir: Path):
    """Download & unzip a Kaggle dataset if it's not already present."""
    if dest_dir.exists() and any(dest_dir.iterdir()):
        print(f"[+] {dataset_slug} already downloaded in {dest_dir}")
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"[-] Downloading {dataset_slug} ...")
    ret = os.system(f'kaggle datasets download -d {dataset_slug} -p "{dest_dir}" --quiet --unzip')
    if ret != 0:
        raise RuntimeError(f"Failed to download {dataset_slug}. Is Kaggle API configured?")

def read_csv_any(filepath: Path):
    """Read CSV/TSV into list-of-dicts. Attempts delimiter sniffing; falls back to comma."""
    try:
        with filepath.open('r', encoding='utf-8', errors='ignore') as f:
            sample = f.read(2048)
            f.seek(0)
            dialect = csv.Sniffer().sniff(sample)
            reader = csv.DictReader(f, dialect=dialect)
            return [row for row in reader]
    except Exception:
        with filepath.open('r', encoding='utf-8', errors='ignore') as f:
            reader = csv.DictReader(f)
            return [row for row in reader]

parser = argparse.ArgumentParser(description="Prepare mental health instruction dataset")
parser.add_argument('--output', type=str, default='mental_health_instructions.txt',
                    help="Output text file (all examples).")
parser.add_argument('--dsm-file', type=str, default='DSM-5.txt',
                    help="Path to DSM-5 knowledge text file")
parser.add_argument('--include-dsm', action='store_true',
                    help="Include DSM-5 knowledge Q&A")
parser.add_argument('--emit-bin', action='store_true',
                    help="Also emit a tokenized .bin for llm.c (GPT-2 tokenizer)")
parser.add_argument('--seed', type=int, default=42, help="Shuffle seed")
parser.add_argument('--no-shuffle', action='store_true', help="Disable shuffling")
args = parser.parse_args()

output_path = Path(args.output)
cache_dir = Path("data_cache")
cache_dir.mkdir(exist_ok=True)

all_lines = []

# 1) KamarulAdha: mental-disorders-identification-reddit-nlp
kamarul_slug = "kamaruladha/mental-disorders-identification-reddit-nlp"
kamarul_dir = cache_dir / "mental_disorders_identification"
download_kaggle_dataset(kamarul_slug, kamarul_dir)
kamarul_csvs = sorted(kamarul_dir.glob("*.csv"))
if not kamarul_csvs:
    raise FileNotFoundError(f"No CSV for {kamarul_slug}")
print(f"[+] Processing {kamarul_csvs[0].name}")
for row in read_csv_any(kamarul_csvs[0]):
    text = (row.get('text') or row.get('body') or row.get('post') or "").strip()
    title = (row.get('title') or "").strip()
    if title and title not in text:
        text = f"{title}\n{text}".strip()
    if not text:
        continue
    label = (row.get('subreddit') or row.get('label') or "").strip()
    if not label:
        continue
    low = label.lower()
    if low == "adhd": label = "ADHD"
    elif low == "ptsd": label = "PTSD"
    elif low == "ocd": label = "OCD"
    elif label.islower(): label = label.capitalize()
    post = text.replace("\n", " ").strip()
    task = "Identify which mental health condition this post is about."
    all_lines.append(f"Post: {post}\nTask: {task}\nAnswer: {label}<|endoftext|>")

# 2) MichelleVP Bipolar
bipolar_slug = "michellevp/mental-health-dataset-bipolar"
bipolar_dir = cache_dir / "mental_health_bipolar"
download_kaggle_dataset(bipolar_slug, bipolar_dir)
bipolar_files = list(bipolar_dir.glob("*"))
bipolar_file = next((f for f in bipolar_files if f.suffix.lower() in (".csv", ".xlsx")), None)
if not bipolar_file:
    raise FileNotFoundError(f"No CSV/XLSX for {bipolar_slug}")
print(f"[+] Processing {bipolar_file.name}")
if bipolar_file.suffix.lower() == ".csv":
    bipolar_rows = read_csv_any(bipolar_file)
else:
    try:
        import pandas as pd
    except ImportError:
        raise RuntimeError("Pandas is required to read Excel (install pandas).")
    bipolar_rows = pd.read_excel(bipolar_file).to_dict(orient="records")
for row in bipolar_rows:
    text = (row.get('text') or row.get('post') or row.get('content') or "").strip()
    if not text:
        continue
    sent = (row.get('sentiment') or row.get('sentiment_label') or "").strip()
    if not sent:
        continue
    s = sent.lower()
    if s == '1' or 'pos' in s: sent = "Positive"
    elif s == '-1' or 'neg' in s: sent = "Negative"
    elif s == '0' or 'neu' in s: sent = "Neutral"
    else: sent = sent.capitalize()
    post = text.replace("\n", " ").strip()
    task = "Determine the sentiment expressed in the following bipolar discussion post."
    all_lines.append(f"Post: {post}\nTask: {task}\nAnswer: {sent}<|endoftext|>")
    risk = (row.get('risk_factor') or "").strip()
    if risk:
        all_lines.append(f"Post: {post}\nTask: Identify any risk factor mentioned in the post.\nAnswer: {risk}<|endoftext|>")

# 3) MichelleVP Anxiety (binary)
anx_slug = "michellevp/predicting-anxiety-in-mental-health-data"
anx_dir = cache_dir / "mental_health_anxiety"
download_kaggle_dataset(anx_slug, anx_dir)
anx_file = next((f for f in anx_dir.glob("*") if f.suffix.lower() in (".csv", ".xlsx")), None)
if not anx_file:
    raise FileNotFoundError(f"No CSV/XLSX for {anx_slug}")
print(f"[+] Processing {anx_file.name}")
if anx_file.suffix.lower() == ".csv":
    anx_rows = read_csv_any(anx_file)
else:
    try:
        import pandas as pd
    except ImportError:
        raise RuntimeError("Pandas is required to read Excel (install pandas).")
    anx_rows = pd.read_excel(anx_file).to_dict(orient="records")
for row in anx_rows:
    text = (row.get('text') or row.get('post') or row.get('content') or "").strip()
    if not text:
        continue
    val = (row.get('anxiety') or row.get('label') or row.get('class') or "").strip().lower()
    if not val:
        continue
    ans = "Yes" if val in ("1","true","yes","anxiety") else "No"
    post = text.replace("\n"," ").strip()
    task = "Does the following post indicate an anxiety disorder? Answer Yes or No."
    all_lines.append(f"Post: {post}\nTask: {task}\nAnswer: {ans}<|endoftext|>")

# 4) Dreaddit Stress
dre_slug = "shuvojitdas/stress-analysis"
dre_dir = cache_dir / "stress_analysis"
download_kaggle_dataset(dre_slug, dre_dir)
dre_files = list(dre_dir.glob("*.csv")) or list(dre_dir.glob("*.tsv"))
if not dre_files:
    raise FileNotFoundError(f"No CSV/TSV for {dre_slug}")
print("[+] Processing Dreaddit files...")
for f in dre_files:
    for row in read_csv_any(f):
        text = (row.get('text') or row.get('post') or row.get('body') or "").strip()
        if not text:
            continue
        lab = (row.get('label') or row.get('stress') or row.get('y') or "").strip().lower()
        ans = "Yes" if lab in ("1","yes","true","stressed") else "No"
        post = text.replace("\n"," ").strip()
        task = "Determine if the user is stressed in the following post. Answer Yes or No."
        all_lines.append(f"Post: {post}\nTask: {task}\nAnswer: {ans}<|endoftext|>")

# 5) NeelGhoshal Reddit mental health data
neel_slug = "neelghoshal/reddit-mental-health-data"
neel_dir = cache_dir / "reddit_mental_health_data"
download_kaggle_dataset(neel_slug, neel_dir)
neel_file = next((f for f in neel_dir.glob("*") if f.suffix.lower() in (".csv",".tsv")), None)
if not neel_file:
    raise FileNotFoundError(f"No CSV/TSV for {neel_slug}")
print(f"[+] Processing {neel_file.name}")
mapping = {'0':"Stress",'1':"Depression",'2':"Bipolar",'3':"Anxiety",'4':"PTSD"}
for row in read_csv_any(neel_file):
    text = (row.get('text') or row.get('post') or row.get('body') or "").strip()
    if not text:
        continue
    lab = (row.get('target') or row.get('label') or row.get('class') or "").strip()
    if not lab:
        continue
    name = mapping.get(lab, lab.capitalize())
    post = text.replace("\n"," ").strip()
    task = "Identify the mental health issue discussed in this post (Stress, Depression, Bipolar, Anxiety, or PTSD)."
    all_lines.append(f"Post: {post}\nTask: {task}\nAnswer: {name}<|endoftext|>")

# 6) Social Anxiety dataset (severity)
sad_slug = "natezhang123/social-anxiety-dataset"
sad_dir = cache_dir / "social_anxiety_dataset"
download_kaggle_dataset(sad_slug, sad_dir)
sad_file = next((f for f in sad_dir.glob("*") if f.suffix.lower() in (".csv",".xlsx")), None)
if not sad_file:
    raise FileNotFoundError(f"No CSV/XLSX for {sad_slug}")
print(f"[+] Processing {sad_file.name}")
if sad_file.suffix.lower() == ".csv":
    sad_rows = read_csv_any(sad_file)
else:
    try:
        import pandas as pd
    except ImportError:
        raise RuntimeError("Pandas is required to read Excel (install pandas).")
    sad_rows = pd.read_excel(sad_file).to_dict(orient="records")
label_col = None
if sad_rows:
    for k in sad_rows[0].keys():
        if k.lower() in ("severity","level","anxiety_level","class","label"):
            label_col = k; break
if not label_col:
    raise RuntimeError("Could not find severity label in Social Anxiety dataset.")
feat_keys = [k for k in sad_rows[0].keys() if k != label_col]
chosen = []
for fk in feat_keys:
    lk = fk.lower()
    if any(s in lk for s in ("anxiety","social","avoid","fear")):
        chosen.append(fk)
    if len(chosen) >= 3: break
if not chosen: chosen = feat_keys[:3]
for row in sad_rows:
    sev = str(row.get(label_col,"")).strip()
    if not sev: continue
    if sev.isdigit():
        sev = int(sev)
        sev_text = "None" if sev<=0 else "Mild" if sev==1 else "Moderate" if sev==2 else "Severe"
    else:
        sev_text = sev.capitalize()
    desc = []
    for fk in chosen:
        val = str(row.get(fk,"")).strip()
        if val: desc.append(f"{fk}: {val}")
    if not desc: continue
    post = f"The individual reports: {'; '.join(desc)}."
    task = "Classify the person's social anxiety level (None, Mild, Moderate, or Severe)."
    all_lines.append(f"Post: {post}\nTask: {task}\nAnswer: {sev_text}<|endoftext|>")

# 7) Entenam RMHD (condition + cause)
entenam_slug = "entenam/reddit-mental-health-dataset"
entenam_dir = cache_dir / "reddit_mental_health_dataset"
download_kaggle_dataset(entenam_slug, entenam_dir)
entenam_file = next((f for f in entenam_dir.glob("*") if f.suffix.lower() in (".csv",".tsv",".xlsx",".json")), None)
if not entenam_file:
    raise FileNotFoundError(f"No data file for {entenam_slug}")
print(f"[+] Processing {entenam_file.name}")
if entenam_file.suffix.lower() in (".csv",".tsv"):
    entenam_rows = read_csv_any(entenam_file)
elif entenam_file.suffix.lower() == ".xlsx":
    try:
        import pandas as pd
    except ImportError:
        raise RuntimeError("Pandas is required to read Excel (install pandas).")
    entenam_rows = pd.read_excel(entenam_file).to_dict(orient="records")
elif entenam_file.suffix.lower() == ".json":
    import json
    with open(entenam_file, 'r', encoding='utf-8') as f:
        entenam_rows = json.load(f)
else:
    entenam_rows = []

if not entenam_rows:
    print("[!] Entenam dataset appears empty after parsing.")
else:
    # find fields
    keys0 = list(entenam_rows[0].keys())
    text_field = next((k for k in keys0 if k.lower() in ("text","post","content","body")), None)
    sub_field  = next((k for k in keys0 if k.lower() in ("subreddit","condition","category","label")), None)
    cause_field= next((k for k in keys0 if k.lower() in ("cause","causes","trigger","issue")), None)
    if not text_field:
        raise RuntimeError("No text field found in Entenam dataset.")
    for row in entenam_rows:
        text = str(row.get(text_field,"")).strip()
        if not text: continue
        post = text.replace("\n"," ").strip()
        if sub_field and row.get(sub_field):
            disorder = str(row[sub_field]).strip()
            disorder = disorder.capitalize() if disorder.islower() else disorder
            all_lines.append(f"Post: {post}\nTask: Identify which mental health condition this post is about.\nAnswer: {disorder}<|endoftext|>")
        if cause_field and row.get(cause_field):
            cause = str(row[cause_field]).strip().capitalize()
            all_lines.append(f"Post: {post}\nTask: What appears to be the primary cause of distress in this post?\nAnswer: {cause}<|endoftext|>")

# 8) CID007 symptom table → diagnosis
sym_slug = "cid007/mental-disorder-classification"
sym_dir = cache_dir / "mental_disorder_classification"
download_kaggle_dataset(sym_slug, sym_dir)
sym_file = next((f for f in sym_dir.glob("*") if f.suffix.lower() in (".csv",".tsv",".xlsx")), None)
if not sym_file:
    raise FileNotFoundError(f"No data file for {sym_slug}")
print(f"[+] Processing {sym_file.name}")
if sym_file.suffix.lower() in (".csv",".tsv"):
    sym_rows = read_csv_any(sym_file)
else:
    try:
        import pandas as pd
    except ImportError:
        raise RuntimeError("Pandas is required to read Excel (install pandas).")
    sym_rows = pd.read_excel(sym_file).to_dict(orient="records")
if sym_rows:
    label_col = None
    symptom_cols = []
    for k in sym_rows[0].keys():
        lk = k.lower()
        if lk in ("disorder","condition","diagnosis","label","class"):
            label_col = k
        else:
            symptom_cols.append(k)
    if not label_col or not symptom_cols:
        raise RuntimeError("Unexpected format in symptom dataset.")
    for row in sym_rows:
        diag = str(row.get(label_col,"")).strip()
        if not diag: continue
        diag_text = "Normal" if diag.lower()=="normal" else diag.title()
        desc = []
        for s in symptom_cols:
            val = str(row.get(s,"")).strip()
            if not val or val.lower()=="nan": continue
            include = False
            try:
                if float(val) >= 1: include = True
            except Exception:
                if val.lower() not in ("no","false","none","0"): include = True
            if include:
                name = s.replace("_"," ").strip().title()
                desc.append(name)
        if not desc:
            if diag_text != "Normal": continue
            desc.append("no significant psychiatric symptoms")
        post = f"Patient exhibits the following symptoms: {', '.join(desc[:5])}."
        task = "Determine the most likely diagnosis for this patient."
        all_lines.append(f"Post: {post}\nTask: {task}\nAnswer: {diag_text}<|endoftext|>")

# 9) Panic disorder detection (binary)
panic_slug = "muhammadshahidazeem/panic-disorder-detection-dataset"
panic_dir = cache_dir / "panic_disorder_detection"
download_kaggle_dataset(panic_slug, panic_dir)
panic_csvs = [f for f in panic_dir.glob("*") if f.suffix.lower() in (".csv",".tsv")]
if not panic_csvs:
    raise FileNotFoundError(f"No CSV for {panic_slug}")
print("[+] Processing Panic Disorder CSVs...")
for f in panic_csvs:
    rows = read_csv_any(f)
    if not rows: continue
    keys0 = list(rows[0].keys())
    label_key = next((k for k in keys0 if ("panic" in k.lower()) and any(x in k.lower() for x in ("label","disorder","target"))), None)
    if not label_key:
        label_key = keys0[-1]
    feat_keys = [k for k in keys0 if k != label_key]
    for r in rows:
        lv = str(r.get(label_key,"")).strip().lower()
        ans = "Yes" if lv in ("1","yes","true","panic") else "No"
        parts = []
        for fk in feat_keys:
            v = str(r.get(fk,"")).strip()
            if not v or v.lower()=="nan": continue
            low = fk.lower()
            if low in ("age","gender","sex"):
                parts.append(f"{fk}: {v}")
            elif low in ("symptom","symptoms","chest_pain","sweating","palpitations","dizziness"):
                if v in ("1","yes","true"): parts.append(f"{fk.replace('_',' ')}: yes")
            if len(parts) >= 5: break
        if not parts: parts.append("no notable symptoms")
        post = f"Patient data - {'; '.join(parts)}."
        task = "Is this a case of Panic Disorder? Answer Yes or No."
        all_lines.append(f"Post: {post}\nTask: {task}\nAnswer: {ans}<|endoftext|>")

# X) Reddit ADHD dataset — jerseyneo (posts & comments from r/ADHD and r/adhdwomen)
adhd_slug = "jerseyneo/reddit-adhd-dataset"
adhd_dir = cache_dir / "reddit_adhd_dataset"
download_kaggle_dataset(adhd_slug, adhd_dir)

# Collect all CSVs (typical: ADHD.csv, ADHD-comment.csv, adhdwomen.csv, adhdwomen-comment.csv)
adhd_csvs = sorted(adhd_dir.glob("*.csv"))
if not adhd_csvs:
    raise FileNotFoundError(f"No CSV files found for {adhd_slug}")

print("[+] Processing ADHD CSVs (streaming)...")

# Optional light dedupe on exact text to avoid obvious repeats across files
_seen_texts = set()
MAX_SEEN = 500000  # cap the set size to avoid unbounded RAM

def _maybe_add_example(raw_text: str, row: dict):
    """Prepare a single ADHD example if the text is usable."""
    global _seen_texts
    text = (raw_text or "").strip()
    if not text or len(text) < 20:
        return
    # prepend title when available and not already included
    title = (row.get("title") or "").strip()
    if title and title not in text:
        text = f"{title}\n{text}"

    # de-newline & minimal normalize
    post = " ".join(text.split())

    # dedupe exact matches (helps when both submissions and comments repeat content)
    if post in _seen_texts:
        return
    if len(_seen_texts) < MAX_SEEN:
        _seen_texts.add(post)

    task = "Identify which mental health condition this post is about."
    answer = "ADHD"
    all_lines.append(f"Post: {post}\nTask: {task}\nAnswer: {answer}<|endoftext|>")

for csv_path in adhd_csvs:
    print(f"    -> {csv_path.name}")
    with csv_path.open("r", encoding="utf-8", errors="ignore") as f:
        # robust delimiter sniff (fall back to comma)
        try:
            sample = f.read(8192)
            f.seek(0)
            dialect = csv.Sniffer().sniff(sample)
        except Exception:
            dialect = csv.excel
        reader = csv.DictReader(f, dialect=dialect)

        # common text-bearing fields across submissions & comments
        candidate_fields = (
            "selftext",   # submissions text
            "body",       # comments text (typical Pushshift export)
            "comment",    # alt
            "text", "content", "message", "post", "description"
        )

        for row in reader:
            # pull the first non-empty field in priority order
            raw = None
            for key in candidate_fields:
                val = row.get(key)
                if val and str(val).strip():
                    raw = str(val)
                    break
            # if nothing found, try joining a couple of plausible fragments
            if not raw:
                joined = " ".join(
                    str(row.get(k, "")).strip()
                    for k in ("title", "selftext", "body")
                    if str(row.get(k, "")).strip()
                ).strip()
                raw = joined if joined else None

            if raw:
                _maybe_add_example(raw, row)


# DSM-5 knowledge injection (optional)
if args.include_dsm:
    if os.path.isfile(args.dsm_file):
        print(f"[+] Injecting DSM-5 knowledge from {args.dsm_file}")
        with open(args.dsm_file, 'r', encoding='utf-8') as f:
            dsm_text = f.read()
        lines = dsm_text.splitlines()
        disorder = None
        bucket = []
        def flush():
            if disorder and bucket:
                crit = " ".join([x for x in bucket if x]).strip()
                if crit:
                    q = f"List the DSM-5 diagnostic criteria for {disorder}."
                    all_lines.append(f"Post: (DSM-5 Reference)\nTask: {q}\nAnswer: {crit}<|endoftext|>")
        for line in lines:
            if line.isupper() and line.strip() and len(line.split()) < 10:
                flush()
                disorder = line.strip().title()
                bucket = []
            else:
                if disorder:
                    bucket.append(line.strip())
        flush()
    else:
        print(f"[!] DSM-5 file not found at {args.dsm_file}; skipping DSM injection.")
else:
    print("[*] DSM-5 injection disabled.")

# Shuffle & write
if not args.no_shuffle:
    random.seed(args.seed)
    random.shuffle(all_lines)

with open(output_path, 'w', encoding='utf-8') as out_f:
    for line in all_lines:
        out_f.write(line + "\n")

print(f"[+] Wrote {len(all_lines)} examples to {output_path}")

# Optional: emit GPT-2 token ids as uint16 .bin for llm.c
if args.emit_bin:
    if GPT2TokenizerFast is None:
        raise RuntimeError("transformers not installed; required for --emit-bin")
    print("[+] Tokenizing to .bin ...")
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    if tok.vocab_size > 65535:
        raise ValueError(f"Vocab {tok.vocab_size} exceeds uint16 range; change dtype.")
    with open(output_path, 'r', encoding='utf-8') as f:
        text = f.read()
    ids = tok.encode(text)
    np.asarray(ids, dtype=np.uint16).tofile(output_path.with_suffix('.bin'))
    print(f"[+] Tokenized binary saved to {output_path.with_suffix('.bin')}")
