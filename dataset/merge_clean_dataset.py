import os
import argparse
import csv
import random
import numpy as np
from pathlib import Path

try:
    # Use Huggingface tokenizer for binary conversion if needed
    from transformers import GPT2TokenizerFast
except ImportError:
    GPT2TokenizerFast = None

# Helper function to download a Kaggle dataset if not already cached
def download_kaggle_dataset(dataset_slug: str, dest_dir: Path):
    """Download and unzip Kaggle dataset to destination directory if not already present."""
    if dest_dir.exists() and any(dest_dir.iterdir()):
        print(f"[+] {dataset_slug} already downloaded in {dest_dir}")
        return
    dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"[-] Downloading {dataset_slug} ...")
    # Requires Kaggle API credentials to be configured
    cmd = f"kaggle datasets download -d {dataset_slug} -p {dest_dir} --quiet --unzip"
    ret = os.system(cmd)
    if ret != 0:
        raise RuntimeError(f"Failed to download dataset: {dataset_slug}. Make sure Kaggle API is configured.")

# Helper to safely read a CSV (or TSV) by trying common delimiters
def read_csv_any(filepath: Path):
    """Read a CSV/TSV file into a list of dicts. Tries to auto-detect delimiter."""
    # Try comma, then tab
    try:
        with filepath.open('r', encoding='utf-8', errors='ignore') as f:
            dialect = csv.Sniffer().sniff(f.read(2048))
            f.seek(0)
            reader = csv.DictReader(f, dialect=dialect)
            rows = [row for row in reader]
            return rows
    except Exception as e:
        # Default to comma
        with filepath.open('r', encoding='utf-8', errors='ignore') as f:
            reader = csv.DictReader(f)
            rows = [row for row in reader]
            return rows

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Prepare mental health instruction dataset")
parser.add_argument('--output', type=str, default='mental_health_instructions.txt',
                    help="Output text file for Post/Task/Answer data")
parser.add_argument('--dsm-file', type=str, default='DSM-5.txt',
                    help="Path to DSM-5 knowledge base text file")
parser.add_argument('--include-dsm', action='store_true',
                    help="Include DSM-5 knowledge-based Q&A in the output")
parser.add_argument('--emit-bin', action='store_true',
                    help="Also emit a tokenized .bin file for llm.c training")
parser.add_argument('--seed', type=int, default=42,
                    help="Random seed for shuffling")
parser.add_argument('--no-shuffle', action='store_true',
                    help="Disable shuffling of output examples")
args = parser.parse_args()

output_path = Path(args.output)
data_cache_dir = Path("data_cache")
data_cache_dir.mkdir(exist_ok=True)

all_lines = []  # collect all output lines for shuffling and writing

### 1. Mental Disorders Identification (Reddit) by Kamarul Adha
kamarul_slug = "kamaruladha/mental-disorders-identification-reddit-nlp"
kamarul_dir = data_cache_dir / "mental_disorders_identification"
download_kaggle_dataset(kamarul_slug, kamarul_dir)
# The dataset might be a single CSV file in the directory (since we unzipped it).
# Find the CSV file (assuming one main CSV).
kamarul_csv_files = list(kamarul_dir.glob("*.csv"))
if not kamarul_csv_files:
    raise FileNotFoundError(f"No CSV found for {kamarul_slug}")
# We assume the main CSV is the first one (there should be one).
kamarul_csv = kamarul_csv_files[0]
print(f"[+] Processing KamarulAdha dataset: {kamarul_csv.name}")
# Read in chunks if extremely large to conserve memory:
kamarul_records = read_csv_any(kamarul_csv)
for row in kamarul_records:
    # Possible columns: let's guess the relevant ones
    # We try common possibilities:
    text = ""
    if 'text' in row and row['text']:
        text = row['text']
    elif 'body' in row and row['body']:
        text = row['body']
    elif 'post' in row and row['post']:
        text = row['post']
    # If there's a title separate, include it:
    if 'title' in row and row['title'] and row['title'] not in text:
        # If title is not already part of text and not empty, prepend it
        title = row['title']
        # Avoid duplicating title if text already contains it (just in case)
        if title.strip() and title not in text:
            text = f"{title}\n{text}"
    text = text.strip()
    if not text:
        continue  # skip empty posts
    # Determine label category
    label = None
    # The dataset might not have an explicit label column apart from subreddit.
    if 'subreddit' in row and row['subreddit']:
        label = row['subreddit'].strip()
    elif 'label' in row and row['label']:
        label = row['label'].strip()
    # Normalize label (e.g., capitalize or replace abbreviations)
    if label:
        # example: use proper naming if needed (optional: e.g. "adhd" -> "ADHD")
        if label.lower() == "adhd":
            label = "ADHD"
        elif label.lower() == "ptsd":
            label = "PTSD"
        elif label.lower() == "ocd":
            label = "OCD"
        # etc., ensure consistent formatting (capitalized words)
        label = label.capitalize() if label.islower() else label
    else:
        # If no label found (shouldn't happen for this dataset), skip
        continue
    # Formulate Post/Task/Answer
    post_text = text.replace('\n', ' ').strip()
    task_text = "Identify which mental health condition this post is about."
    answer_text = label
    example = f"Post: {post_text}\nTask: {task_text}\nAnswer: {answer_text}<|endoftext|>"
    all_lines.append(example)

### 2. Mental Health Dataset (Bipolar) by Michelle VP
bipolar_slug = "michellevp/mental-health-dataset-bipolar"
bipolar_dir = data_cache_dir / "mental_health_bipolar"
download_kaggle_dataset(bipolar_slug, bipolar_dir)
# Find main file (could be CSV or XLSX)
bipolar_files = list(bipolar_dir.glob("*"))
bipolar_csv = None
for f in bipolar_files:
    if f.suffix.lower() in ['.csv', '.xlsx']:
        bipolar_csv = f
        break
if bipolar_csv is None:
    raise FileNotFoundError(f"No CSV/XLSX found for {bipolar_slug}")
print(f"[+] Processing Bipolar dataset: {bipolar_csv.name}")
bipolar_records = []
if bipolar_csv.suffix.lower() == '.csv':
    bipolar_records = read_csv_any(bipolar_csv)
else:
    # If it's an Excel file, use pandas to read it
    try:
        import pandas as pd
        bipolar_records = pd.read_excel(bipolar_csv).to_dict(orient='records')
    except ImportError:
        raise RuntimeError("Pandas is required to read Excel file for Bipolar dataset.")
# Iterate records
for row in bipolar_records:
    # Assume columns: e.g. "post_text" or "text", and "sentiment" possibly.
    text = ""
    if 'text' in row:
        text = str(row['text'])
    elif 'post' in row:
        text = str(row['post'])
    elif 'content' in row:
        text = str(row['content'])
    text = text.strip()
    if not text:
        continue
    # Sentiment label might be a column (e.g. "sentiment") or possibly derived.
    sentiment = None
    if 'sentiment' in row:
        sentiment = str(row['sentiment']).strip()
    elif 'sentiment_label' in row:
        sentiment = str(row['sentiment_label']).strip()
    # Normalize sentiment to one of Positive/Negative/Neutral if possible
    if sentiment:
        s_lower = sentiment.lower()
        if 'pos' in s_lower or s_lower == '1':
            sentiment = "Positive"
        elif 'neg' in s_lower or s_lower == '-1':
            sentiment = "Negative"
        elif 'neu' in s_lower or s_lower == '0':
            sentiment = "Neutral"
        else:
            # If already full word or something unexpected, just capitalize first letter
            sentiment = sentiment.capitalize()
    # If sentiment not explicitly given, we skip (or could infer via text analysis, but not here)
    if not sentiment:
        continue
    post_text = text.replace('\n', ' ').strip()
    task_text = "Determine the sentiment expressed in the following bipolar discussion post."
    answer_text = sentiment
    example = f"Post: {post_text}\nTask: {task_text}\nAnswer: {answer_text}<|endoftext|>"
    all_lines.append(example)
    # (Optional: if risk factor tags exist, generate extra tasks. Assuming column 'risk_factor' or similar.)
    if 'risk_factor' in row and row['risk_factor']:
        factor = str(row['risk_factor']).strip()
        if factor:
            task_text_rf = "Identify any risk factor mentioned in the post."
            answer_text_rf = factor
            example_rf = f"Post: {post_text}\nTask: {task_text_rf}\nAnswer: {answer_text_rf}<|endoftext|>"
            all_lines.append(example_rf)

### 3. Mental Health Data (Anxiety) by Michelle VP
anxiety_slug = "michellevp/predicting-anxiety-in-mental-health-data"
anxiety_dir = data_cache_dir / "mental_health_anxiety"
download_kaggle_dataset(anxiety_slug, anxiety_dir)
anxiety_files = list(anxiety_dir.glob("*"))
anxiety_csv = None
for f in anxiety_files:
    if f.suffix.lower() in ['.csv', '.xlsx']:
        anxiety_csv = f
        break
if anxiety_csv is None:
    raise FileNotFoundError(f"No CSV/XLSX found for {anxiety_slug}")
print(f"[+] Processing Anxiety dataset: {anxiety_csv.name}")
anxiety_records = []
if anxiety_csv.suffix.lower() == '.csv':
    anxiety_records = read_csv_any(anxiety_csv)
else:
    try:
        import pandas as pd
        anxiety_records = pd.read_excel(anxiety_csv).to_dict(orient='records')
    except ImportError:
        raise RuntimeError("Pandas is required to read Excel file for Anxiety dataset.")
for row in anxiety_records:
    text = ""
    if 'text' in row:
        text = str(row['text'])
    elif 'post' in row:
        text = str(row['post'])
    elif 'content' in row:
        text = str(row['content'])
    text = text.strip()
    if not text:
        continue
    # Label for anxiety presence (binary). Check likely columns:
    label_val = None
    if 'anxiety' in row:
        # could be a binary flag like 1/0 or True/False
        label_val = str(row['anxiety']).strip()
    elif 'label' in row:
        label_val = str(row['label']).strip()
    elif 'class' in row:
        label_val = str(row['class']).strip()
    # Interpret the label (assuming 1 = anxious, 0 = not)
    if label_val is None or label_val == "":
        continue
    anxiety_flag = None
    label_val_lower = label_val.lower()
    if label_val_lower in ['1', 'true', 'yes', 'anxiety']:
        anxiety_flag = "Yes"
    elif label_val_lower in ['0', 'false', 'no', 'none']:
        anxiety_flag = "No"
    else:
        # If it's already something like 'anxiety' vs 'other', map accordingly:
        if label_val_lower == 'anxiety':
            anxiety_flag = "Yes"
        else:
            anxiety_flag = "No"
    post_text = text.replace('\n', ' ').strip()
    task_text = "Does the following post indicate an anxiety disorder? Answer Yes or No."
    answer_text = anxiety_flag
    example = f"Post: {post_text}\nTask: {task_text}\nAnswer: {answer_text}<|endoftext|>"
    all_lines.append(example)

### 4. Stress Analysis (Dreaddit) by Shuvojit Das
stress_slug = "shuvojitdas/stress-analysis"
stress_dir = data_cache_dir / "stress_analysis"
download_kaggle_dataset(stress_slug, stress_dir)
# Dreaddit dataset likely has two CSV files: training and test.
stress_csv_files = list(stress_dir.glob("*.csv"))
if not stress_csv_files:
    # Some Kaggle versions might have tsv, check any
    stress_csv_files = list(stress_dir.glob("*.tsv"))
if not stress_csv_files:
    raise FileNotFoundError(f"No CSV/TSV found for {stress_slug}")
print(f"[+] Processing Stress (Dreaddit) dataset...")
for f in stress_csv_files:
    records = read_csv_any(f)
    for row in records:
        text = ""
        if 'text' in row:
            text = row['text']
        elif 'post' in row:
            text = row['post']
        elif 'body' in row:
            text = row['body']
        if not text:
            continue
        text = str(text).strip()
        label = None
        if 'label' in row:
            label = row['label']
        elif 'stress' in row:
            label = row['stress']
        elif 'y' in row:
            label = row['y']
        if label is None:
            continue
        # Label might be '0'/'1' or 'no'/'yes'
        lab_str = str(label).strip().lower()
        if lab_str in ['1', 'yes', 'true', 'stressed']:
            answer = "Yes"
        else:
            answer = "No"
        post_text = text.replace('\n', ' ').strip()
        task_text = "Determine if the user is stressed in the following post. Answer Yes or No."
        answer_text = answer
        example = f"Post: {post_text}\nTask: {task_text}\nAnswer: {answer_text}<|endoftext|>"
        all_lines.append(example)

### 5. Reddit Mental Health Data by Neel Ghoshal
neel_slug = "neelghoshal/reddit-mental-health-data"
neel_dir = data_cache_dir / "reddit_mental_health_data"
download_kaggle_dataset(neel_slug, neel_dir)
neel_files = list(neel_dir.glob("*"))
neel_csv = None
for f in neel_files:
    if f.suffix.lower() in ['.csv', '.tsv']:
        neel_csv = f
        break
if neel_csv is None:
    raise FileNotFoundError(f"No CSV/TSV found for {neel_slug}")
print(f"[+] Processing NeelGhoshal Reddit Mental Health dataset: {neel_csv.name}")
neel_records = read_csv_any(neel_csv)
# According to description, target mapping: 0=Stress, 1=Depression, 2=Bipolar, 3=Anxiety, 4=PTSD (likely).
# We'll map numeric labels to names:
label_map = {
    '0': "Stress",
    '1': "Depression",
    '2': "Bipolar",
    '3': "Anxiety",
    '4': "PTSD"
}
for row in neel_records:
    text = ""
    # likely column for text
    if 'text' in row:
        text = row['text']
    elif 'post' in row:
        text = row['post']
    elif 'body' in row:
        text = row['body']
    if not text:
        continue
    text = str(text).strip()
    label_val = None
    if 'target' in row:
        label_val = str(row['target']).strip()
    elif 'label' in row:
        label_val = str(row['label']).strip()
    elif 'class' in row:
        label_val = str(row['class']).strip()
    if label_val is None or label_val == '':
        continue
    # Use mapping if numeric
    label_name = label_map.get(label_val, None)
    if label_name is None:
        # If label_val is already a name, use it directly
        label_name = label_val.capitalize()
    post_text = text.replace('\n', ' ').strip()
    task_text = "Identify the mental health issue discussed in this post (Stress, Depression, Bipolar, Anxiety, or PTSD)."
    answer_text = label_name
    example = f"Post: {post_text}\nTask: {task_text}\nAnswer: {answer_text}<|endoftext|>"
    all_lines.append(example)

### 6. Social Anxiety Dataset by Nate Zhang
sad_slug = "natezhang123/social-anxiety-dataset"
sad_dir = data_cache_dir / "social_anxiety_dataset"
download_kaggle_dataset(sad_slug, sad_dir)
sad_files = list(sad_dir.glob("*"))
sad_csv = None
for f in sad_files:
    if f.suffix.lower() in ['.csv', '.xlsx']:
        sad_csv = f
        break
if sad_csv is None:
    raise FileNotFoundError(f"No CSV/XLSX found for {sad_slug}")
print(f"[+] Processing Social Anxiety dataset: {sad_csv.name}")
sad_records = []
if sad_csv.suffix.lower() == '.csv':
    sad_records = read_csv_any(sad_csv)
else:
    try:
        import pandas as pd
        sad_records = pd.read_excel(sad_csv).to_dict(orient='records')
    except ImportError:
        raise RuntimeError("Pandas is required to read Excel file for Social Anxiety dataset.")
# This dataset likely has multiple feature columns (survey answers) and one label column for severity.
# We attempt to identify the label column (e.g. "severity" or "level") and some key features.
label_col = None
# Find label column by keywords:
for key in sad_records[0].keys():
    if key.lower() in ['severity', 'level', 'anxiety_level', 'class', 'label']:
        label_col = key
        break
if label_col is None:
    raise RuntimeError("Could not find severity label in Social Anxiety dataset.")
# Choose a few features to summarize (to avoid listing all 30 question responses, pick representative ones if present)
feature_keys = [k for k in sad_records[0].keys() if k != label_col]
# We can pick a subset of features (e.g., those containing specific keywords like 'social', 'anxiety', etc. for brevity)
chosen_feats = []
for fk in feature_keys:
    lk = fk.lower()
    if 'anxiety' in lk or 'social' in lk or 'avoid' in lk or 'fear' in lk:
        chosen_feats.append(fk)
    if len(chosen_feats) >= 3:
        break
# If we didn't find any specific, just choose first 3 questions
if not chosen_feats:
    chosen_feats = feature_keys[:3]
for row in sad_records:
    # Determine severity label
    severity = str(row[label_col]).strip()
    if severity.isdigit():
        # Map numeric levels to text
        # Assuming e.g. 0-3: 0=None, 1=Mild, 2=Moderate, 3=Severe
        level = int(severity)
        if level <= 0:
            severity_text = "None"
        elif level == 1:
            severity_text = "Mild"
        elif level == 2:
            severity_text = "Moderate"
        else:
            severity_text = "Severe"
    else:
        severity_text = severity.capitalize()
    # Construct a brief description from chosen features
    descriptions = []
    for fk in chosen_feats:
        if fk in row:
            val = str(row[fk]).strip()
            if val.isdigit():
                # interpret numeric answer on a scale (just include value)
                descriptions.append(f"{fk}: {val}")
            else:
                descriptions.append(f"{fk}: {val}")
    if not descriptions:
        # If no feature info, skip
        continue
    desc_text = "; ".join(descriptions)
    post_text = f"The individual reports: {desc_text}."
    task_text = "Classify the person's social anxiety level (None, Mild, Moderate, or Severe)."
    answer_text = severity_text
    example = f"Post: {post_text}\nTask: {task_text}\nAnswer: {answer_text}<|endoftext|>"
    all_lines.append(example)

### 7. Reddit Mental Health Dataset (RMHD) by Entenam
entenam_slug = "entenam/reddit-mental-health-dataset"
entenam_dir = data_cache_dir / "reddit_mental_health_dataset"
download_kaggle_dataset(entenam_slug, entenam_dir)
entenam_files = list(entenam_dir.glob("*"))
entenam_csv = None
for f in entenam_files:
    if f.suffix.lower() in ['.csv', '.tsv', '.xlsx', '.json']:
        entenam_csv = f
        break
if entenam_csv is None:
    raise FileNotFoundError(f"No data file found for {entenam_slug}")
print(f"[+] Processing Entenam RMHD dataset: {entenam_csv.name}")
entenam_records = []
if entenam_csv.suffix.lower() == '.csv' or entenam_csv.suffix.lower() == '.tsv':
    entenam_records = read_csv_any(entenam_csv)
elif entenam_csv.suffix.lower() == '.xlsx':
    import pandas as pd
    entenam_records = pd.read_excel(entenam_csv).to_dict(orient='records')
elif entenam_csv.suffix.lower() == '.json':
    import json
    with open(entenam_csv, 'r', encoding='utf-8') as f:
        entenam_records = json.load(f)
# We expect fields like 'text', 'subreddit' (or 'condition'), and possibly 'cause' or similar annotation.
text_field = None
for key in entenam_records[0].keys():
    if key.lower() in ['text', 'post', 'content', 'body']:
        text_field = key; break
if text_field is None:
    raise RuntimeError("No text field found in Entenam dataset.")
sub_field = None
cause_field = None
for key in entenam_records[0].keys():
    lk = key.lower()
    if lk in ['subreddit', 'condition', 'category', 'label']:
        sub_field = key
    if lk in ['cause', 'causes', 'trigger', 'issue']:
        cause_field = key
# Iterate
for row in entenam_records:
    text = str(row[text_field]).strip()
    if not text:
        continue
    # Disorder category from subreddit/condition
    disorder = None
    if sub_field and row.get(sub_field):
        disorder = str(row[sub_field]).strip()
        # Normalize format (capitalize etc.)
        disorder = disorder.capitalize() if disorder.islower() else disorder
    # Cause annotation
    cause = None
    if cause_field and row.get(cause_field):
        cause = str(row[cause_field]).strip()
        cause = cause.capitalize()  # simple normalization
    post_text = text.replace('\n', ' ').strip()
    # Always generate a disorder classification task
    if disorder:
        task_text = "Identify which mental health condition this post is about."
        answer_text = disorder
        example = f"Post: {post_text}\nTask: {task_text}\nAnswer: {answer_text}<|endoftext|>"
        all_lines.append(example)
    # If cause is annotated, generate a cause-identification task
    if cause:
        task_text_c = "What appears to be the primary cause of distress in this post?"
        answer_text_c = cause
        example_c = f"Post: {post_text}\nTask: {task_text_c}\nAnswer: {answer_text_c}<|endoftext|>"
        all_lines.append(example_c)

### 8. Mental Disorder Classification (Symptoms) by CID007
symptom_slug = "cid007/mental-disorder-classification"
symptom_dir = data_cache_dir / "mental_disorder_classification"
download_kaggle_dataset(symptom_slug, symptom_dir)
symptom_files = list(symptom_dir.glob("*"))
symptom_csv = None
for f in symptom_files:
    if f.suffix.lower() in ['.csv', '.tsv', '.xlsx']:
        symptom_csv = f
        break
if symptom_csv is None:
    raise FileNotFoundError(f"No data file found for {symptom_slug}")
print(f"[+] Processing Symptom-based dataset: {symptom_csv.name}")
symptom_records = []
if symptom_csv.suffix.lower() == '.csv' or symptom_csv.suffix.lower() == '.tsv':
    symptom_records = read_csv_any(symptom_csv)
elif symptom_csv.suffix.lower() == '.xlsx':
    import pandas as pd
    symptom_records = pd.read_excel(symptom_csv).to_dict(orient='records')
# Identify symptom columns (likely 17 of them) and label column
symptom_cols = []
label_col = None
if symptom_records:
    for key in symptom_records[0].keys():
        lk = key.lower()
        if lk in ['disorder', 'condition', 'diagnosis', 'label', 'class']:
            label_col = key
        else:
            symptom_cols.append(key)
# Ensure we have symptom columns and label
if label_col is None or not symptom_cols:
    raise RuntimeError("Unexpected format in symptom dataset.")
for row in symptom_records:
    diagnosis = str(row[label_col]).strip()
    if not diagnosis:
        continue
    # Normalize diagnosis text (capitalize each word)
    diagnosis_text = diagnosis
    # Example: if "Normal" is one category
    if diagnosis_text.lower() == 'normal':
        diagnosis_text = "Normal"
    else:
        diagnosis_text = diagnosis_text.title()
    # Construct a brief symptom list description.
    # We will list a subset of symptoms that are "present" or have high severity.
    symptom_desc = []
    for sym in symptom_cols:
        val = str(row[sym]).strip()
        if val == '' or val.lower() == 'nan':
            continue
        # If value is numeric (e.g. severity 0-3), include if moderate or severe:
        included = False
        try:
            num = float(val)
            if num >= 1:  # symptom present (assuming 1 means present, or severity >0)
                included = True
        except:
            # If value is non-numeric, include if it's not indicating absence
            if val.lower() not in ['no', 'false', 'none', '0']:
                included = True
        if included:
            # Use symptom name (we assume sym is already a descriptive name from dataset)
            # Some symptom column names might be abbreviated; we just use them as is or make them more readable.
            sym_name = sym
            # Replace underscores with spaces and capitalize
            sym_name = sym_name.replace('_', ' ').strip().capitalize()
            symptom_desc.append(sym_name)
    if not symptom_desc:
        # If no symptom flagged, still list one or two with low severity if needed (or skip if normal)
        if diagnosis_text == "Normal":
            symptom_desc.append("no significant psychiatric symptoms")
        else:
            # If diagnosis but no symptom flagged (unlikely), skip
            continue
    # Join a few symptoms for text
    symptoms_text = ", ".join(symptom_desc[:5])
    post_text = f"Patient exhibits the following symptoms: {symptoms_text}."
    task_text = "Determine the most likely diagnosis for this patient."
    answer_text = diagnosis_text
    example = f"Post: {post_text}\nTask: {task_text}\nAnswer: {answer_text}<|endoftext|>"
    all_lines.append(example)

### 9. Panic Disorder Detection by Muhammad Shahid Azeem
panic_slug = "muhammadshahidazeem/panic-disorder-detection-dataset"
panic_dir = data_cache_dir / "panic_disorder_detection"
download_kaggle_dataset(panic_slug, panic_dir)
panic_files = list(panic_dir.glob("*"))
# Expect two CSVs (training and testing)
panic_csvs = [f for f in panic_files if f.suffix.lower() == '.csv' or f.suffix.lower() == '.tsv']
if not panic_csvs:
    raise FileNotFoundError(f"No CSV files found for {panic_slug}")
print(f"[+] Processing Panic Disorder dataset...")
for csv_file in panic_csvs:
    records = read_csv_any(csv_file)
    # Identify potential feature columns and label column.
    # Likely there's a column like 'Panic_Disorder' as label (0/1).
    label_key = None
    # Find label column by looking for obvious keywords
    for key in records[0].keys():
        lk = key.lower()
        if 'panic' in lk and ('label' in lk or 'disorder' in lk or 'target' in lk):
            label_key = key
            break
    if label_key is None:
        # If not found, assume last column is label
        label_key = list(records[0].keys())[-1]
    feature_keys = [k for k in records[0].keys() if k != label_key]
    for row in records:
        label_val = str(row[label_key]).strip().lower()
        if label_val in ['1', 'yes', 'true', 'panic']:
            diagnosis = "Yes"
        else:
            diagnosis = "No"
        # Create a brief case description from a few features (for readability, use only some key features)
        desc_parts = []
        for fk in feature_keys:
            val = str(row[fk]).strip()
            if val == '' or val.lower() == 'nan':
                continue
            # For privacy, skip any personal identifiable info if present (unlikely in this dataset).
            # We'll include only a few medical or symptom features.
            if fk.lower() in ['age', 'gender', 'sex']:
                desc_parts.append(f"{fk}: {val}")
            elif fk.lower() in ['symptom', 'symptoms', 'chest_pain', 'sweating', 'palpitations', 'dizziness']:
                # if feature name suggests a symptom or sign, include if value indicates presence
                # Many features might be binary (0/1 for symptom presence)
                if val in ['1', 'yes', 'true']:
                    desc_parts.append(f"{fk.replace('_', ' ')}: yes")
            # Limit number of features in description to keep it short
            if len(desc_parts) >= 5:
                break
        if not desc_parts:
            # If no features chosen (e.g. if mostly zeros), just say "No significant symptoms reported" or similar
            desc_parts.append("no notable symptoms")
        case_desc = "; ".join(desc_parts)
        post_text = f"Patient data - {case_desc}."
        task_text = "Is this a case of Panic Disorder? Answer Yes or No."
        answer_text = diagnosis
        example = f"Post: {post_text}\nTask: {task_text}\nAnswer: {answer_text}<|endoftext|>"
        all_lines.append(example)

### DSM-5 Knowledge Base Injection (Optional)
if args.include_dsm:
    # Read DSM-5 text and generate Q&A tasks for knowledge injection
    if os.path.isfile(args.dsm_file):
        print("[+] Injecting DSM-5 knowledge from", args.dsm_file)
        with open(args.dsm_file, 'r', encoding='utf-8') as dsmf:
            dsm_text = dsmf.read()
        # Assuming DSM-5.txt contains sections headed by disorder names followed by criteria text.
        # We will create questions for each disorder's criteria.
        # Simple approach: find lines that look like disorder titles (e.g., all caps or followed by diagnostic criteria) and use them.
        lines = dsm_text.splitlines()
        disorder = None
        criteria_accum = []
        for line in lines:
            # Identify disorder name lines (for example, lines in all caps or followed by a period maybe are criteria text).
            if line.isupper() and line.strip() != "" and len(line.split()) < 10:
                # Treat this as a disorder heading
                # If we had a previous disorder collected, generate a QA pair for it
                if disorder and criteria_accum:
                    criteria_text = " ".join(criteria_accum).strip()
                    q_task = f"List the DSM-5 diagnostic criteria for {disorder}."
                    a_text = criteria_text
                    example = f"Post: (DSM-5 Reference)\nTask: {q_task}\nAnswer: {a_text}<|endoftext|>"
                    all_lines.append(example)
                # Start new disorder section
                disorder = line.strip().title()
                criteria_accum = []
            else:
                # accumulate criteria text
                if disorder:
                    criteria_accum.append(line.strip())
        # Handle last disorder in file
        if disorder and criteria_accum:
            criteria_text = " ".join(criteria_accum).strip()
            q_task = f"List the DSM-5 diagnostic criteria for {disorder}."
            a_text = criteria_text
            example = f"Post: (DSM-5 Reference)\nTask: {q_task}\nAnswer: {a_text}<|endoftext|>"
            all_lines.append(example)
    else:
        print(f"[!] DSM-5 file not found at {args.dsm_file}; skipping DSM injection.")
else:
    print("[*] DSM-5 knowledge base injection is disabled.")

# Shuffle all examples unless disabled
random.seed(args.seed)
if not args.no_shuffle:
    random.shuffle(all_lines)

# Write out to the output file
with open(output_path, 'w', encoding='utf-8') as out_f:
    for line in all_lines:
        out_f.write(line + "\n")

print(f"[+] Wrote {len(all_lines)} examples to {output_path}")

# If requested, also emit a tokenized .bin version for llm.c
if args.emit_bin:
    if GPT2TokenizerFast is None:
        raise RuntimeError("GPT2TokenizerFast not available for binary encoding.")
    print("[+] Tokenizing output to binary format...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    # Read all text as one string for tokenization
    with open(output_path, 'r', encoding='utf-8') as f:
        data_text = f.read()
    tokens = tokenizer.encode(data_text)
    # Use 16-bit unsigned integers for token IDs (GPT-2 vocab fits in 65535)
    arr = np.array(tokens, dtype=np.uint16)
    bin_path = output_path.with_suffix('.bin')
    arr.tofile(bin_path)
    print(f"[+] Tokenized binary saved to {bin_path}")
