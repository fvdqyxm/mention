import os
import re
import argparse
from collections import defaultdict
from datetime import datetime

import pandas as pd

# --- date parsing (same logic as before) ---
DATE_RE = re.compile(r"(\d{1,2})[-/](\d{1,2})[-/](\d{2,4})")

def parse_date_from_filename(fn):
    m = DATE_RE.search(fn)
    if not m:
        return None
    mm, dd, yy = map(int, m.groups())
    if yy < 100:
        yy += 2000
    try:
        return datetime(yy, mm, dd)
    except:
        return None

TOKEN_RE = re.compile(r"[a-z']+")

def tokenize(text):
    text = text.lower()
    text = re.sub(r"\(\d{1,2}:\d{2}(:\d{2})?\):", " ", text)
    return set(TOKEN_RE.findall(text))  # SET = presence only


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--txt_dir", required=True)
    ap.add_argument("--out", default="speech_word_presence.csv")
    args = ap.parse_args()

    records = []

    for fn in os.listdir(args.txt_dir):
        if not fn.endswith(".txt"):
            continue

        dt = parse_date_from_filename(fn)
        if not dt:
            continue

        with open(os.path.join(args.txt_dir, fn), "r", encoding="utf-8") as f:
            text = f.read()

        words_present = tokenize(text)
        month = f"{dt.year:04d}-{dt.month:02d}"

        records.append({
            "file": fn,
            "month": month,
            "words": words_present
        })

    df = pd.DataFrame(records)

    # Build presence probabilities
    months = sorted(df.month.unique())
    vocab = sorted(set.union(*df.words))

    rows = []
    for month in months:
        sub = df[df.month == month]
        n = len(sub)

        for w in vocab:
            count = sum(1 for ws in sub.words if w in ws)
            prob = count / n
            rows.append((month, w, prob))

    out_df = pd.DataFrame(rows, columns=["month", "word", "speech_prob"])
    out_df.to_csv(args.out, index=False)

    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

