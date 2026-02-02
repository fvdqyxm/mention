#!/usr/bin/env python3
"""
BEST + EFFICIENT Trump transcript analyzer (Python 3.9+)

Outputs:
- analysis_out/unigram_probs_by_month.csv      -> P(word | month)
- analysis_out/bigram_probs_by_month.parquet   -> P(w2 | w1, month) (top-K per prefix)
- analysis_out/word_clusters.json              -> word -> cluster_id
- analysis_out/cluster_top_words.json          -> cluster_id -> top words
- analysis_out/vocab.json                      -> vocab + counts

Designed for 1000+ transcripts:
- counts are streaming-ish and cheap
- clustering uses sparse co-occurrence + TruncatedSVD + MiniBatchKMeans

Usage:
  python3 best_trump_lm_and_clusters.py --txt_dir out/extracted_txt --out analysis_out

Recommended for speed:
  python3 best_trump_lm_and_clusters.py --txt_dir out/extracted_txt --out analysis_out \
    --vocab 20000 --co_vocab 6000 --window 6 --svd_dim 128 --clusters 16 --topk_bigrams 50
"""

import os
import re
import json
import math
import argparse
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import MiniBatchKMeans


# ----------------------------
# Date parsing
# ----------------------------
DATE_PATTERNS = [
    # 1-27-26 or 01-27-2026
    re.compile(r"(?P<m>\d{1,2})[-/](?P<d>\d{1,2})[-/](?P<y>\d{2,4})"),
    # January 27, 2026
    re.compile(
        r"(?P<month>january|february|march|april|may|june|july|august|september|october|november|december)\s+"
        r"(?P<d>\d{1,2}),\s+(?P<y>\d{4})",
        re.IGNORECASE
    ),
]

MONTH_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12
}

def normalize_year(y: int) -> int:
    return 2000 + y if y < 100 else y

def parse_date_from_filename(fname: str) -> Optional[datetime]:
    base = os.path.basename(fname)
    for pat in DATE_PATTERNS:
        m = pat.search(base)
        if not m:
            continue
        gd = m.groupdict()
        if "m" in gd and gd.get("m"):
            mm = int(gd["m"]); dd = int(gd["d"]); yy = normalize_year(int(gd["y"]))
            try:
                return datetime(yy, mm, dd)
            except ValueError:
                return None
        if gd.get("month"):
            mm = MONTH_MAP[gd["month"].lower()]
            dd = int(gd["d"]); yy = int(gd["y"])
            try:
                return datetime(yy, mm, dd)
            except ValueError:
                return None
    return None

def parse_date_from_text(text: str) -> Optional[datetime]:
    head = text[:2500]
    for pat in DATE_PATTERNS:
        m = pat.search(head)
        if not m:
            continue
        gd = m.groupdict()
        if "m" in gd and gd.get("m"):
            mm = int(gd["m"]); dd = int(gd["d"]); yy = normalize_year(int(gd["y"]))
            try:
                return datetime(yy, mm, dd)
            except ValueError:
                return None
        if gd.get("month"):
            mm = MONTH_MAP[gd["month"].lower()]
            dd = int(gd["d"]); yy = int(gd["y"])
            try:
                return datetime(yy, mm, dd)
            except ValueError:
                return None
    return None

def month_bucket(dt: datetime) -> str:
    return f"{dt.year:04d}-{dt.month:02d}"


# ----------------------------
# Tokenization
# ----------------------------
TOKEN_RE = re.compile(r"[a-z']+")

def tokenize(text: str) -> List[str]:
    text = text.lower()
    # remove timestamps like "(00:12):"
    text = re.sub(r"\(\s*\d{1,2}:\d{2}(:\d{2})?\s*\)\s*:\s*", " ", text)
    toks = TOKEN_RE.findall(text)
    toks = [t for t in toks if len(t) >= 2 and t not in ENGLISH_STOP_WORDS]
    return toks


# ----------------------------
# Efficient PPMI on sparse co-occurrence
# ----------------------------
def build_sparse_cooccurrence(
    docs_tokens: List[List[str]],
    vocab: List[str],
    window: int
) -> sparse.csr_matrix:
    """
    Build sparse symmetric co-occurrence counts matrix for vocab words.

    Complexity ~ O(total_tokens * window) but only for top co_vocab (e.g., 6000).
    """
    idx = {w:i for i,w in enumerate(vocab)}
    rows = []
    cols = []
    data = []

    for toks in docs_tokens:
        ids = [idx.get(t) for t in toks]
        n = len(ids)
        for i in range(n):
            wi = ids[i]
            if wi is None:
                continue
            lo = max(0, i - window)
            hi = min(n, i + window + 1)
            for j in range(lo, hi):
                if j == i:
                    continue
                wj = ids[j]
                if wj is None:
                    continue
                # store directed; we'll symmetrize later
                rows.append(wi); cols.append(wj); data.append(1.0)

    M = sparse.coo_matrix((data, (rows, cols)), shape=(len(vocab), len(vocab)), dtype=np.float32).tocsr()
    # symmetrize and zero diagonal
    M = (M + M.T).tocsr()
    M.setdiag(0)
    M.eliminate_zeros()
    return M


def sparse_ppmi(M: sparse.csr_matrix, eps: float = 1e-12) -> sparse.csr_matrix:
    """
    Compute PPMI from sparse co-occurrence matrix.

    PPMI(i,j) = max(log( P(i,j) / (P(i)P(j)) ), 0)
    """
    M = M.tocsr()
    total = float(M.sum()) + eps
    row_sum = np.array(M.sum(axis=1)).ravel() + eps
    col_sum = np.array(M.sum(axis=0)).ravel() + eps

    # We'll transform non-zeros in COO form for speed
    Mcoo = M.tocoo()
    pij = Mcoo.data / total
    pi = row_sum[Mcoo.row] / total
    pj = col_sum[Mcoo.col] / total
    pmi = np.log((pij + eps) / (pi * pj + eps))
    ppmi = np.maximum(pmi, 0.0)

    X = sparse.coo_matrix((ppmi.astype(np.float32), (Mcoo.row, Mcoo.col)), shape=M.shape).tocsr()
    X.eliminate_zeros()
    return X


# ----------------------------
# Probability models
# ----------------------------
def smooth_prob(count: int, total: int, vocab_size: int, alpha: float) -> float:
    return (count + alpha) / (total + alpha * vocab_size)

def build_unigram_models(
    docs_by_month: Dict[str, List[List[str]]],
    vocab: List[str],
    alpha: float
) -> pd.DataFrame:
    """
    Return long-form dataframe: month, word, prob
    """
    vocab_size = len(vocab)
    records = []
    for month, docs in docs_by_month.items():
        counts = Counter()
        total = 0
        for toks in docs:
            counts.update(toks)
            total += len(toks)
        for w in vocab:
            p = smooth_prob(counts.get(w, 0), total, vocab_size, alpha)
            records.append((month, w, p))
    return pd.DataFrame(records, columns=["month", "word", "prob"])


def build_bigram_models_topk(
    docs_by_month: Dict[str, List[List[str]]],
    vocab_set: set,
    alpha: float,
    topk: int
) -> pd.DataFrame:
    """
    Efficient bigram model by month, but store only top-k next words per prefix.
    Output columns: month, w1, w2, prob
    """
    out_rows = []

    for month, docs in docs_by_month.items():
        # counts[w1][w2] and totals[w1]
        next_counts: Dict[str, Counter] = defaultdict(Counter)
        totals: Counter = Counter()

        for toks in docs:
            # keep only vocab words
            toks = [t for t in toks if t in vocab_set]
            for i in range(len(toks) - 1):
                w1 = toks[i]; w2 = toks[i+1]
                next_counts[w1][w2] += 1
                totals[w1] += 1

        # For each w1, keep topk w2 by smoothed probability
        for w1, ctr in next_counts.items():
            total = totals[w1]
            # restrict candidates to observed + smoothing denominator uses |V|
            V = len(vocab_set)
            candidates = ctr.most_common(topk * 3)  # over-sample then compute probs
            scored = []
            for w2, c in candidates:
                p = (c + alpha) / (total + alpha * V)
                scored.append((w2, p))
            scored.sort(key=lambda x: x[1], reverse=True)
            for w2, p in scored[:topk]:
                out_rows.append((month, w1, w2, float(p)))

    return pd.DataFrame(out_rows, columns=["month", "w1", "w2", "prob"])


# ----------------------------
# Main
# ----------------------------
def load_docs(txt_dir: str) -> List[Tuple[str, datetime, str]]:
    docs = []
    for fn in sorted(os.listdir(txt_dir)):
        if not fn.endswith(".txt"):
            continue
        path = os.path.join(txt_dir, fn)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        dt = parse_date_from_filename(fn) or parse_date_from_text(text)
        if dt is None:
            # skip undated docs
            continue
        docs.append((fn, dt, text))
    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--txt_dir", required=True, help="Folder with extracted .txt transcripts")
    ap.add_argument("--out", default="analysis_out", help="Output folder")

    # Vocab choices
    ap.add_argument("--vocab", type=int, default=20000, help="Top-N vocab for probability models")
    ap.add_argument("--co_vocab", type=int, default=6000, help="Top-N vocab for co-occurrence/clustering")

    # Relationship settings
    ap.add_argument("--window", type=int, default=6, help="Co-occurrence window size")
    ap.add_argument("--svd_dim", type=int, default=128, help="SVD embedding dimension")
    ap.add_argument("--clusters", type=int, default=16, help="Number of word clusters")

    # LM settings
    ap.add_argument("--alpha", type=float, default=0.5, help="Smoothing alpha for probabilities")
    ap.add_argument("--topk_bigrams", type=int, default=50, help="Top-K next words stored per prefix per month")

    # Speed knobs
    ap.add_argument("--max_docs_for_cooccurrence", type=int, default=0,
                    help="If >0, use only most recent N docs for co-occurrence (faster). 0 = use all.")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    docs = load_docs(args.txt_dir)
    if not docs:
        raise SystemExit(f"No dated .txt transcripts found in {args.txt_dir}")

    docs.sort(key=lambda x: x[1])
    print(f"Loaded {len(docs)} transcripts from {docs[0][1].date()} to {docs[-1][1].date()}")

    # Tokenize
    tokenized = []
    months = []
    for fn, dt, text in docs:
        toks = tokenize(text)
        tokenized.append(toks)
        months.append(month_bucket(dt))

    # Build global word counts (fast)
    global_counts = Counter()
    for toks in tokenized:
        global_counts.update(toks)

    vocab = [w for w, _ in global_counts.most_common(args.vocab)]
    vocab_set = set(vocab)

    co_vocab = [w for w, _ in global_counts.most_common(args.co_vocab)]
    co_vocab_set = set(co_vocab)

    # Group docs by month (for time-aware probs)
    docs_by_month: Dict[str, List[List[str]]] = defaultdict(list)
    for m, toks in zip(months, tokenized):
        # keep vocab tokens for probability models
        docs_by_month[m].append([t for t in toks if t in vocab_set])

    # 1) Unigram probs by month (P(word|month))
    print("Computing unigram probabilities by month...")
    uni_df = build_unigram_models(docs_by_month, vocab=vocab, alpha=args.alpha)
    uni_out = os.path.join(out_dir, "unigram_probs_by_month.csv")
    uni_df.to_csv(uni_out, index=False)

    # 2) Bigram probs by month (top-k per prefix)
    print("Computing bigram probabilities by month (top-k)...")
    bi_df = build_bigram_models_topk(docs_by_month, vocab_set=vocab_set, alpha=args.alpha, topk=args.topk_bigrams)
    bi_out = os.path.join(out_dir, "bigram_probs_by_month.parquet")
    bi_df.to_parquet(bi_out, index=False)

    # Save vocab
    vocab_out = os.path.join(out_dir, "vocab.json")
    with open(vocab_out, "w", encoding="utf-8") as f:
        json.dump({
            "vocab_top_n": args.vocab,
            "co_vocab_top_n": args.co_vocab,
            "total_unique_tokens": len(global_counts),
            "top_words": [{"word": w, "count": int(global_counts[w])} for w in vocab[:500]]
        }, f, indent=2)

    # 3) Clustering with sparse PPMI + SVD + MiniBatchKMeans
    # Optionally limit co-occurrence to most recent docs for speed
    toks_for_co = tokenized
    if args.max_docs_for_cooccurrence and args.max_docs_for_cooccurrence > 0:
        toks_for_co = tokenized[-args.max_docs_for_cooccurrence:]
        print(f"Using only most recent {len(toks_for_co)} docs for co-occurrence (speed knob).")

    # Filter tokens to co_vocab for co-occurrence
    toks_for_co = [[t for t in toks if t in co_vocab_set] for toks in toks_for_co]

    print("Building sparse co-occurrence matrix...")
    M = build_sparse_cooccurrence(toks_for_co, vocab=co_vocab, window=args.window)
    print(f"Co-occurrence matrix: shape={M.shape}, nnz={M.nnz}")

    print("Computing sparse PPMI...")
    X = sparse_ppmi(M)
    print(f"PPMI matrix: nnz={X.nnz}")

    print("Reducing with TruncatedSVD...")
    svd = TruncatedSVD(n_components=args.svd_dim, random_state=0)
    emb = svd.fit_transform(X)  # dense (co_vocab x svd_dim), manageable for co_vocab ~ 6000

    print("Clustering with MiniBatchKMeans...")
    km = MiniBatchKMeans(n_clusters=args.clusters, random_state=0, batch_size=1024, n_init=10)
    labels = km.fit_predict(emb)

    word_to_cluster = {w: int(c) for w, c in zip(co_vocab, labels)}

    # Cluster summaries: top words by global count inside each cluster
    cluster_to_words: Dict[int, List[str]] = defaultdict(list)
    for w, c in word_to_cluster.items():
        cluster_to_words[c].append(w)
    for c in cluster_to_words:
        cluster_to_words[c].sort(key=lambda w: global_counts[w], reverse=True)

    clusters_out = os.path.join(out_dir, "word_clusters.json")
    with open(clusters_out, "w", encoding="utf-8") as f:
        json.dump(word_to_cluster, f, indent=2)

    cluster_top_out = os.path.join(out_dir, "cluster_top_words.json")
    with open(cluster_top_out, "w", encoding="utf-8") as f:
        json.dump({str(c): ws[:60] for c, ws in cluster_to_words.items()}, f, indent=2)

    print("\nDONE. Wrote:")
    print(f"  - {uni_out}")
    print(f"  - {bi_out}")
    print(f"  - {clusters_out}")
    print(f"  - {cluster_top_out}")
    print(f"  - {vocab_out}")

    print("\nHow to use outputs:")
    print("  Unigram: filter unigram_probs_by_month.csv for (month, word).")
    print("  Bigram:  filter bigram_probs_by_month.parquet for (month, w1='we', w2 candidates).")
    print("  Clusters: look up word in word_clusters.json, then view cluster_top_words.json.")


if __name__ == "__main__":
    main()

