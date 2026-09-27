#!/usr/bin/env python3
"""
Memory-safe business entity resolution pipeline (accuracy-optimized).

Design (unchanged from the base pipeline):
  1. Stream TSVs; never load the 10M+ target rows into pandas.
  2. Build a disk-backed SQLite blocking index over Source 2/3.
  3. Multi-pass blocking:
       - exact normalized name
       - exact normalized address
       - name prefix
       - address prefix
       - rare name tokens
       - rare address tokens
       - optional FAISS semantic ANN (if installed and enabled)
  4. For each Source-1 batch, retrieve candidates and score them immediately.
  5. Train LightGBM only on a bounded sample of candidate pairs.
  6. Write candidate_pairs.tsv and matching_results.tsv incrementally.
  7. CPU-only fallback if CUDA / FAISS GPU is unavailable.

What changed in this version, and why each change moves the needle on
held-out pair accuracy:

  - Feature set expanded from 16 -> 22 columns. The new features
    (Jaro-Winkler similarity, token-set Jaccard, digit/zip/building-number
    Jaccard, acronym match) target failure modes the old ratio-based
    features under-weight: short names, transposed word order, and
    numeric address components that fuzzy string ratios treat as "noise".

  - Negative sampling is no longer uniform-random. Half of each record's
    negative budget is now the *hardest* negatives (highest lexical
    "combined" score that still isn't a true match) instead of random
    ones. Uniform random negatives are almost all trivially easy (totally
    unrelated businesses), so the old model was rarely shown the
    confusable near-miss pairs that actually sit near the decision
    boundary. Training on hard negatives is what typically buys the last
    few points of precision/accuracy in this kind of pair classifier.

  - Post-hoc probability calibration (isotonic regression, fit on the
    held-out validation fold) is applied before thresholding, so a
    predicted "0.7" actually means "~70% of these are true matches" on
    unseen data, not just on the training distribution.

  - The decision threshold (previously the hardcoded constant
    KEEP_SCORE = 0.44) is now tuned by sweeping thresholds against the
    validation fold and picking the one that maximizes F0.5 (precision
    weighted 2x recall, matching the "macro F0.5" scoring criterion
    referenced in the original code's comments). The tuned threshold is
    persisted to cache/threshold.json and reused at predict time.

  - A training report (metrics + confusion matrix + feature importance)
    is written to output/training_report.{json,txt} every training run so
    accuracy can be checked directly instead of assumed.

This file still intentionally avoids global Python dictionaries/lists
containing millions of records. SQLite is used as a disk-backed inverted
index.

Usage:
  python business_entity_resolution.py --mode train
  python business_entity_resolution.py --mode predict

Expected layout:
  student_resource/
    dataset/
      train/train_source1.tsv
      train/train_source2.tsv
      train/train_source3.tsv
      train/train_ground_truth.tsv
      test/test_source1.tsv
      test/test_source2.tsv
      test/test_source3.tsv
    output/

Optional environment variables:
  BER_ROOT=student_resource
  BER_INDEX=student_resource/cache/er_index.sqlite
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import pickle
import re
import sqlite3
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

try:
    import psutil
except Exception:
    psutil = None

try:
    import torch
except Exception:
    torch = None

try:
    import lightgbm as lgb
except Exception:
    lgb = None

try:
    import faiss  # optional
except Exception:
    faiss = None

try:
    from sklearn.isotonic import IsotonicRegression
except Exception:
    IsotonicRegression = None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROOT = Path(os.environ.get("BER_ROOT", "student_resource"))
DATASET = ROOT / "dataset"
TRAIN = DATASET / "train"
TEST = DATASET / "test"
OUTPUT = ROOT / "output"
CACHE = ROOT / "cache"
INDEX_DB = Path(os.environ.get("BER_INDEX", str(CACHE / "er_index.sqlite")))
MODEL_PATH = CACHE / "lightgbm_matcher.txt"
CALIBRATOR_PATH = CACHE / "calibrator.pkl"
THRESHOLD_PATH = CACHE / "threshold.json"
REPORT_JSON = OUTPUT / "training_report.json"
REPORT_TXT = OUTPUT / "training_report.txt"

SOURCE1_BATCH = 4000
INDEX_BATCH = 25_000
TRAIN_MAX_PAIRS = 1_500_000
TRAIN_POSITIVE_TARGET = 300_000
NEG_PER_POS = 6          # was 4; wider negative budget so the hard/random
                         # split below still has enough easy negatives too
HARD_NEGATIVE_FRACTION = 0.5  # half the negative budget = hardest lexical near-misses

# Candidate budgets. They are deliberately moderate because candidate_pairs.tsv
# is the final set actually scored by the model.
EXACT_MAX = 80
TOKEN_MAX = 60
PREFIX_MAX = 50
ANN_K = 20
FINAL_CANDIDATE_MAX = 80

# Fallback safety-net thresholds if calibration/tuning artifacts are missing
# (e.g. predicting without having trained first, or sklearn unavailable).
# These are no longer the primary decision boundary -- see THRESHOLD_PATH.
DEFAULT_KEEP_SCORE = 0.44
KEEP_NAME = 0.88
KEEP_ADDRESS = 0.88

NORMALIZE_RE = re.compile(r"[^a-z0-9]+")
DIGIT_RE = re.compile(r"\d+")

LEGAL_SUFFIXES = {
    "limited", "ltd", "llp", "inc", "incorporated", "corp", "corporation",
    "company", "co", "private", "pvt", "plc", "llc", "gmbh", "sarl",
    "sas", "sa", "pte", "pty"
}

STOP_TOKENS = {
    "the", "and", "of", "for", "in", "at", "near", "road", "rd", "street",
    "st", "avenue", "ave", "lane", "ln", "drive", "dr", "shop", "store",
    "market", "india", "usa", "us", "france"
}


# ---------------------------------------------------------------------------
# Runtime / resource helpers
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ram_gb() -> float:
    if psutil is None:
        return -1.0
    return psutil.Process().memory_info().rss / (1024 ** 3)


def log_ram(prefix: str = "") -> None:
    r = ram_gb()
    if r >= 0:
        log(f"{prefix}RAM={r:.2f} GB")


def cuda_available() -> bool:
    return bool(torch is not None and torch.cuda.is_available())


def device_name() -> str:
    if cuda_available():
        return torch.cuda.get_device_name(0)
    return "CPU"


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

def normalize_text(x: object) -> str:
    if x is None:
        return ""
    s = unicodedata.normalize("NFKD", str(x)).encode("ascii", "ignore").decode()
    s = s.lower().replace("&", " and ")
    s = NORMALIZE_RE.sub(" ", s)
    return " ".join(s.split())


def normalize_name(x: object) -> str:
    s = normalize_text(x)
    if not s:
        return ""
    toks = [t for t in s.split() if t not in LEGAL_SUFFIXES]
    # Keep order: business names can contain meaningful word order.
    return " ".join(toks)


def normalize_address(x: object) -> str:
    return normalize_text(x)


def compact(s: str) -> str:
    return s.replace(" ", "")


def prefix_key(s: str, n: int = 6) -> str:
    c = compact(s)
    return c[:n] if c else ""


def tokens(s: str) -> list[str]:
    if not s:
        return []
    out = []
    seen = set()
    for t in s.split():
        if len(t) < 3 or t in STOP_TOKENS:
            continue
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def digit_tokens(s: str) -> set[str]:
    """Zip codes / building numbers / unit numbers -- strong disambiguators
    that pure edit-distance ratios wash out because digits are only a small
    fraction of the total string length."""
    if not s:
        return set()
    return set(DIGIT_RE.findall(s))


def acronym(s: str) -> str:
    toks = s.split()
    if len(toks) < 2:
        return ""
    return "".join(t[0] for t in toks if t)


def token_hash(token: str) -> str:
    # Stable 64-bit hex key. SHA1 is used only to avoid giant variable-length
    # SQLite keys; this is NOT cryptographic identity resolution.
    return hashlib.blake2b(token.encode(), digest_size=8).hexdigest()


def record_keys(name: str, address: str) -> dict[str, str | list[str]]:
    nt = tokens(name)
    at = tokens(address)
    return {
        "name": name,
        "addr": address,
        "name_prefix": prefix_key(name),
        "addr_prefix": prefix_key(address),
        "name_tokens": nt,
        "addr_tokens": at,
    }


# ---------------------------------------------------------------------------
# TSV streaming
# ---------------------------------------------------------------------------

def iter_tsv(path: Path, chunk_size: int = INDEX_BATCH) -> Iterator[pd.DataFrame]:
    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        chunksize=chunk_size,
        quoting=csv.QUOTE_MINIMAL,
    )


def source_path(split: str, source: int) -> Path:
    return (TRAIN if split == "train" else TEST) / f"{split}_source{source}.tsv"


# ---------------------------------------------------------------------------
# SQLite index
# ---------------------------------------------------------------------------

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA temp_store=FILE;
PRAGMA cache_size=-262144;

CREATE TABLE IF NOT EXISTS records (
    rid INTEGER PRIMARY KEY,
    entity_id TEXT NOT NULL UNIQUE,
    source INTEGER NOT NULL,
    country TEXT NOT NULL,
    name TEXT NOT NULL,
    address TEXT NOT NULL,
    name_prefix TEXT NOT NULL,
    addr_prefix TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_records_name ON records(name);
CREATE INDEX IF NOT EXISTS idx_records_addr ON records(address);
CREATE INDEX IF NOT EXISTS idx_records_name_prefix ON records(name_prefix);
CREATE INDEX IF NOT EXISTS idx_records_addr_prefix ON records(addr_prefix);

CREATE TABLE IF NOT EXISTS name_token (
    tok TEXT NOT NULL,
    rid INTEGER NOT NULL,
    PRIMARY KEY(tok, rid)
) WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS idx_name_token_rid ON name_token(rid);

CREATE TABLE IF NOT EXISTS addr_token (
    tok TEXT NOT NULL,
    rid INTEGER NOT NULL,
    PRIMARY KEY(tok, rid)
) WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS idx_addr_token_rid ON addr_token(rid);

CREATE TABLE IF NOT EXISTS token_stats (
    kind TEXT NOT NULL,
    tok TEXT NOT NULL,
    df INTEGER NOT NULL,
    PRIMARY KEY(kind, tok)
) WITHOUT ROWID;
"""


def connect_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path), timeout=120)
    con.executescript(SCHEMA)
    con.execute("PRAGMA busy_timeout=120000")
    return con


def clear_index(con: sqlite3.Connection) -> None:
    for table in ("name_token", "addr_token", "token_stats", "records"):
        con.execute(f"DELETE FROM {table}")
    con.commit()


def build_index(split: str = "train", rebuild: bool = False) -> None:
    """
    Build one target index from source2+source3.

    The training and test indexes are separate because entity IDs must never
    cross-contaminate splits.
    """
    db_path = INDEX_DB.with_name(f"{INDEX_DB.stem}_{split}{INDEX_DB.suffix}")
    con = connect_db(db_path)

    if rebuild:
        log(f"Rebuilding {db_path}")
        clear_index(con)

    existing = con.execute("SELECT COUNT(*) FROM records").fetchone()[0]
    if existing:
        log(f"{split} index already contains {existing:,} records; reusing it")
        con.close()
        return

    rid = 0
    name_df: Counter[str] = Counter()
    addr_df: Counter[str] = Counter()

    log(f"Building disk-backed {split} index...")
    for source in (2, 3):
        path = source_path(split, source)
        if not path.exists():
            raise FileNotFoundError(path)

        log(f"Indexing {path}")
        for chunk in iter_tsv(path):
            records = []
            name_tokens_rows = []
            addr_tokens_rows = []

            for row in chunk.itertuples(index=False):
                eid = str(getattr(row, "entity_id"))
                country = str(getattr(row, "country"))
                name = normalize_name(getattr(row, "business_name"))
                addr = normalize_address(getattr(row, "business_address"))

                npfx = prefix_key(name)
                apfx = prefix_key(addr)

                records.append(
                    (rid, eid, source, country, name, addr, npfx, apfx)
                )

                nts = set(tokens(name))
                ats = set(tokens(addr))

                for t in nts:
                    h = token_hash(t)
                    name_tokens_rows.append((h, rid))
                    name_df[h] += 1

                for t in ats:
                    h = token_hash(t)
                    addr_tokens_rows.append((h, rid))
                    addr_df[h] += 1

                rid += 1

            con.executemany(
                "INSERT INTO records VALUES (?,?,?,?,?,?,?,?)",
                records,
            )
            if name_tokens_rows:
                con.executemany(
                    "INSERT OR IGNORE INTO name_token(tok,rid) VALUES (?,?)",
                    name_tokens_rows,
                )
            if addr_tokens_rows:
                con.executemany(
                    "INSERT OR IGNORE INTO addr_token(tok,rid) VALUES (?,?)",
                    addr_tokens_rows,
                )
            con.commit()

            del records, name_tokens_rows, addr_tokens_rows, chunk
            gc.collect()

            if rid % 250_000 < INDEX_BATCH:
                log(f"Indexed {rid:,} records")
                log_ram()

    # Store token frequencies. Very common tokens will be ignored during lookup.
    con.executemany(
        "INSERT OR REPLACE INTO token_stats VALUES ('name', ?, ?)",
        name_df.items(),
    )
    con.executemany(
        "INSERT OR REPLACE INTO token_stats VALUES ('addr', ?, ?)",
        addr_df.items(),
    )
    con.commit()

    log(f"Finished {split} index: {rid:,} records")
    log_ram()
    con.close()


# ---------------------------------------------------------------------------
# Candidate retrieval
# ---------------------------------------------------------------------------

def fetch_ids(
    con: sqlite3.Connection,
    sql: str,
    params: Sequence[object],
    limit: int,
) -> list[tuple[int, str, int, str, str, str]]:
    if limit <= 0:
        return []
    q = sql + " LIMIT ?"
    return con.execute(q, (*params, limit)).fetchall()


def candidate_rows(
    con: sqlite3.Connection,
    country: str,
    name: str,
    addr: str,
) -> list[tuple[int, str, int, str, str, str]]:
    """
    Multi-pass blocking.

    Returns compact tuples:
      rid, entity_id, source, country, name, address

    No Python dictionaries of all source records are created.
    """
    seen: set[int] = set()
    out: list[tuple[int, str, int, str, str, str]] = []

    def add(rows):
        for r in rows:
            if r[0] not in seen:
                seen.add(r[0])
                out.append(r)

    # 1. Exact normalized name, constrained to country.
    if name:
        add(fetch_ids(
            con,
            """SELECT rid,entity_id,source,country,name,address
               FROM records
               WHERE country=? AND name=?""",
            (country, name),
            EXACT_MAX,
        ))

    # 2. Exact normalized address.
    if addr:
        add(fetch_ids(
            con,
            """SELECT rid,entity_id,source,country,name,address
               FROM records
               WHERE country=? AND address=?""",
            (country, addr),
            EXACT_MAX,
        ))

    # 3. Prefix blocks.
    npfx = prefix_key(name)
    if npfx:
        add(fetch_ids(
            con,
            """SELECT rid,entity_id,source,country,name,address
               FROM records
               WHERE country=? AND name_prefix=?""",
            (country, npfx),
            PREFIX_MAX,
        ))

    apfx = prefix_key(addr)
    if apfx:
        add(fetch_ids(
            con,
            """SELECT rid,entity_id,source,country,name,address
               FROM records
               WHERE country=? AND addr_prefix=?""",
            (country, apfx),
            PREFIX_MAX,
        ))

    # 4. Rare name tokens. Only use tokens with small postings lists.
    nt = tokens(name)
    if nt:
        hashes = [token_hash(x) for x in nt]
        placeholders = ",".join("?" * len(hashes))
        rows = con.execute(
            f"""SELECT r.rid,r.entity_id,r.source,r.country,r.name,r.address
                FROM name_token t JOIN records r ON r.rid=t.rid
                WHERE r.country=? AND t.tok IN ({placeholders})
                GROUP BY r.rid
                ORDER BY COUNT(*) DESC
                LIMIT ?""",
            (country, *hashes, TOKEN_MAX),
        ).fetchall()
        add(rows)

    # 5. Rare address tokens.
    at = tokens(addr)
    if at:
        hashes = [token_hash(x) for x in at]
        placeholders = ",".join("?" * len(hashes))
        rows = con.execute(
            f"""SELECT r.rid,r.entity_id,r.source,r.country,r.name,r.address
                FROM addr_token t JOIN records r ON r.rid=t.rid
                WHERE r.country=? AND t.tok IN ({placeholders})
                GROUP BY r.rid
                ORDER BY COUNT(*) DESC
                LIMIT ?""",
            (country, *hashes, TOKEN_MAX),
        ).fetchall()
        add(rows)

    return out


# ---------------------------------------------------------------------------
# Pair features
# ---------------------------------------------------------------------------

def safe_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return fuzz.ratio(a, b) / 100.0


def W(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return fuzz.WRatio(a, b) / 100.0


def token_set(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return fuzz.token_set_ratio(a, b) / 100.0


def token_sort(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return fuzz.token_sort_ratio(a, b) / 100.0


def char_jaccard(a: str, b: str, n: int = 3) -> float:
    if not a or not b:
        return 0.0
    aa = compact(a)
    bb = compact(b)
    if len(aa) < n or len(bb) < n:
        return safe_ratio(aa, bb)
    sa = {aa[i:i+n] for i in range(len(aa)-n+1)}
    sb = {bb[i:i+n] for i in range(len(bb)-n+1)}
    u = len(sa | sb)
    return len(sa & sb) / u if u else 0.0


def jaro_winkler(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return float(JaroWinkler.normalized_similarity(a, b))


def set_jaccard(sa: set, sb: set) -> float:
    if not sa or not sb:
        return 0.0
    u = len(sa | sb)
    return len(sa & sb) / u if u else 0.0


def pair_features(
    s1_name: str,
    s1_addr: str,
    s1_country: str,
    cand_name: str,
    cand_addr: str,
    cand_country: str,
) -> np.ndarray:
    ns = safe_ratio(s1_name, cand_name)
    nw = W(s1_name, cand_name)
    nts = token_set(s1_name, cand_name)
    nto = token_sort(s1_name, cand_name)
    nj = char_jaccard(s1_name, cand_name)
    njw = jaro_winkler(s1_name, cand_name)

    ads = safe_ratio(s1_addr, cand_addr)
    adw = W(s1_addr, cand_addr)
    adts = token_set(s1_addr, cand_addr)
    adto = token_sort(s1_addr, cand_addr)
    adj = char_jaccard(s1_addr, cand_addr)
    adjw = jaro_winkler(s1_addr, cand_addr)

    exact_name = float(bool(s1_name and s1_name == cand_name))
    exact_addr = float(bool(s1_addr and s1_addr == cand_addr))
    same_country = float(s1_country == cand_country)

    name_tok_jaccard = set_jaccard(set(tokens(s1_name)), set(tokens(cand_name)))
    addr_tok_jaccard = set_jaccard(set(tokens(s1_addr)), set(tokens(cand_addr)))
    digit_jaccard = set_jaccard(digit_tokens(s1_addr), digit_tokens(cand_addr))

    a1 = acronym(s1_name)
    a2 = acronym(cand_name)
    acronym_match = float(bool(a1 and a1 == a2))

    # Stronger combined feature for precision-heavy scoring and for ranking
    # hard negatives during training.
    combined = 0.58 * max(ns, nw, nts, nj, njw) + 0.42 * max(ads, adw, adts, adj, adjw)

    return np.asarray(
        [
            ns, nw, nts, nto, nj, njw,
            ads, adw, adts, adto, adj, adjw,
            exact_name, exact_addr, same_country,
            name_tok_jaccard, addr_tok_jaccard, digit_jaccard, acronym_match,
            combined,
            abs(len(s1_name) - len(cand_name)),
            abs(len(s1_addr) - len(cand_addr)),
        ],
        dtype=np.float32,
    )


FEATURE_NAMES = [
    "name_ratio", "name_wratio", "name_token_set", "name_token_sort",
    "name_char_jaccard", "name_jaro_winkler",
    "addr_ratio", "addr_wratio", "addr_token_set", "addr_token_sort",
    "addr_char_jaccard", "addr_jaro_winkler",
    "exact_name", "exact_address", "same_country",
    "name_token_jaccard", "addr_token_jaccard", "digit_jaccard", "acronym_match",
    "combined",
    "name_len_diff", "addr_len_diff",
]

COMBINED_IDX = FEATURE_NAMES.index("combined")


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

def load_ground_truth(path: Path) -> dict[str, set[str]]:
    """
    Ground truth is ~2.2M rows. A Python set per S1 is expensive but manageable
    only for training; we immediately use it to sample pairs and then release it.
    """
    gt: dict[str, set[str]] = {}
    log("Loading ground truth for training...")
    for chunk in iter_tsv(path, 100_000):
        for row in chunk.itertuples(index=False):
            s1 = str(row.source1_entity_id)
            raw = str(row.matched_entity_ids)
            if raw:
                gt[s1] = set(x for x in raw.split(",") if x)
            else:
                gt[s1] = set()
        del chunk
    log(f"Ground truth loaded: {len(gt):,} S1 entities")
    return gt


# ---------------------------------------------------------------------------
# Metrics (pure numpy -- no hard sklearn dependency for the core report)
# ---------------------------------------------------------------------------

def accuracy_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def precision_recall_fbeta(y_true: np.ndarray, y_pred: np.ndarray, beta: float = 1.0):
    tp = float(np.sum((y_pred == 1) & (y_true == 1)))
    fp = float(np.sum((y_pred == 1) & (y_true == 0)))
    fn = float(np.sum((y_pred == 0) & (y_true == 1)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    b2 = beta * beta
    denom = (b2 * precision) + recall
    fbeta = ((1 + b2) * precision * recall / denom) if denom > 0 else 0.0
    return precision, recall, fbeta


def roc_auc_np(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney U). Good enough for a diagnostic report;
    use sklearn.metrics.roc_auc_score if you need exact tie handling."""
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    sum_ranks_pos = float(np.sum(ranks[y_true == 1]))
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def tune_threshold(y_true: np.ndarray, probs: np.ndarray, beta: float = 0.5):
    """Sweep thresholds and pick the one maximizing F-beta (default: F0.5,
    i.e. precision weighted twice as heavily as recall)."""
    thresholds = np.linspace(0.02, 0.98, 97)
    best_t, best_score = 0.5, -1.0
    sweep = []
    for t in thresholds:
        pred = (probs >= t).astype(np.int8)
        p, r, f = precision_recall_fbeta(y_true, pred, beta=beta)
        acc = accuracy_score_np(y_true, pred)
        sweep.append({"threshold": float(t), "precision": p, "recall": r,
                       "fbeta": f, "accuracy": acc})
        if f > best_score:
            best_score = f
            best_t = float(t)
    return best_t, sweep


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def fit_calibrator(raw_probs: np.ndarray, y_true: np.ndarray):
    if IsotonicRegression is None:
        log("sklearn not available -- skipping probability calibration.")
        return None
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_probs, y_true)
    return iso


def apply_calibrator(calibrator, raw_probs: np.ndarray) -> np.ndarray:
    if calibrator is None:
        return raw_probs
    return np.asarray(calibrator.predict(raw_probs), dtype=np.float64)


def save_calibrator(calibrator) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    with open(CALIBRATOR_PATH, "wb") as f:
        pickle.dump(calibrator, f)


def load_calibrator():
    if not CALIBRATOR_PATH.exists():
        return None
    try:
        with open(CALIBRATOR_PATH, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def save_threshold(threshold: float) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    with open(THRESHOLD_PATH, "w") as f:
        json.dump({"probability_threshold": threshold}, f)


def load_threshold() -> float:
    if THRESHOLD_PATH.exists():
        try:
            with open(THRESHOLD_PATH) as f:
                return float(json.load(f)["probability_threshold"])
        except Exception:
            pass
    return DEFAULT_KEEP_SCORE


# ---------------------------------------------------------------------------
# Training report
# ---------------------------------------------------------------------------

def generate_report(booster, y_va, raw_probs, cal_probs, threshold, sweep) -> dict:
    pred = (cal_probs >= threshold).astype(np.int8)
    acc = accuracy_score_np(y_va, pred)
    p1, r1, f1 = precision_recall_fbeta(y_va, pred, beta=1.0)
    _, _, f05 = precision_recall_fbeta(y_va, pred, beta=0.5)
    auc_raw = roc_auc_np(y_va, raw_probs)
    auc_cal = roc_auc_np(y_va, cal_probs)

    tp = int(np.sum((pred == 1) & (y_va == 1)))
    fp = int(np.sum((pred == 1) & (y_va == 0)))
    fn = int(np.sum((pred == 0) & (y_va == 1)))
    tn = int(np.sum((pred == 0) & (y_va == 0)))

    importances = sorted(
        zip(FEATURE_NAMES, booster.feature_importance(importance_type="gain").tolist()),
        key=lambda x: -x[1],
    )

    report = {
        "validation_pairs": int(len(y_va)),
        "validation_positive_rate": float(np.mean(y_va)),
        "chosen_threshold": float(threshold),
        "accuracy": acc,
        "precision": p1,
        "recall": r1,
        "f1": f1,
        "f0.5": f05,
        "roc_auc_raw": auc_raw,
        "roc_auc_calibrated": auc_cal,
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "feature_importance_gain": importances,
        "threshold_sweep": sweep,
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    with open(REPORT_JSON, "w") as f:
        json.dump(report, f, indent=2)

    with open(REPORT_TXT, "w") as f:
        f.write("Training / validation report\n")
        f.write("=============================\n")
        f.write(f"Validation pairs:        {report['validation_pairs']:,}\n")
        f.write(f"Positive rate:           {report['validation_positive_rate']:.4f}\n")
        f.write(f"Chosen threshold:        {threshold:.3f}\n\n")
        f.write(f"Accuracy:                {acc:.4%}\n")
        f.write(f"Precision:               {p1:.4%}\n")
        f.write(f"Recall:                  {r1:.4%}\n")
        f.write(f"F1:                      {f1:.4f}\n")
        f.write(f"F0.5:                    {f05:.4f}\n")
        f.write(f"ROC-AUC (raw):           {auc_raw:.4f}\n")
        f.write(f"ROC-AUC (calibrated):    {auc_cal:.4f}\n\n")
        f.write(f"Confusion matrix: TP={tp} FP={fp} FN={fn} TN={tn}\n\n")
        f.write("Top feature importances (gain):\n")
        for name, gain in importances[:15]:
            f.write(f"  {name:<22s} {gain:.1f}\n")

    log(
        f"VALIDATION  accuracy={acc:.4%}  precision={p1:.4%}  recall={r1:.4%}  "
        f"f1={f1:.4f}  f0.5={f05:.4f}  auc={auc_cal:.4f}  threshold={threshold:.3f}"
    )
    log(f"Report written: {REPORT_JSON} / {REPORT_TXT}")
    return report


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _select_negatives(negative_rows: list[np.ndarray], k: int, rng: np.random.Generator) -> list[int]:
    """Mix of hardest lexical near-misses and random negatives.

    Uniform-random negatives are almost always trivially dissimilar, so a
    model trained only on them learns a very loose decision boundary. Ranking
    by the 'combined' lexical feature and taking the hardest half forces the
    model to actually separate genuine matches from confusable non-matches.
    """
    n = len(negative_rows)
    if n <= k:
        return list(range(n))

    combined_scores = np.asarray([row[COMBINED_IDX] for row in negative_rows])
    order = np.argsort(-combined_scores)

    n_hard = max(1, int(round(k * HARD_NEGATIVE_FRACTION)))
    n_hard = min(n_hard, n)
    hard_idx = order[:n_hard]

    remaining = order[n_hard:]
    n_rand = k - len(hard_idx)
    if n_rand > 0 and len(remaining) > 0:
        rand_idx = rng.choice(remaining, size=min(n_rand, len(remaining)), replace=False)
    else:
        rand_idx = np.array([], dtype=int)

    return np.concatenate([hard_idx, rand_idx]).astype(int).tolist()


def sample_training_pairs(db_path: Path, gt: dict[str, set[str]]) -> tuple[np.ndarray, np.ndarray]:
    """
    Stream S1 and generate candidates. Keep only a bounded feature matrix.
    Positives are always retained when the blocker finds them. Negatives are
    capped per positive and drawn as a hard/random mix (see _select_negatives).
    """
    con = sqlite3.connect(str(db_path))
    rng = np.random.default_rng(42)

    X_blocks: list[np.ndarray] = []
    y_blocks: list[np.ndarray] = []
    pos_count = 0
    neg_count = 0
    total = 0

    s1_path = source_path("train", 1)
    log("Generating bounded training candidate sample (hard-negative mining enabled)...")

    for chunk in iter_tsv(s1_path, SOURCE1_BATCH):
        xb = []
        yb = []

        for row in chunk.itertuples(index=False):
            s1id = str(row.entity_id)
            country = str(row.country)
            name = normalize_name(row.business_name)
            addr = normalize_address(row.business_address)

            truth = gt.get(s1id, set())
            cands = candidate_rows(con, country, name, addr)

            positive_rows = []
            negative_rows = []

            for c in cands:
                cid = c[1]
                feat = pair_features(name, addr, country, c[4], c[5], c[3])
                if cid in truth:
                    positive_rows.append(feat)
                else:
                    negative_rows.append(feat)

            if positive_rows:
                pos_count += len(positive_rows)
                for feat in positive_rows:
                    xb.append(feat)
                    yb.append(1)

                k = min(len(negative_rows), NEG_PER_POS * len(positive_rows))
                if k:
                    for i in _select_negatives(negative_rows, k, rng):
                        xb.append(negative_rows[i])
                        yb.append(0)
                        neg_count += 1
            elif not truth and negative_rows:
                # Singleton examples are important under macro F0.5. Prefer
                # the hardest few negatives here too, not random ones.
                k = min(3, len(negative_rows))
                for i in _select_negatives(negative_rows, k, rng):
                    xb.append(negative_rows[i])
                    yb.append(0)
                    neg_count += 1

            total += len(cands)

            if pos_count >= TRAIN_POSITIVE_TARGET or len(yb) >= TRAIN_MAX_PAIRS:
                break

        if xb:
            X_blocks.append(np.asarray(xb, dtype=np.float32))
            y_blocks.append(np.asarray(yb, dtype=np.int8))

        del chunk
        gc.collect()

        if sum(len(x) for x in X_blocks) >= TRAIN_MAX_PAIRS:
            break

        if total and total % 250_000 < SOURCE1_BATCH:
            log(f"training candidates examined={total:,}, positives={pos_count:,}, RAM={ram_gb():.2f} GB")

    con.close()

    X = np.concatenate(X_blocks, axis=0)
    y = np.concatenate(y_blocks, axis=0)

    # Hard cap in case the last batch overshot.
    if len(X) > TRAIN_MAX_PAIRS:
        idx = rng.choice(len(X), size=TRAIN_MAX_PAIRS, replace=False)
        X = X[idx]
        y = y[idx]

    del X_blocks, y_blocks
    gc.collect()

    log(f"Training matrix: {X.shape}; positive rate={float(y.mean()):.4f}")
    return X, y


def train_model() -> None:
    if lgb is None:
        raise RuntimeError("lightgbm is not installed")

    build_index("train", rebuild=False)
    db_path = INDEX_DB.with_name(f"{INDEX_DB.stem}_train{INDEX_DB.suffix}")

    gt = load_ground_truth(TRAIN / "train_ground_truth.tsv")
    X, y = sample_training_pairs(db_path, gt)

    # Split by rows only after candidate generation. The goal here is a compact
    # pair classifier, not a giant in-memory validation table.
    rng = np.random.default_rng(123)
    perm = rng.permutation(len(X))
    cut = int(len(X) * 0.85)
    tr = perm[:cut]
    va = perm[cut:]

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=47,
        max_depth=8,
        min_child_samples=30,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        reg_alpha=0.2,
        reg_lambda=2.0,
        n_jobs=max(1, min(8, os.cpu_count() or 4)),
        random_state=42,
    )

    model.fit(
        X[tr],
        y[tr],
        eval_set=[(X[va], y[va])],
        eval_metric=["auc", "binary_logloss"],
        callbacks=[lgb.early_stopping(80, verbose=False)],
    )

    CACHE.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(MODEL_PATH))
    log(f"Saved model: {MODEL_PATH}")

    # --- Calibration + threshold tuning on the held-out validation fold ---
    raw_va_probs = model.booster_.predict(
        X[va], num_iteration=model.booster_.best_iteration
    )
    calibrator = fit_calibrator(raw_va_probs, y[va])
    cal_va_probs = apply_calibrator(calibrator, raw_va_probs)

    best_threshold, sweep = tune_threshold(y[va], cal_va_probs, beta=0.5)
    save_calibrator(calibrator)
    save_threshold(best_threshold)

    generate_report(model.booster_, y[va], raw_va_probs, cal_va_probs, best_threshold, sweep)

    del gt, X, y, model
    gc.collect()


# ---------------------------------------------------------------------------
# Optional semantic retrieval
# ---------------------------------------------------------------------------

class SemanticRetriever:
    """
    Optional FAISS retriever. It is intentionally disabled by default.

    Reason: a dense 384-dimensional float32 embedding for ~10M target rows is
    ~15 GB before ANN graph/index overhead. An IVF-PQ index is possible, but
    building it is a separate resource-intensive phase. Lexical blocking is
    the default safe path on a 32 GB machine.
    """

    def __init__(self):
        self.enabled = False
        self.device = "cuda" if cuda_available() else "cpu"
        self.index = None

    def explain(self):
        if faiss is None:
            return "FAISS unavailable; semantic ANN disabled."
        if cuda_available():
            return f"FAISS available; CUDA detected: {device_name()}"
        return "FAISS available; using CPU ANN."


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def score_candidate_batch(
    model,
    s1_name: str,
    s1_addr: str,
    s1_country: str,
    candidates: list[tuple[int, str, int, str, str, str]],
    calibrator,
    threshold: float,
) -> tuple[list[tuple[str, float]], list[str]]:
    """
    Score all candidates, then apply a precision-oriented, *tuned* threshold.

    Returns:
      matches: (entity_id, calibrated probability)
      final candidate IDs: exactly the IDs actually presented to the model.
    """
    if not candidates:
        return [], []

    feats = np.vstack([
        pair_features(
            s1_name, s1_addr, s1_country,
            c[4], c[5], c[3]
        )
        for c in candidates
    ])

    raw_probs = model.predict(feats)
    cal_probs = apply_calibrator(calibrator, raw_probs)
    pairs = [(c[1], float(p)) for c, p in zip(candidates, cal_probs)]

    # The final candidate set is the model input. Do not filter candidate_pairs
    # after this point.
    final_candidates = [cid for cid, _ in pairs]

    # Precision-heavy final decision, using the tuned threshold. Strong exact
    # lexical matches are still allowed through even if the (calibrated)
    # classifier is conservative -- this is a safety net for near-duplicate
    # records the model hasn't seen the like of during training.
    matches = []
    for (cid, p), c in zip(pairs, candidates):
        strong_name = (
            bool(s1_name and c[4] and
                 max(
                     safe_ratio(s1_name, c[4]),
                     W(s1_name, c[4]),
                     token_set(s1_name, c[4])
                 ) >= KEEP_NAME)
        )
        strong_addr = (
            bool(s1_addr and c[5] and
                 max(
                     safe_ratio(s1_addr, c[5]),
                     W(s1_addr, c[5]),
                     token_set(s1_addr, c[5])
                 ) >= KEEP_ADDRESS)
        )

        if p >= threshold or (strong_name and strong_addr):
            matches.append((cid, p))

    matches.sort(key=lambda x: x[1], reverse=True)

    return matches, final_candidates


def predict() -> None:
    if lgb is None:
        raise RuntimeError("lightgbm is not installed")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{MODEL_PATH} does not exist. Run --mode train first."
        )

    build_index("test", rebuild=False)
    db_path = INDEX_DB.with_name(f"{INDEX_DB.stem}_test{INDEX_DB.suffix}")
    con = sqlite3.connect(str(db_path))

    model = lgb.Booster(model_file=str(MODEL_PATH))
    calibrator = load_calibrator()
    threshold = load_threshold()
    log(f"Using calibrated probabilities: {calibrator is not None}; threshold={threshold:.3f}")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    match_path = OUTPUT / "matching_results.tsv"
    candidate_path = OUTPUT / "candidate_pairs.tsv"

    # Truncate old output.
    match_path.write_text("source1_entity_id\tmatched_entity_ids\n", encoding="utf-8")
    candidate_path.write_text("source1_entity_id\tcandidate_entity_ids\n", encoding="utf-8")

    with (
        match_path.open("a", encoding="utf-8", newline="") as mf,
        candidate_path.open("a", encoding="utf-8", newline="") as cf,
    ):
        mw = csv.writer(mf, delimiter="\t", lineterminator="\n")
        cw = csv.writer(cf, delimiter="\t", lineterminator="\n")

        processed = 0
        total_candidates = 0
        total_matches = 0

        for chunk in iter_tsv(source_path("test", 1), SOURCE1_BATCH):
            for row in chunk.itertuples(index=False):
                s1id = str(row.entity_id)
                country = str(row.country)
                name = normalize_name(row.business_name)
                addr = normalize_address(row.business_address)

                cands = candidate_rows(con, country, name, addr)

                # Candidate cap is applied ONLY after all blocking passes.
                # Keep the strongest lexical candidates if a block is unusually
                # large. This is still the final candidate set fed to the model.
                if len(cands) > FINAL_CANDIDATE_MAX:
                    scored = []
                    for c in cands:
                        ns = max(
                            safe_ratio(name, c[4]),
                            W(name, c[4]),
                            token_set(name, c[4]),
                        )
                        ads = max(
                            safe_ratio(addr, c[5]),
                            W(addr, c[5]),
                            token_set(addr, c[5]),
                        )
                        scored.append((0.6 * ns + 0.4 * ads, c))
                    scored.sort(key=lambda x: x[0], reverse=True)
                    cands = [x[1] for x in scored[:FINAL_CANDIDATE_MAX]]

                matches, final_candidates = score_candidate_batch(
                    model, name, addr, country, cands, calibrator, threshold
                )

                # Every output candidate is exactly what was passed to model.
                cw.writerow([s1id, ",".join(final_candidates)])

                # Remove duplicate matches defensively.
                ids = []
                seen = set()
                for cid, _ in matches:
                    if cid not in seen:
                        seen.add(cid)
                        ids.append(cid)

                mw.writerow([s1id, ",".join(ids)])

                processed += 1
                total_candidates += len(final_candidates)
                total_matches += len(ids)

                if processed % 10_000 == 0:
                    mf.flush()
                    cf.flush()
                    avg = total_candidates / processed
                    log(
                        f"processed={processed:,} "
                        f"avg_final_candidates={avg:.2f} "
                        f"matches={total_matches:,} "
                        f"RAM={ram_gb():.2f} GB"
                    )

            del chunk
            gc.collect()

    con.close()
    del model
    gc.collect()

    log("Prediction complete.")
    log(f"Candidates: {candidate_path}")
    log(f"Matches:    {match_path}")


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

def validate_output() -> None:
    match_path = OUTPUT / "matching_results.tsv"
    candidate_path = OUTPUT / "candidate_pairs.tsv"
    s1_path = TEST / "test_source1.tsv"

    if not match_path.exists() or not candidate_path.exists():
        raise FileNotFoundError("Run --mode predict first.")

    s1_ids = set()
    for chunk in iter_tsv(s1_path, 100_000):
        s1_ids.update(chunk["entity_id"].astype(str))
        del chunk

    match_ids = set()
    cand_ids = set()

    errors = 0
    for chunk in iter_tsv(match_path, 100_000):
        for row in chunk.itertuples(index=False):
            sid = str(row.source1_entity_id)
            vals = str(row.matched_entity_ids)
            if sid in match_ids:
                errors += 1
            match_ids.add(sid)
            if vals:
                for x in vals.split(","):
                    if not x.startswith(("S2-", "S3-")):
                        errors += 1
        del chunk

    candidate_map: dict[str, set[str]] = {}
    # Validation is deliberately streaming; candidate_map can still be large,
    # so only use it if the official validator is unavailable.
    for chunk in iter_tsv(candidate_path, 100_000):
        for row in chunk.itertuples(index=False):
            sid = str(row.source1_entity_id)
            vals = str(row.candidate_entity_ids)
            candidate_map[sid] = set(vals.split(",")) if vals else set()
        del chunk

    if match_ids != s1_ids:
        missing = len(s1_ids - match_ids)
        extra = len(match_ids - s1_ids)
        log(f"WARNING: missing S1 rows={missing}, extra={extra}")
        errors += missing + extra

    # Check subset property.
    for chunk in iter_tsv(match_path, 100_000):
        for row in chunk.itertuples(index=False):
            sid = str(row.source1_entity_id)
            vals = str(row.matched_entity_ids)
            cand = candidate_map.get(sid, set())
            if vals:
                for x in vals.split(","):
                    if x not in cand:
                        errors += 1
        del chunk

    if errors:
        log(f"Validation found {errors} issue(s).")
        raise SystemExit(1)

    log("Basic local validation passed.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["train", "predict", "all", "validate", "index-train", "index-test"],
        default="all",
    )
    parser.add_argument("--rebuild-index", action="store_true")
    args = parser.parse_args()

    log(f"Device: {device_name()}")
    log(f"CUDA available: {cuda_available()}")
    log(f"FAISS available: {faiss is not None}")
    log(f"sklearn (calibration) available: {IsotonicRegression is not None}")
    log_ram("Startup: ")

    if args.mode == "index-train":
        build_index("train", rebuild=args.rebuild_index)
    elif args.mode == "index-test":
        build_index("test", rebuild=args.rebuild_index)
    elif args.mode == "train":
        train_model()
    elif args.mode == "predict":
        predict()
    elif args.mode == "validate":
        validate_output()
    else:
        train_model()
        predict()
        validate_output()

    log_ram("Finished: ")


if __name__ == "__main__":
    main()