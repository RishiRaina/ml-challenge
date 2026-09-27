#!/usr/bin/env python3
"""
Business Entity Resolution Challenge - single-file pipeline.

Run from student_resource/:

python business_entity_resolution.py \
  --train-dir dataset/train \
  --test-dir dataset/test \
  --output-dir output \
  --artifact-dir artifacts \
  --optuna-trials 12

Inputs:
  dataset/train/train_source1.tsv
  dataset/train/train_source2.tsv
  dataset/train/train_source3.tsv
  dataset/train/train_ground_truth.tsv
  dataset/test/test_source1.tsv
  dataset/test/test_source2.tsv
  dataset/test/test_source3.tsv

Outputs:
  output/matching_results.tsv
  output/candidate_pairs.tsv

The code does not use external business databases, APIs, geocoding,
or internet entity lookup.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rapidfuzz.fuzz import (
    WRatio,
    ratio,
    partial_ratio,
    token_set_ratio,
    token_sort_ratio,
)
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# DATA
# ============================================================

def read_tsv(path):
    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )


def load_data(train_dir, test_dir):
    train_dir = Path(train_dir)
    test_dir = Path(test_dir)

    train = {
        "s1": read_tsv(train_dir / "train_source1.tsv"),
        "s2": read_tsv(train_dir / "train_source2.tsv"),
        "s3": read_tsv(train_dir / "train_source3.tsv"),
        "gt": read_tsv(train_dir / "train_ground_truth.tsv"),
    }

    test = {
        "s1": read_tsv(test_dir / "test_source1.tsv"),
        "s2": read_tsv(test_dir / "test_source2.tsv"),
        "s3": read_tsv(test_dir / "test_source3.tsv"),
    }

    return train, test


def truth_map(gt):
    result = {}

    for _, row in gt.iterrows():
        source1_id = row["source1_entity_id"]
        value = str(row["matched_entity_ids"])

        result[source1_id] = [
            x.strip()
            for x in value.split(",")
            if x.strip()
        ]

    return result


def targets(source2, source3):
    return pd.concat(
        [source2, source3],
        ignore_index=True,
    )


# ============================================================
# NORMALIZATION
# ============================================================

LEGAL_SUFFIXES = {
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "co",
    "company",
    "ltd",
    "limited",
    "llc",
    "llp",
    "plc",
    "pvt",
    "private",
    "pte",
    "gmbh",
    "sa",
    "sarl",
    "ag",
    "bv",
    "sdn",
    "bhd",
}

ABBREVIATIONS = {
    "street": "st",
    "road": "rd",
    "avenue": "ave",
    "av": "ave",
    "boulevard": "blvd",
    "drive": "dr",
    "lane": "ln",
    "highway": "hwy",
    "apartment": "apt",
    "building": "bldg",
    "floor": "fl",
    "centre": "center",
    "&": "and",
}


def normalize_string(value):
    value = unicodedata.normalize(
        "NFKC",
        str(value or ""),
    ).lower()

    value = value.replace("&", " and ")

    value = re.sub(
        r"[^\w\s]",
        " ",
        value,
        flags=re.UNICODE,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    ).strip()

    return value


def tokens(value, drop_legal=False):
    result = normalize_string(value).split()

    if drop_legal:
        result = [
            token
            for token in result
            if token not in LEGAL_SUFFIXES
        ]

    return tuple(
        ABBREVIATIONS.get(token, token)
        for token in result
    )


def normalized_name(value):
    return " ".join(
        sorted(
            tokens(
                value,
                drop_legal=True,
            )
        )
    )


def compact_name(value):
    return "".join(
        tokens(
            value,
            drop_legal=True,
        )
    )


def normalized_address(value):
    return " ".join(
        tokens(
            value,
            drop_legal=False,
        )
    )


def digit_tokens(value):
    value = normalize_string(value)

    return tuple(
        re.findall(
            r"\d+[a-z]?",
            value,
        )
    )


def prepare(df):
    result = df.copy()

    result["business_name"] = (
        result["business_name"]
        .fillna("")
        .astype(str)
    )

    result["business_address"] = (
        result["business_address"]
        .fillna("")
        .astype(str)
    )

    result["country"] = (
        result["country"]
        .fillna("")
        .astype(str)
    )

    result["name_norm"] = result["business_name"].map(
        normalized_name
    )

    result["name_compact"] = result["business_name"].map(
        compact_name
    )

    result["address_norm"] = result["business_address"].map(
        normalized_address
    )

    result["name_tokens"] = result["business_name"].map(
        lambda x: tokens(x, drop_legal=True)
    )

    result["address_tokens"] = result["business_address"].map(
        lambda x: tokens(x, drop_legal=False)
    )

    result["digits"] = result["business_address"].map(
        digit_tokens
    )

    result["country_norm"] = result["country"].map(
        normalize_string
    )

    return result


# ============================================================
# BLOCKING
# ============================================================

class Blocker:
    """
    Multi-pass candidate generation:

    1. Exact normalized-name blocking within country.
    2. Rare-token inverted-index blocking.
    3. Name character TF-IDF nearest neighbors.
    4. Address character TF-IDF nearest neighbors.

    The final candidate set is capped before the ML model sees it.
    """

    def __init__(
        self,
        target_df,
        candidate_cap=18,
        name_k=8,
        address_k=8,
    ):
        self.target = prepare(
            target_df
        ).reset_index(drop=True)

        self.candidate_cap = candidate_cap
        self.name_k = name_k
        self.address_k = address_k

        self.by_country = defaultdict(list)
        self.exact_name = defaultdict(list)
        self.postings = defaultdict(list)

        token_counts = Counter()

        for index, row in self.target.iterrows():
            country = row["country_norm"]

            self.by_country[country].append(index)

            if row["name_norm"]:
                self.exact_name[
                    (country, row["name_norm"])
                ].append(index)

            all_tokens = (
                set(row["name_tokens"])
                |
                set(row["address_tokens"])
            )

            for token in all_tokens:
                if len(token) >= 3 and token.isalnum():
                    self.postings[
                        (country, token)
                    ].append(index)

                    token_counts[
                        (country, token)
                    ] += 1

        self.token_counts = token_counts

        self.name_vectorizers = {}
        self.name_nn = {}

        self.address_vectorizers = {}
        self.address_nn = {}

        self._build_indexes()

    def _build_indexes(self):
        for country, indexes in self.by_country.items():

            names = (
                self.target.iloc[indexes]
                ["business_name"]
                .astype(str)
                .tolist()
            )

            addresses = (
                self.target.iloc[indexes]
                ["business_address"]
                .astype(str)
                .tolist()
            )

            # ----------------------
            # Name index
            # ----------------------

            if any(names):
                vectorizer = TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(2, 5),
                    sublinear_tf=True,
                    max_features=100_000,
                )

                try:
                    matrix = vectorizer.fit_transform(
                        names
                    )

                    neighbors = NearestNeighbors(
                        n_neighbors=min(
                            self.name_k,
                            len(indexes),
                        ),
                        metric="cosine",
                    )

                    neighbors.fit(matrix)

                    self.name_vectorizers[country] = vectorizer
                    self.name_nn[country] = (
                        indexes,
                        neighbors,
                    )

                except ValueError:
                    pass

            # ----------------------
            # Address index
            # ----------------------

            if any(addresses):
                vectorizer = TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 6),
                    sublinear_tf=True,
                    max_features=120_000,
                )

                try:
                    matrix = vectorizer.fit_transform(
                        addresses
                    )

                    neighbors = NearestNeighbors(
                        n_neighbors=min(
                            self.address_k,
                            len(indexes),
                        ),
                        metric="cosine",
                    )

                    neighbors.fit(matrix)

                    self.address_vectorizers[country] = vectorizer
                    self.address_nn[country] = (
                        indexes,
                        neighbors,
                    )

                except ValueError:
                    pass

    @staticmethod
    def jaccard(a, b):
        a = set(a)
        b = set(b)

        if not a or not b:
            return 0.0

        return len(a & b) / len(a | b)

    def cheap_score(self, left, right):
        return (
            4.0
            * float(
                bool(left["name_norm"])
                and left["name_norm"] == right["name_norm"]
            )
            +
            3.0
            * float(
                bool(left["address_norm"])
                and left["address_norm"] == right["address_norm"]
            )
            +
            1.5
            * self.jaccard(
                left["name_tokens"],
                right["name_tokens"],
            )
            +
            1.25
            * self.jaccard(
                left["address_tokens"],
                right["address_tokens"],
            )
            +
            0.75
            * self.jaccard(
                left["digits"],
                right["digits"],
            )
            +
            0.5
            * float(
                left["country_norm"]
                == right["country_norm"]
            )
        )

    def get_candidates_for_row(self, row):
        country = row["country_norm"]

        pool = set()

        # ----------------------
        # Exact normalized name
        # ----------------------

        pool.update(
            self.exact_name.get(
                (
                    country,
                    row["name_norm"],
                ),
                [],
            )[:8]
        )

        # ----------------------
        # Rare-token blocking
        # ----------------------

        candidate_tokens = sorted(
            set(row["name_tokens"])
            |
            set(row["address_tokens"]),
            key=lambda token:
                self.token_counts.get(
                    (country, token),
                    10**12,
                ),
        )[:20]

        for token in candidate_tokens:
            pool.update(
                self.postings.get(
                    (country, token),
                    [],
                )
            )

        # ----------------------
        # Name TF-IDF kNN
        # ----------------------

        if country in self.name_vectorizers:
            vectorizer = self.name_vectorizers[country]
            indexes, nn = self.name_nn[country]

            query = vectorizer.transform(
                [row["business_name"]]
            )

            _, nearest = nn.kneighbors(query)

            pool.update(
                indexes[int(index)]
                for index in nearest[0]
            )

        # ----------------------
        # Address TF-IDF kNN
        # ----------------------

        if country in self.address_vectorizers:
            vectorizer = self.address_vectorizers[country]
            indexes, nn = self.address_nn[country]

            query = vectorizer.transform(
                [row["business_address"]]
            )

            _, nearest = nn.kneighbors(query)

            pool.update(
                indexes[int(index)]
                for index in nearest[0]
            )

        # ----------------------
        # Rank candidates
        # ----------------------

        ranked = []

        for index in pool:
            candidate = self.target.iloc[index]

            score = self.cheap_score(
                row,
                candidate,
            )

            ranked.append(
                (
                    score,
                    str(candidate["entity_id"]),
                )
            )

        ranked.sort(
            reverse=True
        )

        return [
            entity_id
            for _, entity_id in ranked[
                : self.candidate_cap
            ]
        ]

    def generate(self, source1_df):
        source1 = prepare(source1_df)

        result = {}

        for _, row in source1.iterrows():
            result[row["entity_id"]] = (
                self.get_candidates_for_row(row)
            )

        return result


def blocking_recall(candidates, truth):
    total = 0
    hits = 0

    for source1_id, true_ids in truth.items():
        true_set = set(true_ids)
        candidate_set = set(
            candidates.get(
                source1_id,
                [],
            )
        )

        total += len(true_set)
        hits += len(
            true_set & candidate_set
        )

    if total == 0:
        return 1.0

    return hits / total


# ============================================================
# FEATURES
# ============================================================

def jaccard(a, b):
    a = set(a)
    b = set(b)

    if not a or not b:
        return 0.0

    return len(a & b) / len(a | b)


def min_overlap(a, b):
    a = set(a)
    b = set(b)

    if not a or not b:
        return 0.0

    return len(a & b) / min(
        len(a),
        len(b),
    )


class FeatureBuilder:
    def __init__(
        self,
        use_embeddings=True,
        embedding_model=None,
    ):
        self.enabled = False
        self.model = None

        if use_embeddings:
            try:
                from sentence_transformers import (
                    SentenceTransformer
                )

                self.model = SentenceTransformer(
                    embedding_model
                    or
                    "sentence-transformers/"
                    "paraphrase-multilingual-"
                    "MiniLM-L12-v2"
                )

                self.enabled = True

            except Exception as error:
                print(
                    "[WARN] Could not load embeddings:"
                    f" {error}"
                )

    def encode(self, texts):
        return self.model.encode(
            texts,
            batch_size=128,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def build(
        self,
        source1_df,
        target_df,
        pairs,
    ):
        source1 = (
            prepare(source1_df)
            .set_index("entity_id")
        )

        target = (
            prepare(target_df)
            .set_index("entity_id")
        )

        feature_rows = []

        full_text_left = []
        full_text_right = []

        name_left = []
        name_right = []

        for pair in pairs:
            left = source1.loc[
                pair["source1_entity_id"]
            ]

            right = target.loc[
                pair["candidate_entity_id"]
            ]

            left_name = str(
                left["business_name"]
            )

            right_name = str(
                right["business_name"]
            )

            left_address = str(
                left["business_address"]
            )

            right_address = str(
                right["business_address"]
            )

            features = {
                # Country
                "same_country":
                    float(
                        left["country_norm"]
                        ==
                        right["country_norm"]
                        and bool(
                            left["country_norm"]
                        )
                    ),

                # Name
                "name_exact":
                    float(
                        bool(left["name_norm"])
                        and
                        left["name_norm"]
                        ==
                        right["name_norm"]
                    ),

                "name_compact":
                    float(
                        bool(left["name_compact"])
                        and
                        left["name_compact"]
                        ==
                        right["name_compact"]
                    ),

                "name_ratio":
                    ratio(
                        left_name,
                        right_name,
                    ) / 100.0,

                "name_wratio":
                    WRatio(
                        left_name,
                        right_name,
                    ) / 100.0,

                "name_partial":
                    partial_ratio(
                        left_name,
                        right_name,
                    ) / 100.0,

                "name_token_set":
                    token_set_ratio(
                        left_name,
                        right_name,
                    ) / 100.0,

                "name_token_sort":
                    token_sort_ratio(
                        left_name,
                        right_name,
                    ) / 100.0,

                "name_jaccard":
                    jaccard(
                        left["name_tokens"],
                        right["name_tokens"],
                    ),

                "name_overlap":
                    min_overlap(
                        left["name_tokens"],
                        right["name_tokens"],
                    ),

                # Address
                "address_exact":
                    float(
                        bool(left["address_norm"])
                        and
                        left["address_norm"]
                        ==
                        right["address_norm"]
                    ),

                "address_ratio":
                    ratio(
                        left_address,
                        right_address,
                    ) / 100.0,

                "address_wratio":
                    WRatio(
                        left_address,
                        right_address,
                    ) / 100.0,

                "address_partial":
                    partial_ratio(
                        left_address,
                        right_address,
                    ) / 100.0,

                "address_token_set":
                    token_set_ratio(
                        left_address,
                        right_address,
                    ) / 100.0,

                "address_token_sort":
                    token_sort_ratio(
                        left_address,
                        right_address,
                    ) / 100.0,

                "address_jaccard":
                    jaccard(
                        left["address_tokens"],
                        right["address_tokens"],
                    ),

                "address_overlap":
                    min_overlap(
                        left["address_tokens"],
                        right["address_tokens"],
                    ),

                # Numeric/address evidence
                "digit_jaccard":
                    jaccard(
                        left["digits"],
                        right["digits"],
                    ),

                "digit_exact":
                    float(
                        bool(left["digits"])
                        and
                        left["digits"]
                        ==
                        right["digits"]
                    ),

                # Length
                "name_length_ratio":
                    min(
                        len(left_name),
                        len(right_name),
                    )
                    /
                    max(
                        1,
                        max(
                            len(left_name),
                            len(right_name),
                        ),
                    ),

                "address_length_ratio":
                    min(
                        len(left_address),
                        len(right_address),
                    )
                    /
                    max(
                        1,
                        max(
                            len(left_address),
                            len(right_address),
                        ),
                    ),

                # Source
                "candidate_is_s2":
                    float(
                        str(
                            right["entity_id"]
                        ).startswith("S2-")
                    ),

                "candidate_is_s3":
                    float(
                        str(
                            right["entity_id"]
                        ).startswith("S3-")
                    ),
            }

            feature_rows.append(
                features
            )

            full_text_left.append(
                f"{left_name} [SEP] {left_address}"
            )

            full_text_right.append(
                f"{right_name} [SEP] {right_address}"
            )

            name_left.append(
                left_name
            )

            name_right.append(
                right_name
            )

        features_df = pd.DataFrame(
            feature_rows
        ).astype(np.float32)

        # ----------------------
        # Semantic embeddings
        # ----------------------

        if self.enabled and feature_rows:
            left_embeddings = self.encode(
                full_text_left
            )

            right_embeddings = self.encode(
                full_text_right
            )

            features_df[
                "semantic_cosine"
            ] = np.sum(
                left_embeddings
                *
                right_embeddings,
                axis=1,
            )

            left_name_embeddings = self.encode(
                name_left
            )

            right_name_embeddings = self.encode(
                name_right
            )

            features_df[
                "semantic_name_cosine"
            ] = np.sum(
                left_name_embeddings
                *
                right_name_embeddings,
                axis=1,
            )

        else:
            features_df[
                "semantic_cosine"
            ] = 0.0

            features_df[
                "semantic_name_cosine"
            ] = 0.0

        return features_df


# ============================================================
# TRAINING DATA
# ============================================================

def build_pair_rows(
    candidate_map,
    target_df,
):
    valid_ids = set(
        target_df["entity_id"]
    )

    rows = []

    for source1_id, candidate_ids in candidate_map.items():
        for candidate_id in candidate_ids:

            if candidate_id not in valid_ids:
                continue

            rows.append(
                {
                    "source1_entity_id":
                        source1_id,

                    "candidate_entity_id":
                        candidate_id,
                }
            )

    return rows


def sample_training_pairs(
    candidate_rows,
    truth,
    negatives_per_positive=8,
    seed=42,
):
    rng = random.Random(seed)

    true_pairs = {
        (source1_id, candidate_id)
        for source1_id, candidate_ids
        in truth.items()
        for candidate_id in candidate_ids
    }

    grouped = defaultdict(list)

    for row in candidate_rows:
        item = dict(row)

        item["label"] = int(
            (
                item["source1_entity_id"],
                item["candidate_entity_id"],
            )
            in true_pairs
        )

        grouped[
            item["source1_entity_id"]
        ].append(item)

    result = []
    seen = set()

    for source1_id, rows in grouped.items():
        positives = [
            row
            for row in rows
            if row["label"] == 1
        ]

        negatives = [
            row
            for row in rows
            if row["label"] == 0
        ]

        max_negatives = max(
            5,
            negatives_per_positive
            *
            max(
                1,
                len(positives),
            ),
        )

        if len(negatives) > max_negatives:
            hard_count = max_negatives // 2

            hard_negatives = negatives[
                :hard_count
            ]

            remaining = negatives[
                hard_count:
            ]

            random_negatives = rng.sample(
                remaining,
                max_negatives
                -
                len(hard_negatives),
            )

            negatives = (
                hard_negatives
                +
                random_negatives
            )

        for row in positives + negatives:
            key = (
                row["source1_entity_id"],
                row["candidate_entity_id"],
            )

            if key not in seen:
                seen.add(key)
                result.append(row)

    # Ensure every known positive is available for training.
    for source1_id, candidate_ids in truth.items():
        for candidate_id in candidate_ids:
            key = (
                source1_id,
                candidate_id,
            )

            if key not in seen:
                result.append(
                    {
                        "source1_entity_id":
                            source1_id,

                        "candidate_entity_id":
                            candidate_id,

                        "label": 1,
                    }
                )

    rng.shuffle(result)

    return result


# ============================================================
# METRICS
# ============================================================

def f05(precision, recall):
    denominator = (
        0.25 * precision
        +
        recall
    )

    if denominator <= 0:
        return 0.0

    return (
        1.25
        *
        precision
        *
        recall
        /
        denominator
    )


def macro_f05(
    truth,
    predictions,
):
    values = []

    for source1_id, true_ids in truth.items():
        true_set = set(true_ids)
        predicted_set = set(
            predictions.get(
                source1_id,
                [],
            )
        )

        tp = len(
            true_set
            &
            predicted_set
        )

        fp = len(
            predicted_set
            -
            true_set
        )

        fn = len(
            true_set
            -
            predicted_set
        )

        if tp + fp:
            precision = (
                tp
                /
                (tp + fp)
            )
        else:
            precision = 1.0

        if tp + fn:
            recall = (
                tp
                /
                (tp + fn)
            )
        else:
            recall = 1.0

        values.append(
            f05(
                precision,
                recall,
            )
        )

    if not values:
        return 0.0

    return float(
        np.mean(values)
    )


def precision_recall(
    truth,
    predictions,
):
    tp = fp = fn = 0

    for source1_id, true_ids in truth.items():
        true_set = set(true_ids)

        predicted_set = set(
            predictions.get(
                source1_id,
                [],
            )
        )

        tp += len(
            true_set
            &
            predicted_set
        )

        fp += len(
            predicted_set
            -
            true_set
        )

        fn += len(
            true_set
            -
            predicted_set
        )

    precision = (
        tp / (tp + fp)
        if tp + fp
        else 1.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn
        else 1.0
    )

    return (
        precision,
        recall,
        tp,
        fp,
        fn,
    )


# ============================================================
# THRESHOLD OPTIMIZATION
# ============================================================

def threshold_search(
    rows,
    scores,
    truth,
):
    grouped = defaultdict(list)

    for row, score in zip(
        rows,
        scores,
    ):
        grouped[
            row["source1_entity_id"]
        ].append(
            (
                row["candidate_entity_id"],
                float(score),
            )
        )

    score_values = np.asarray(
        scores,
        dtype=float,
    )

    thresholds = np.unique(
        np.concatenate(
            [
                np.linspace(
                    0.50,
                    0.995,
                    100,
                ),

                np.clip(
                    score_values,
                    0.0,
                    1.0,
                ),
            ]
        )
    )

    best_score = -1.0
    best_threshold = 0.90

    for threshold in thresholds:
        predictions = {}

        for source1_id, candidates in grouped.items():
            predictions[source1_id] = [
                candidate_id
                for candidate_id, score
                in candidates
                if score >= threshold
            ]

        score = macro_f05(
            truth,
            predictions,
        )

        if score > best_score:
            best_score = score
            best_threshold = float(
                threshold
            )

    return (
        best_threshold,
        best_score,
    )


# ============================================================
# LIGHTGBM + OPTUNA
# ============================================================

def make_lightgbm(parameters=None):
    from lightgbm import LGBMClassifier

    defaults = {
        "objective": "binary",
        "n_estimators": 400,
        "learning_rate": 0.035,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 25,
        "subsample": 0.90,
        "colsample_bytree": 0.90,
        "reg_alpha": 0.05,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
    }

    if parameters:
        defaults.update(
            parameters
        )

    return LGBMClassifier(
        **defaults
    )


def tune_lightgbm(
    X,
    y,
    groups,
    trials=12,
):
    try:
        import optuna
        import lightgbm  # noqa: F401
    except Exception as error:
        print(
            "[WARN] Optuna/LightGBM unavailable:"
            f" {error}"
        )

        return {
            "n_estimators": 400,
            "learning_rate": 0.035,
            "num_leaves": 31,
            "max_depth": -1,
            "min_child_samples": 25,
            "subsample": 0.90,
            "colsample_bytree": 0.90,
            "reg_alpha": 0.05,
            "reg_lambda": 1.0,
        }

    unique_groups = np.unique(
        groups
    )

    n_splits = min(
        4,
        len(unique_groups),
    )

    if n_splits < 2:
        return {
            "n_estimators": 400,
            "learning_rate": 0.035,
            "num_leaves": 31,
            "max_depth": -1,
            "min_child_samples": 25,
            "subsample": 0.90,
            "colsample_bytree": 0.90,
            "reg_alpha": 0.05,
            "reg_lambda": 1.0,
        }

    folds = list(
        GroupKFold(
            n_splits=n_splits
        ).split(
            X,
            y,
            groups,
        )
    )

    def objective(trial):
        parameters = {
            "n_estimators":
                trial.suggest_int(
                    "n_estimators",
                    200,
                    700,
                ),

            "learning_rate":
                trial.suggest_float(
                    "learning_rate",
                    0.01,
                    0.10,
                    log=True,
                ),

            "num_leaves":
                trial.suggest_int(
                    "num_leaves",
                    12,
                    96,
                ),

            "max_depth":
                trial.suggest_int(
                    "max_depth",
                    3,
                    12,
                ),

            "min_child_samples":
                trial.suggest_int(
                    "min_child_samples",
                    10,
                    80,
                ),

            "subsample":
                trial.suggest_float(
                    "subsample",
                    0.65,
                    1.0,
                ),

            "colsample_bytree":
                trial.suggest_float(
                    "colsample_bytree",
                    0.65,
                    1.0,
                ),

            "reg_alpha":
                trial.suggest_float(
                    "reg_alpha",
                    1e-4,
                    2.0,
                    log=True,
                ),

            "reg_lambda":
                trial.suggest_float(
                    "reg_lambda",
                    1e-3,
                    8.0,
                    log=True,
                ),
        }

        fold_scores = []

        for train_indices, valid_indices in folds:
            model = make_lightgbm(
                parameters
            )

            model.fit(
                X[train_indices],
                y[train_indices],
            )

            probabilities = (
                model
                .predict_proba(
                    X[valid_indices]
                )[:, 1]
            )

            order = np.argsort(
                -probabilities
            )

            labels = (
                y[valid_indices]
                [order]
            )

            true_positive = np.cumsum(
                labels
            )

            false_positive = np.cumsum(
                1 - labels
            )

            precision = (
                true_positive
                /
                np.maximum(
                    1,
                    true_positive
                    +
                    false_positive,
                )
            )

            total_positive = max(
                1,
                int(labels.sum())
            )

            recall = (
                true_positive
                /
                total_positive
            )

            fold_f05 = (
                1.25
                *
                precision
                *
                recall
                /
                np.maximum(
                    1e-12,
                    0.25 * precision
                    +
                    recall,
                )
            )

            fold_scores.append(
                float(
                    np.max(
                        fold_f05
                    )
                )
            )

        return float(
            np.mean(
                fold_scores
            )
        )

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=42
        ),
    )

    study.optimize(
        objective,
        n_trials=trials,
        show_progress_bar=False,
    )

    return dict(
        study.best_params
    )


# ============================================================
# FINAL RESOLVER / ENSEMBLE
# ============================================================

class Resolver:
    def __init__(
        self,
        parameters,
        ensemble=True,
    ):
        self.parameters = parameters
        self.ensemble = ensemble

        self.models = []
        self.weights = []

    def fit(self, X, y):
        lightgbm_model = make_lightgbm(
            self.parameters
        )

        lightgbm_model.fit(
            X,
            y,
        )

        self.models = [
            lightgbm_model
        ]

        self.weights = [
            1.0
        ]

        if self.ensemble:
            extra_trees = ExtraTreesClassifier(
                n_estimators=500,
                min_samples_leaf=2,
                max_features="sqrt",
                random_state=42,
                n_jobs=-1,
            )

            logistic_regression = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=2.0,
                    max_iter=2000,
                    random_state=42,
                ),
            )

            extra_trees.fit(
                X,
                y,
            )

            logistic_regression.fit(
                X,
                y,
            )

            self.models.extend(
                [
                    extra_trees,
                    logistic_regression,
                ]
            )

            self.weights = [
                0.70,
                0.20,
                0.10,
            ]

        return self

    def predict(self, X):
        probabilities = np.vstack(
            [
                model.predict_proba(X)[:, 1]
                for model in self.models
            ]
        )

        weights = (
            np.asarray(
                self.weights,
                dtype=float,
            )
            /
            np.sum(
                self.weights
            )
        )

        return np.average(
            probabilities,
            axis=0,
            weights=weights,
        )

    def save(self, path):
        joblib.dump(
            self,
            path,
        )


# ============================================================
# OUTPUT
# ============================================================

def write_outputs(
    source1,
    predictions,
    candidates,
    output_dir,
):
    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    matching = pd.DataFrame(
        [
            {
                "source1_entity_id":
                    source1_id,

                "matched_entity_ids":
                    ",".join(
                        sorted(
                            set(
                                predictions.get(
                                    source1_id,
                                    [],
                                )
                            )
                        )
                    ),
            }
            for source1_id
            in source1["entity_id"]
        ]
    )

    candidate_output = pd.DataFrame(
        [
            {
                "source1_entity_id":
                    source1_id,

                "candidate_entity_ids":
                    ",".join(
                        sorted(
                            set(
                                candidates.get(
                                    source1_id,
                                    [],
                                )
                            )
                        )
                    ),
            }
            for source1_id
            in source1["entity_id"]
        ]
    )

    matching.to_csv(
        output_dir
        /
        "matching_results.tsv",
        sep="\t",
        index=False,
    )

    candidate_output.to_csv(
        output_dir
        /
        "candidate_pairs.tsv",
        sep="\t",
        index=False,
    )


def validate_outputs(
    test_dir,
    output_dir,
):
    test_dir = Path(test_dir)
    output_dir = Path(output_dir)

    source1 = read_tsv(
        test_dir / "test_source1.tsv"
    )

    source2 = read_tsv(
        test_dir / "test_source2.tsv"
    )

    source3 = read_tsv(
        test_dir / "test_source3.tsv"
    )

    matching = read_tsv(
        output_dir
        /
        "matching_results.tsv"
    )

    candidates = read_tsv(
        output_dir
        /
        "candidate_pairs.tsv"
    )

    errors = []

    expected_source1_ids = set(
        source1["entity_id"]
    )

    valid_target_ids = (
        set(source2["entity_id"])
        |
        set(source3["entity_id"])
    )

    if list(
        matching.columns
    ) != [
        "source1_entity_id",
        "matched_entity_ids",
    ]:
        errors.append(
            "Wrong matching_results.tsv columns"
        )

    if list(
        candidates.columns
    ) != [
        "source1_entity_id",
        "candidate_entity_ids",
    ]:
        errors.append(
            "Wrong candidate_pairs.tsv columns"
        )

    if (
        len(matching)
        != len(expected_source1_ids)
        or
        set(
            matching["source1_entity_id"]
        )
        != expected_source1_ids
    ):
        errors.append(
            "matching_results.tsv must contain every "
            "test Source-1 entity exactly once"
        )

    if (
        len(candidates)
        != len(expected_source1_ids)
        or
        set(
            candidates["source1_entity_id"]
        )
        != expected_source1_ids
    ):
        errors.append(
            "candidate_pairs.tsv must contain every "
            "test Source-1 entity exactly once"
        )

    candidate_map = {}

    for _, row in candidates.iterrows():
        source1_id = (
            row["source1_entity_id"]
        )

        ids = [
            x
            for x
            in str(
                row["candidate_entity_ids"]
            ).split(",")
            if x
        ]

        if len(ids) != len(set(ids)):
            errors.append(
                f"Duplicate candidate IDs for {source1_id}"
            )

        invalid = (
            set(ids)
            -
            valid_target_ids
        )

        if invalid:
            errors.append(
                f"Invalid candidate IDs for {source1_id}: "
                f"{sorted(invalid)[:5]}"
            )

        candidate_map[source1_id] = set(
            ids
        )

    for _, row in matching.iterrows():
        source1_id = (
            row["source1_entity_id"]
        )

        ids = [
            x
            for x
            in str(
                row["matched_entity_ids"]
            ).split(",")
            if x
        ]

        if len(ids) != len(set(ids)):
            errors.append(
                f"Duplicate match IDs for {source1_id}"
            )

        invalid = (
            set(ids)
            -
            valid_target_ids
        )

        if invalid:
            errors.append(
                f"Invalid match IDs for {source1_id}: "
                f"{sorted(invalid)[:5]}"
            )

        if not set(ids).issubset(
            candidate_map.get(
                source1_id,
                set(),
            )
        ):
            errors.append(
                f"Match outside candidate set for "
                f"{source1_id}"
            )

    if errors:
        print("FAIL")

        for number, error in enumerate(
            errors,
            start=1,
        ):
            print(
                f"{number}. {error}"
            )

        return False

    print("PASS")
    return True


# ============================================================
# MAIN PIPELINE
# ============================================================

def run(args):
    train, test = load_data(
        args.train_dir,
        args.test_dir,
    )

    truth = truth_map(
        train["gt"]
    )

    train_target = targets(
        train["s2"],
        train["s3"],
    )

    # --------------------------------------------------------
    # 1. Entity-level train/validation split
    # --------------------------------------------------------

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42,
    )

    train_indices, validation_indices = next(
        splitter.split(
            train["s1"],
            groups=train["s1"]["entity_id"],
        )
    )

    source1_train = (
        train["s1"]
        .iloc[train_indices]
        .reset_index(drop=True)
    )

    source1_validation = (
        train["s1"]
        .iloc[validation_indices]
        .reset_index(drop=True)
    )

    truth_train = {
        source1_id:
            truth.get(
                source1_id,
                [],
            )
        for source1_id
        in source1_train["entity_id"]
    }

    truth_validation = {
        source1_id:
            truth.get(
                source1_id,
                [],
            )
        for source1_id
        in source1_validation["entity_id"]
    }

    # --------------------------------------------------------
    # 2. Candidate generation
    # --------------------------------------------------------

    blocker = Blocker(
        train_target,
        candidate_cap=args.candidate_cap,
        name_k=args.name_k,
        address_k=args.address_k,
    )

    candidates_train = (
        blocker.generate(
            source1_train
        )
    )

    candidates_validation = (
        blocker.generate(
            source1_validation
        )
    )

    validation_blocking_recall = (
        blocking_recall(
            candidates_validation,
            truth_validation,
        )
    )

    validation_candidate_count = (
        np.mean(
            [
                len(candidates)
                for candidates
                in
                candidates_validation.values()
            ]
        )
    )

    print(
        "[BLOCKING] validation recall: "
        f"{validation_blocking_recall:.4f}"
    )

    print(
        "[BLOCKING] mean validation candidates: "
        f"{validation_candidate_count:.2f}"
    )

    # --------------------------------------------------------
    # 3. Training and validation pair creation
    # --------------------------------------------------------

    train_candidate_rows = (
        build_pair_rows(
            candidates_train,
            train_target,
        )
    )

    validation_candidate_rows = (
        build_pair_rows(
            candidates_validation,
            train_target,
        )
    )

    train_pairs = sample_training_pairs(
        train_candidate_rows,
        truth_train,
        negatives_per_positive=8,
        seed=42,
    )

    # --------------------------------------------------------
    # 4. Feature engineering
    # --------------------------------------------------------

    feature_builder = FeatureBuilder(
        use_embeddings=not args.no_embeddings,
        embedding_model=args.embedding_model,
    )

    X_train = feature_builder.build(
        source1_train,
        train_target,
        train_pairs,
    )

    y_train = np.asarray(
        [
            row["label"]
            for row
            in train_pairs
        ],
        dtype=np.int8,
    )

    X_validation = feature_builder.build(
        source1_validation,
        train_target,
        validation_candidate_rows,
    )

    groups = np.asarray(
        [
            row["source1_entity_id"]
            for row
            in train_pairs
        ]
    )

    print(
        "[TRAIN] number of training pairs: "
        f"{len(train_pairs)}"
    )

    print(
        "[TRAIN] positive rate: "
        f"{y_train.mean():.4f}"
    )

    # --------------------------------------------------------
    # 5. Hyperparameter optimization
    # --------------------------------------------------------

    print(
        "[OPTUNA] trials: "
        f"{args.optuna_trials}"
    )

    best_parameters = tune_lightgbm(
        X_train.values,
        y_train,
        groups,
        trials=args.optuna_trials,
    )

    print(
        "[OPTUNA] best parameters:"
    )

    print(
        json.dumps(
            best_parameters,
            indent=2,
        )
    )

    # --------------------------------------------------------
    # 6. Compare single model vs ensemble
    # --------------------------------------------------------

    single_model = Resolver(
        best_parameters,
        ensemble=False,
    ).fit(
        X_train.values,
        y_train,
    )

    single_scores = (
        single_model.predict(
            X_validation.values
        )
    )

    single_threshold, single_f05 = (
        threshold_search(
            validation_candidate_rows,
            single_scores,
            truth_validation,
        )
    )

    ensemble_model = Resolver(
        best_parameters,
        ensemble=True,
    ).fit(
        X_train.values,
        y_train,
    )

    ensemble_scores = (
        ensemble_model.predict(
            X_validation.values
        )
    )

    ensemble_threshold, ensemble_f05 = (
        threshold_search(
            validation_candidate_rows,
            ensemble_scores,
            truth_validation,
        )
    )

    if ensemble_f05 > single_f05:
        selected_model = ensemble_model
        threshold = ensemble_threshold
        validation_f05 = ensemble_f05
        model_type = "ensemble"
    else:
        selected_model = single_model
        threshold = single_threshold
        validation_f05 = single_f05
        model_type = "lightgbm"

    print(
        "[MODEL] selected: "
        f"{model_type}"
    )

    print(
        "[VALIDATION] macro F0.5: "
        f"{validation_f05:.6f}"
    )

    print(
        "[VALIDATION] threshold: "
        f"{threshold:.4f}"
    )

    validation_predictions = defaultdict(list)

    selected_validation_scores = (
        selected_model.predict(
            X_validation.values
        )
    )

    for row, score in zip(
        validation_candidate_rows,
        selected_validation_scores,
    ):
        if score >= threshold:
            validation_predictions[
                row["source1_entity_id"]
            ].append(
                row["candidate_entity_id"]
            )

    validation_precision, validation_recall, tp, fp, fn = (
        precision_recall(
            truth_validation,
            validation_predictions,
        )
    )

    print(
        "[VALIDATION] precision: "
        f"{validation_precision:.6f}"
    )

    print(
        "[VALIDATION] recall: "
        f"{validation_recall:.6f}"
    )

    print(
        "[VALIDATION] "
        f"TP={tp}, FP={fp}, FN={fn}"
    )

    # --------------------------------------------------------
    # 7. Retrain on all labeled training entities
    # --------------------------------------------------------

    full_blocker = Blocker(
        train_target,
        candidate_cap=args.candidate_cap,
        name_k=args.name_k,
        address_k=args.address_k,
    )

    full_candidates = (
        full_blocker.generate(
            train["s1"]
        )
    )

    full_candidate_rows = (
        build_pair_rows(
            full_candidates,
            train_target,
        )
    )

    full_train_pairs = (
        sample_training_pairs(
            full_candidate_rows,
            truth,
            negatives_per_positive=8,
            seed=42,
        )
    )

    X_full = feature_builder.build(
        train["s1"],
        train_target,
        full_train_pairs,
    )

    y_full = np.asarray(
        [
            row["label"]
            for row
            in full_train_pairs
        ],
        dtype=np.int8,
    )

    final_model = Resolver(
        best_parameters,
        ensemble=(
            model_type
            ==
            "ensemble"
        ),
    ).fit(
        X_full.values,
        y_full,
    )

    # --------------------------------------------------------
    # 8. Test candidate generation
    # --------------------------------------------------------

    test_target = targets(
        test["s2"],
        test["s3"],
    )

    test_blocker = Blocker(
        test_target,
        candidate_cap=args.candidate_cap,
        name_k=args.name_k,
        address_k=args.address_k,
    )

    test_candidates = (
        test_blocker.generate(
            test["s1"]
        )
    )

    test_candidate_rows = (
        build_pair_rows(
            test_candidates,
            test_target,
        )
    )

    # --------------------------------------------------------
    # 9. Test scoring
    # --------------------------------------------------------

    if test_candidate_rows:
        X_test = feature_builder.build(
            test["s1"],
            test_target,
            test_candidate_rows,
        )

        test_scores = (
            final_model.predict(
                X_test.values
            )
        )

    else:
        test_scores = np.array(
            [],
            dtype=float,
        )

    predictions = defaultdict(list)

    for row, score in zip(
        test_candidate_rows,
        test_scores,
    ):
        if score >= threshold:
            predictions[
                row["source1_entity_id"]
            ].append(
                row["candidate_entity_id"]
            )

    # Ensure every Source-1 entity appears.
    for source1_id in test["s1"]["entity_id"]:
        predictions.setdefault(
            source1_id,
            [],
        )

    # --------------------------------------------------------
    # 10. Write outputs
    # --------------------------------------------------------

    write_outputs(
        test["s1"],
        predictions,
        test_candidates,
        args.output_dir,
    )

    # --------------------------------------------------------
    # 11. Save model/config
    # --------------------------------------------------------

    artifact_dir = Path(
        args.artifact_dir
    )

    artifact_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_model.save(
        artifact_dir
        /
        "resolver_model.joblib"
    )

    config = {
        "model_type": model_type,
        "threshold": threshold,
        "best_parameters": best_parameters,
        "candidate_cap": args.candidate_cap,
        "name_k": args.name_k,
        "address_k": args.address_k,
        "embeddings_enabled":
            feature_builder.enabled,
        "validation_f05":
            validation_f05,
        "validation_precision":
            validation_precision,
        "validation_recall":
            validation_recall,
        "validation_blocking_recall":
            validation_blocking_recall,
    }

    (
        artifact_dir
        /
        "config.json"
    ).write_text(
        json.dumps(
            config,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # 12. Validate final files
    # --------------------------------------------------------

    validate_outputs(
        args.test_dir,
        args.output_dir,
    )

    print()
    print(
        "DONE"
    )

    print(
        "matching_results.tsv:"
        f" {Path(args.output_dir).resolve() / 'matching_results.tsv'}"
    )

    print(
        "candidate_pairs.tsv:"
        f" {Path(args.output_dir).resolve() / 'candidate_pairs.tsv'}"
    )


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-dir",
        default="dataset/train",
    )

    parser.add_argument(
        "--test-dir",
        default="dataset/test",
    )

    parser.add_argument(
        "--output-dir",
        default="output",
    )

    parser.add_argument(
        "--artifact-dir",
        default="artifacts",
    )

    parser.add_argument(
        "--optuna-trials",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--candidate-cap",
        type=int,
        default=18,
    )

    parser.add_argument(
        "--name-k",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--address-k",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--embedding-model",
        default=(
            "sentence-transformers/"
            "paraphrase-multilingual-MiniLM-L12-v2"
        ),
    )

    parser.add_argument(
        "--no-embeddings",
        action="store_true",
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
    )

    args = parser.parse_args()

    if args.validate_only:
        success = validate_outputs(
            args.test_dir,
            args.output_dir,
        )

        raise SystemExit(
            0 if success else 1
        )

    run(args)


if __name__ == "__main__":
    main()