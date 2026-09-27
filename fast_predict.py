#!/usr/bin/env python3

"""
FAST PREDICTOR V2
=================

Uses the already-created:

    student_resource/cache/er_index_test.sqlite
    student_resource/cache/lightgbm_matcher.txt
    student_resource/cache/calibrator.pkl
    student_resource/cache/threshold.json

Does NOT rebuild the index.
Does NOT retrain the model.

Source1 columns:

    entity_id
    business_name
    business_address
    country
"""

from __future__ import annotations

import csv
import multiprocessing as mp
import os
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

import business_entity_resolution as ber


# ============================================================================
# PATHS
# ============================================================================

ROOT = Path("student_resource")

SOURCE1 = ROOT / "dataset" / "test" / "test_source1.tsv"

DB_PATH = ROOT / "cache" / "er_index_test.sqlite"
MODEL_PATH = ROOT / "cache" / "lightgbm_matcher.txt"

OUTPUT = ROOT / "output"

MATCH_FILE = OUTPUT / "matching_results.tsv"
CANDIDATE_FILE = OUTPUT / "candidate_pairs.tsv"


# ============================================================================
# PERFORMANCE
# ============================================================================

CPU_COUNT = os.cpu_count() or 4

# Your CPU has 16 logical processors.
WORKERS = min(8, max(2, CPU_COUNT - 2))

# Number of Source1 rows read at once.
READ_CHUNK = 32000

# Multiprocessing scheduling batch.
POOL_CHUNKSIZE = 128

# Print progress every N records.
PRINT_EVERY = 5000


# ============================================================================
# CANDIDATE LIMITS
# ============================================================================

EXACT_NAME_LIMIT = 80
EXACT_ADDRESS_LIMIT = 80

NAME_PREFIX_LIMIT = 30
ADDRESS_PREFIX_LIMIT = 30

TOKEN_LIMIT = 20

FINAL_CANDIDATE_LIMIT = 60


# ============================================================================
# MATCH THRESHOLDS
# ============================================================================

KEEP_NAME = 0.88
KEEP_ADDRESS = 0.88


# ============================================================================
# WORKER GLOBALS
# ============================================================================

CON = None
MODEL = None
CALIBRATOR = None
THRESHOLD = None


# ============================================================================
# STARTUP BANNER
# ============================================================================

print("=" * 70)
print("FAST PREDICTION V2")
print("=" * 70)
print("Script started successfully.")
print()


# ============================================================================
# WORKER INITIALIZATION
# ============================================================================

def init_worker(db_path, model_path):

    global CON
    global MODEL
    global CALIBRATOR
    global THRESHOLD

    # ------------------------------------------------------------
    # SQLite read-only connection
    # ------------------------------------------------------------

    db_uri = (
        "file:"
        + Path(db_path).resolve().as_posix()
        + "?mode=ro"
    )

    CON = sqlite3.connect(
        db_uri,
        uri=True,
        timeout=60,
        check_same_thread=False,
    )

    CON.execute(
        "PRAGMA query_only=ON"
    )

    CON.execute(
        "PRAGMA temp_store=MEMORY"
    )

    CON.execute(
        "PRAGMA cache_size=-32768"
    )

    # ------------------------------------------------------------
    # Load LightGBM once per worker
    # ------------------------------------------------------------

    MODEL = lgb.Booster(
        model_file=str(model_path)
    )

    # ------------------------------------------------------------
    # Load calibration + threshold
    # ------------------------------------------------------------

    CALIBRATOR = ber.load_calibrator()

    THRESHOLD = ber.load_threshold()


# ============================================================================
# SQLITE FETCH
# ============================================================================

def fetch_rows(sql, params, limit):

    return CON.execute(
        sql + " LIMIT ?",
        (*params, limit),
    ).fetchall()


# ============================================================================
# FAST CANDIDATE GENERATION
# ============================================================================

def get_candidates(
    country,
    name,
    address,
):

    seen = set()
    candidates = []

    def add(rows):

        for row in rows:

            rid = row[0]

            if rid not in seen:

                seen.add(rid)
                candidates.append(row)

    # ========================================================================
    # 1. EXACT NAME
    # ========================================================================

    if name:

        rows = fetch_rows(
            """
            SELECT
                rid,
                entity_id,
                source,
                country,
                name,
                address
            FROM records
            WHERE country = ?
              AND name = ?
            """,
            (
                country,
                name,
            ),
            EXACT_NAME_LIMIT,
        )

        add(rows)

    # ========================================================================
    # If exact name found, this is usually already a very strong candidate
    # set. Avoid doing additional expensive SQLite searches.
    # ========================================================================

    if candidates:

        return candidates[:FINAL_CANDIDATE_LIMIT]

    # ========================================================================
    # 2. EXACT ADDRESS
    # ========================================================================

    if address:

        rows = fetch_rows(
            """
            SELECT
                rid,
                entity_id,
                source,
                country,
                name,
                address
            FROM records
            WHERE country = ?
              AND address = ?
            """,
            (
                country,
                address,
            ),
            EXACT_ADDRESS_LIMIT,
        )

        add(rows)

    if candidates:

        return candidates[:FINAL_CANDIDATE_LIMIT]

    # ========================================================================
    # 3. NAME PREFIX
    # ========================================================================

    name_prefix = ber.prefix_key(name)

    if name_prefix:

        rows = fetch_rows(
            """
            SELECT
                rid,
                entity_id,
                source,
                country,
                name,
                address
            FROM records
            WHERE country = ?
              AND name_prefix = ?
            """,
            (
                country,
                name_prefix,
            ),
            NAME_PREFIX_LIMIT,
        )

        add(rows)

    # ========================================================================
    # 4. ADDRESS PREFIX
    # ========================================================================

    address_prefix = ber.prefix_key(address)

    if address_prefix:

        rows = fetch_rows(
            """
            SELECT
                rid,
                entity_id,
                source,
                country,
                name,
                address
            FROM records
            WHERE country = ?
              AND addr_prefix = ?
            """,
            (
                country,
                address_prefix,
            ),
            ADDRESS_PREFIX_LIMIT,
        )

        add(rows)

    if candidates:

        return candidates[:FINAL_CANDIDATE_LIMIT]

    # ========================================================================
    # 5. NAME TOKEN FALLBACK
    # ========================================================================

    name_tokens = ber.tokens(name)

    if name_tokens:

        # At most three tokens to keep the SQL query cheap.
        name_tokens = name_tokens[:3]

        hashes = [
            ber.token_hash(token)
            for token in name_tokens
        ]

        placeholders = ",".join(
            "?" for _ in hashes
        )

        rows = CON.execute(
            f"""
            SELECT
                r.rid,
                r.entity_id,
                r.source,
                r.country,
                r.name,
                r.address
            FROM name_token t
            JOIN records r
              ON r.rid = t.rid
            WHERE r.country = ?
              AND t.tok IN ({placeholders})
            GROUP BY r.rid
            ORDER BY COUNT(*) DESC
            LIMIT ?
            """,
            (
                country,
                *hashes,
                TOKEN_LIMIT,
            ),
        ).fetchall()

        add(rows)

    return candidates[:FINAL_CANDIDATE_LIMIT]


# ============================================================================
# SCORE CANDIDATES
# ============================================================================

def score_candidates(
    source_name,
    source_address,
    country,
    candidates,
):

    if not candidates:

        return [], []

    # ========================================================================
    # Build the same 22-feature vectors used during training.
    # ========================================================================

    feature_rows = []

    for candidate in candidates:

        feature_rows.append(
            ber.pair_features(
                source_name,
                source_address,
                country,
                candidate[4],
                candidate[5],
                candidate[3],
            )
        )

    features = np.vstack(
        feature_rows
    )

    # ========================================================================
    # LightGBM
    # ========================================================================

    probabilities = MODEL.predict(
        features
    )

    # ========================================================================
    # Calibration
    # ========================================================================

    probabilities = ber.apply_calibrator(
        CALIBRATOR,
        probabilities,
    )

    # ========================================================================
    # Candidate IDs
    # ========================================================================

    candidate_ids = []

    seen = set()

    for candidate in candidates:

        cid = str(candidate[1])

        if cid not in seen:

            seen.add(cid)
            candidate_ids.append(cid)

    # ========================================================================
    # Final decisions
    # ========================================================================

    matches = []

    seen_matches = set()

    for candidate, probability in zip(
        candidates,
        probabilities,
    ):

        cid = str(candidate[1])

        candidate_name = candidate[4]
        candidate_address = candidate[5]

        # --------------------------------------------------------
        # Strong name
        # --------------------------------------------------------

        if (
            source_name
            and candidate_name
        ):

            name_score = max(
                ber.safe_ratio(
                    source_name,
                    candidate_name,
                ),
                ber.W(
                    source_name,
                    candidate_name,
                ),
                ber.token_set(
                    source_name,
                    candidate_name,
                ),
            )

            strong_name = (
                name_score >= KEEP_NAME
            )

        else:

            strong_name = False

        # --------------------------------------------------------
        # Strong address
        # --------------------------------------------------------

        if (
            source_address
            and candidate_address
        ):

            address_score = max(
                ber.safe_ratio(
                    source_address,
                    candidate_address,
                ),
                ber.W(
                    source_address,
                    candidate_address,
                ),
                ber.token_set(
                    source_address,
                    candidate_address,
                ),
            )

            strong_address = (
                address_score >= KEEP_ADDRESS
            )

        else:

            strong_address = False

        # --------------------------------------------------------
        # Final decision
        # --------------------------------------------------------

        if (
            float(probability) >= THRESHOLD
            or (
                strong_name
                and strong_address
            )
        ):

            if cid not in seen_matches:

                seen_matches.add(cid)

                matches.append(
                    (
                        cid,
                        float(probability),
                    )
                )

    # Highest confidence first.
    matches.sort(
        key=lambda x: x[1],
        reverse=True,
    )

    match_ids = [
        cid
        for cid, _ in matches
    ]

    return (
        match_ids,
        candidate_ids,
    )


# ============================================================================
# PROCESS ONE SOURCE1 RECORD
# ============================================================================

def process_one(row):

    source1_id = str(
        row[0]
    )

    raw_name = str(
        row[1]
    )

    raw_address = str(
        row[2]
    )

    country = str(
        row[3]
    )

    # ========================================================================
    # Normalize
    # ========================================================================

    name = ber.normalize_name(
        raw_name
    )

    address = ber.normalize_address(
        raw_address
    )

    # ========================================================================
    # Blocking
    # ========================================================================

    candidates = get_candidates(
        country,
        name,
        address,
    )

    # ========================================================================
    # Scoring
    # ========================================================================

    matches, candidate_ids = score_candidates(
        name,
        address,
        country,
        candidates,
    )

    return (
        source1_id,
        matches,
        candidate_ids,
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    print(
        f"Workers: {WORKERS}"
    )

    print(
        f"CPU logical processors: {CPU_COUNT}"
    )

    print(
        f"Source1: {SOURCE1}"
    )

    print(
        f"Test index: {DB_PATH}"
    )

    print(
        f"Model: {MODEL_PATH}"
    )

    print()

    # ========================================================================
    # Validate files
    # ========================================================================

    required = [
        SOURCE1,
        DB_PATH,
        MODEL_PATH,
    ]

    for path in required:

        if not path.exists():

            raise FileNotFoundError(
                f"Required file not found:\n{path}"
            )

    # ========================================================================
    # Verify SQLite index
    # ========================================================================

    print(
        "Checking test index..."
    )

    check_con = sqlite3.connect(
        str(DB_PATH)
    )

    try:

        indexed_count = check_con.execute(
            "SELECT COUNT(*) FROM records"
        ).fetchone()[0]

    finally:

        check_con.close()

    print(
        f"Indexed target records: "
        f"{indexed_count:,}"
    )

    print()

    # ========================================================================
    # Verify Source1 columns
    # ========================================================================

    header = pd.read_csv(
        SOURCE1,
        sep="\t",
        dtype=str,
        nrows=0,
    )

    expected_columns = [
        "entity_id",
        "business_name",
        "business_address",
        "country",
    ]

    actual_columns = list(
        header.columns
    )

    print(
        f"Source1 columns: "
        f"{actual_columns}"
    )

    if actual_columns != expected_columns:

        raise RuntimeError(
            "\nSource1 columns are not what the predictor expects.\n"
            f"Expected: {expected_columns}\n"
            f"Found: {actual_columns}\n"
        )

    print()

    # ========================================================================
    # Load artifacts
    # ========================================================================

    global THRESHOLD
    global CALIBRATOR

    CALIBRATOR = ber.load_calibrator()

    THRESHOLD = ber.load_threshold()

    print(
        f"Using calibrated probabilities: "
        f"{CALIBRATOR is not None}"
    )

    print(
        f"Decision threshold: "
        f"{THRESHOLD:.3f}"
    )

    print()

    # ========================================================================
    # Output directory
    # ========================================================================

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================================
    # Overwrite partial outputs
    # ========================================================================

    MATCH_FILE.write_text(
        "source1_entity_id\tmatched_entity_ids\n",
        encoding="utf-8",
    )

    CANDIDATE_FILE.write_text(
        "source1_entity_id\tcandidate_entity_ids\n",
        encoding="utf-8",
    )

    # ========================================================================
    # Expected Source1 count
    # ========================================================================

    EXPECTED = 1_732_544

    processed = 0
    total_candidates = 0
    total_matches = 0

    start_time = time.time()

    # ========================================================================
    # Output files
    # ========================================================================

    with (
        MATCH_FILE.open(
            "a",
            encoding="utf-8",
            newline="",
        ) as match_handle,

        CANDIDATE_FILE.open(
            "a",
            encoding="utf-8",
            newline="",
        ) as candidate_handle
    ):

        match_writer = csv.writer(
            match_handle,
            delimiter="\t",
            lineterminator="\n",
        )

        candidate_writer = csv.writer(
            candidate_handle,
            delimiter="\t",
            lineterminator="\n",
        )

        # ====================================================================
        # Persistent worker pool
        # ====================================================================

        print(
            "Starting persistent worker pool..."
        )

        print(
            "Prediction is now running."
        )

        print()

        with mp.Pool(
            processes=WORKERS,
            initializer=init_worker,
            initargs=(
                str(DB_PATH),
                str(MODEL_PATH),
            ),
        ) as pool:

            # ================================================================
            # Stream Source1
            # ================================================================

            for chunk in pd.read_csv(
                SOURCE1,
                sep="\t",
                dtype=str,
                keep_default_na=False,
                chunksize=READ_CHUNK,
            ):

                tasks = (
                    (
                        str(row.entity_id),
                        str(row.business_name),
                        str(row.business_address),
                        str(row.country),
                    )
                    for row in chunk.itertuples(
                        index=False
                    )
                )

                # ============================================================
                # Persistent multiprocessing
                # ============================================================

                for result in pool.imap(
                    process_one,
                    tasks,
                    chunksize=POOL_CHUNKSIZE,
                ):

                    (
                        source1_id,
                        matches,
                        candidate_ids,
                    ) = result

                    # --------------------------------------------------------
                    # Candidate output
                    # --------------------------------------------------------

                    candidate_writer.writerow(
                        [
                            source1_id,
                            ",".join(
                                candidate_ids
                            ),
                        ]
                    )

                    # --------------------------------------------------------
                    # Match output
                    # --------------------------------------------------------

                    match_writer.writerow(
                        [
                            source1_id,
                            ",".join(
                                matches
                            ),
                        ]
                    )

                    processed += 1

                    total_candidates += len(
                        candidate_ids
                    )

                    total_matches += len(
                        matches
                    )

                    # ========================================================
                    # Progress
                    # ========================================================

                    if processed % PRINT_EVERY == 0:

                        match_handle.flush()
                        candidate_handle.flush()

                        elapsed = (
                            time.time()
                            - start_time
                        )

                        rows_per_second = (
                            processed
                            / elapsed
                        )

                        remaining = (
                            EXPECTED
                            - processed
                        )

                        eta_seconds = (
                            remaining
                            / rows_per_second
                        )

                        eta_minutes = (
                            eta_seconds
                            / 60
                        )

                        percent = (
                            processed
                            / EXPECTED
                            * 100
                        )

                        average_candidates = (
                            total_candidates
                            / processed
                        )

                        average_matches = (
                            total_matches
                            / processed
                        )

                        print(
                            f"[{time.strftime('%H:%M:%S')}] "
                            f"{processed:,}/{EXPECTED:,} "
                            f"({percent:.2f}%) | "
                            f"{rows_per_second:.1f} rows/s | "
                            f"ETA {eta_minutes:.1f} min | "
                            f"avg candidates "
                            f"{average_candidates:.2f} | "
                            f"avg matches "
                            f"{average_matches:.3f}",
                            flush=True,
                        )

    # ========================================================================
    # Complete
    # ========================================================================

    elapsed = (
        time.time()
        - start_time
    )

    print()
    print("=" * 70)
    print("PREDICTION COMPLETE")
    print("=" * 70)

    print(
        f"Processed: "
        f"{processed:,}"
    )

    print(
        f"Expected: "
        f"{EXPECTED:,}"
    )

    print(
        f"Total candidates: "
        f"{total_candidates:,}"
    )

    print(
        f"Total matches: "
        f"{total_matches:,}"
    )

    print(
        f"Average candidates/S1: "
        f"{total_candidates / max(processed, 1):.2f}"
    )

    print(
        f"Average matches/S1: "
        f"{total_matches / max(processed, 1):.3f}"
    )

    print(
        f"Elapsed: "
        f"{elapsed / 60:.2f} minutes"
    )

    print()

    print(
        f"matching_results.tsv:"
    )

    print(
        MATCH_FILE
    )

    print()

    print(
        f"candidate_pairs.tsv:"
    )

    print(
        CANDIDATE_FILE
    )

    print()

    if processed == EXPECTED:

        print(
            "SUCCESS: all Source1 rows processed."
        )

    else:

        print(
            "WARNING: output is incomplete!"
        )


# ============================================================================
# WINDOWS ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    mp.freeze_support()

    main()