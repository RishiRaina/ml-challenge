#!/usr/bin/env python3

"""
FAST prediction for the Business Entity Resolution challenge.

Uses:
    - existing test SQLite index
    - existing LightGBM model
    - existing isotonic calibrator
    - existing tuned threshold

Parallelizes Source1 prediction across persistent worker processes.

IMPORTANT:
Source1 columns are:
    entity_id
    business_name
    business_address
    country
"""

from __future__ import annotations

import csv
import os
import sys
import time
import sqlite3
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Import the existing pipeline functions.
# ---------------------------------------------------------------------------

import business_entity_resolution as ber

try:
    import lightgbm as lgb
except Exception as e:
    raise RuntimeError(
        "LightGBM could not be imported. Make sure the same Python environment "
        "used for training is active."
    ) from e


# ===========================================================================
# CONFIG
# ===========================================================================

ROOT = Path("student_resource")

TEST_SOURCE1 = ROOT / "dataset" / "test" / "test_source1.tsv"

DB_PATH = ROOT / "cache" / "er_index_test.sqlite"
MODEL_PATH = ROOT / "cache" / "lightgbm_matcher.txt"
CALIBRATOR_PATH = ROOT / "cache" / "calibrator.pkl"
THRESHOLD_PATH = ROOT / "cache" / "threshold.json"

OUTPUT_DIR = ROOT / "output"

MATCH_PATH = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_PATH = OUTPUT_DIR / "candidate_pairs.tsv"


# ---------------------------------------------------------------------------
# Performance settings
# ---------------------------------------------------------------------------

# Your CPU has 16 logical processors.
# 8 workers is a reasonable starting point because SQLite + RapidFuzz
# are both CPU/storage intensive.
WORKERS = max(2, min(8, (os.cpu_count() or 4) - 2))

# Number of Source1 rows loaded at a time by the main process.
READ_CHUNK = 16000

# Number of tasks grouped into one multiprocessing scheduling unit.
POOL_CHUNKSIZE = 64

# Flush output periodically so that files visibly grow during prediction.
FLUSH_EVERY = 5000

# The original pipeline uses this final candidate cap.
FINAL_CANDIDATE_MAX = 80

KEEP_NAME = 0.88
KEEP_ADDRESS = 0.88


# ===========================================================================
# GLOBALS INITIALIZED INSIDE EACH WORKER
# ===========================================================================

WORKER_CON = None
WORKER_MODEL = None
WORKER_CALIBRATOR = None
WORKER_THRESHOLD = None


# ===========================================================================
# WORKER INITIALIZATION
# ===========================================================================

def worker_init(db_path, model_path, calibrator_path, threshold_path):
    """
    Runs once per worker.

    Each worker gets:
        - its own SQLite read-only connection
        - its own LightGBM Booster
        - its own calibration object
        - the same threshold
    """

    global WORKER_CON
    global WORKER_MODEL
    global WORKER_CALIBRATOR
    global WORKER_THRESHOLD

    # -----------------------------------------------------------------------
    # SQLite read-only connection.
    #
    # Multiple workers are readers only, so this is safe.
    # -----------------------------------------------------------------------

    uri = f"file:{Path(db_path).resolve().as_posix()}?mode=ro"

    WORKER_CON = sqlite3.connect(
        uri,
        uri=True,
        timeout=60,
        check_same_thread=False,
    )

    # Read-only performance settings.
    WORKER_CON.execute("PRAGMA query_only=ON")
    WORKER_CON.execute("PRAGMA temp_store=MEMORY")
    WORKER_CON.execute("PRAGMA cache_size=-65536")

    # -----------------------------------------------------------------------
    # Load LightGBM model.
    # -----------------------------------------------------------------------

    WORKER_MODEL = lgb.Booster(
        model_file=str(model_path)
    )

    # -----------------------------------------------------------------------
    # Load calibration.
    # -----------------------------------------------------------------------

    WORKER_CALIBRATOR = ber.load_calibrator()

    # -----------------------------------------------------------------------
    # Load threshold.
    # -----------------------------------------------------------------------

    WORKER_THRESHOLD = ber.load_threshold()


# ===========================================================================
# SINGLE SOURCE1 ROW
# ===========================================================================

def process_one(row):
    """
    Process exactly one Source1 record.

    Input:
        (
            source1_entity_id,
            business_name,
            business_address,
            country
        )

    Returns:
        (
            source1_entity_id,
            matches,
            final_candidates
        )
    """

    global WORKER_CON
    global WORKER_MODEL
    global WORKER_CALIBRATOR
    global WORKER_THRESHOLD

    source1_id, raw_name, raw_address, raw_country = row

    # -----------------------------------------------------------------------
    # Normalize exactly as the original pipeline does.
    # -----------------------------------------------------------------------

    name = ber.normalize_name(raw_name)
    address = ber.normalize_address(raw_address)
    country = str(raw_country)

    # -----------------------------------------------------------------------
    # Generate candidates using the existing blocking/index logic.
    # -----------------------------------------------------------------------

    candidates = ber.candidate_rows(
        WORKER_CON,
        country,
        name,
        address,
    )

    if not candidates:
        return (
            source1_id,
            [],
            [],
        )

    # -----------------------------------------------------------------------
    # Apply the same FINAL_CANDIDATE_MAX logic as the original predictor.
    # -----------------------------------------------------------------------

    if len(candidates) > FINAL_CANDIDATE_MAX:

        scored = []

        for c in candidates:

            candidate_name = c[4]
            candidate_address = c[5]

            name_score = max(
                ber.safe_ratio(name, candidate_name),
                ber.W(name, candidate_name),
                ber.token_set(name, candidate_name),
            )

            address_score = max(
                ber.safe_ratio(address, candidate_address),
                ber.W(address, candidate_address),
                ber.token_set(address, candidate_address),
            )

            lexical_score = (
                0.6 * name_score +
                0.4 * address_score
            )

            scored.append(
                (
                    lexical_score,
                    c,
                )
            )

        scored.sort(
            key=lambda x: x[0],
            reverse=True,
        )

        candidates = [
            x[1]
            for x in scored[:FINAL_CANDIDATE_MAX]
        ]

    # -----------------------------------------------------------------------
    # Build model features.
    # -----------------------------------------------------------------------

    features = np.vstack([
        ber.pair_features(
            name,
            address,
            country,
            c[4],
            c[5],
            c[3],
        )
        for c in candidates
    ])

    # -----------------------------------------------------------------------
    # LightGBM prediction.
    # -----------------------------------------------------------------------

    raw_probs = WORKER_MODEL.predict(features)

    # -----------------------------------------------------------------------
    # Isotonic calibration.
    # -----------------------------------------------------------------------

    cal_probs = ber.apply_calibrator(
        WORKER_CALIBRATOR,
        raw_probs,
    )

    # -----------------------------------------------------------------------
    # Candidate IDs.
    #
    # c[1] is the entity ID according to the existing SQLite index schema.
    # -----------------------------------------------------------------------

    pairs = [
        (
            c[1],
            float(prob),
            c,
        )
        for c, prob in zip(
            candidates,
            cal_probs,
        )
    ]

    final_candidates = []

    seen_candidates = set()

    for cid, _, _ in pairs:

        if cid not in seen_candidates:

            seen_candidates.add(cid)

            final_candidates.append(cid)

    # -----------------------------------------------------------------------
    # Final matching decision.
    # -----------------------------------------------------------------------

    matches = []

    seen_matches = set()

    for cid, probability, c in pairs:

        candidate_name = c[4]
        candidate_address = c[5]

        strong_name = bool(
            name
            and candidate_name
            and max(
                ber.safe_ratio(
                    name,
                    candidate_name,
                ),
                ber.W(
                    name,
                    candidate_name,
                ),
                ber.token_set(
                    name,
                    candidate_name,
                ),
            ) >= KEEP_NAME
        )

        strong_address = bool(
            address
            and candidate_address
            and max(
                ber.safe_ratio(
                    address,
                    candidate_address,
                ),
                ber.W(
                    address,
                    candidate_address,
                ),
                ber.token_set(
                    address,
                    candidate_address,
                ),
            ) >= KEEP_ADDRESS
        )

        # Same decision rule as the original predictor.
        if (
            probability >= WORKER_THRESHOLD
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
                        probability,
                    )
                )

    # Highest-confidence matches first.
    matches.sort(
        key=lambda x: x[1],
        reverse=True,
    )

    match_ids = [
        cid
        for cid, _ in matches
    ]

    return (
        source1_id,
        match_ids,
        final_candidates,
    )


# ===========================================================================
# MAIN
# ===========================================================================

def main():

    print("=" * 60)
    print("FAST PREDICTION")
    print("=" * 60)

    print(f"Workers: {WORKERS}")
    print(f"CPU logical processors: {os.cpu_count()}")
    print(f"Source1: {TEST_SOURCE1}")
    print(f"Test index: {DB_PATH}")
    print(f"Model: {MODEL_PATH}")
    print()

    # -----------------------------------------------------------------------
    # Check required files.
    # -----------------------------------------------------------------------

    required_files = [
        TEST_SOURCE1,
        DB_PATH,
        MODEL_PATH,
    ]

    for path in required_files:

        if not path.exists():

            raise FileNotFoundError(
                f"Required file does not exist:\n{path}"
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------------------------
    # Check test index size.
    # -----------------------------------------------------------------------

    print("Checking test index...")

    check_con = sqlite3.connect(
        str(DB_PATH)
    )

    try:

        result = check_con.execute(
            """
            SELECT COUNT(*)
            FROM records
            """
        ).fetchone()

        if result is not None:

            indexed_records = int(result[0])

            print(
                f"Indexed target records: "
                f"{indexed_records:,}"
            )

    finally:

        check_con.close()

    print()

    # -----------------------------------------------------------------------
    # Load Source1 header only to verify columns.
    # -----------------------------------------------------------------------

    header = pd.read_csv(
        TEST_SOURCE1,
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

    actual_columns = list(header.columns)

    print(
        "Source1 columns:",
        actual_columns,
    )

    if actual_columns != expected_columns:

        raise RuntimeError(
            "\nUnexpected Source1 columns.\n"
            f"Expected: {expected_columns}\n"
            f"Found:    {actual_columns}\n"
        )

    print()

    # -----------------------------------------------------------------------
    # Load threshold just for display.
    # -----------------------------------------------------------------------

    threshold = ber.load_threshold()

    calibrator = ber.load_calibrator()

    print(
        f"Using calibrated probabilities: "
        f"{calibrator is not None}"
    )

    print(
        f"Decision threshold: {threshold:.3f}"
    )

    print()

    # -----------------------------------------------------------------------
    # Truncate/create output files.
    # -----------------------------------------------------------------------

    MATCH_PATH.write_text(
        "source1_entity_id\tmatched_entity_ids\n",
        encoding="utf-8",
    )

    CANDIDATE_PATH.write_text(
        "source1_entity_id\tcandidate_entity_ids\n",
        encoding="utf-8",
    )

    # -----------------------------------------------------------------------
    # Timing.
    # -----------------------------------------------------------------------

    start_time = time.time()

    processed = 0
    total_candidates = 0
    total_matches = 0

    # Expected number of Source1 records.
    EXPECTED = 1_732_544

    # -----------------------------------------------------------------------
    # Open output files.
    # -----------------------------------------------------------------------

    with (
        MATCH_PATH.open(
            "a",
            encoding="utf-8",
            newline="",
        ) as match_file,

        CANDIDATE_PATH.open(
            "a",
            encoding="utf-8",
            newline="",
        ) as candidate_file
    ):

        match_writer = csv.writer(
            match_file,
            delimiter="\t",
            lineterminator="\n",
        )

        candidate_writer = csv.writer(
            candidate_file,
            delimiter="\t",
            lineterminator="\n",
        )

        # -------------------------------------------------------------------
        # IMPORTANT:
        # ONE pool for the ENTIRE run.
        #
        # We do NOT recreate workers for every pandas chunk.
        # -------------------------------------------------------------------

        print(
            "Starting persistent worker pool..."
        )

        print(
            "Do not start another prediction process."
        )

        print()

        with mp.Pool(
            processes=WORKERS,
            initializer=worker_init,
            initargs=(
                str(DB_PATH),
                str(MODEL_PATH),
                str(CALIBRATOR_PATH),
                str(THRESHOLD_PATH),
            ),
        ) as pool:

            # ---------------------------------------------------------------
            # Stream Source1.
            # ---------------------------------------------------------------

            for chunk in pd.read_csv(
                TEST_SOURCE1,
                sep="\t",
                dtype=str,
                keep_default_na=False,
                chunksize=READ_CHUNK,
            ):

                # -----------------------------------------------------------
                # Convert pandas rows to compact tuples.
                #
                # IMPORTANT:
                # Actual columns:
                #   entity_id
                #   business_name
                #   business_address
                #   country
                # -----------------------------------------------------------

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

                # -----------------------------------------------------------
                # Persistent multiprocessing.
                # -----------------------------------------------------------

                for result in pool.imap(
                    process_one,
                    tasks,
                    chunksize=POOL_CHUNKSIZE,
                ):

                    (
                        source1_id,
                        match_ids,
                        candidate_ids,
                    ) = result

                    # -------------------------------------------------------
                    # candidate_pairs.tsv
                    #
                    # This MUST contain the exact candidates fed to
                    # the final model.
                    # -------------------------------------------------------

                    candidate_writer.writerow(
                        [
                            source1_id,
                            ",".join(
                                candidate_ids
                            ),
                        ]
                    )

                    # -------------------------------------------------------
                    # matching_results.tsv
                    # -------------------------------------------------------

                    match_writer.writerow(
                        [
                            source1_id,
                            ",".join(
                                match_ids
                            ),
                        ]
                    )

                    processed += 1

                    total_candidates += len(
                        candidate_ids
                    )

                    total_matches += len(
                        match_ids
                    )

                    # -------------------------------------------------------
                    # Progress.
                    # -------------------------------------------------------

                    if processed % FLUSH_EVERY == 0:

                        match_file.flush()
                        candidate_file.flush()

                        elapsed = (
                            time.time()
                            - start_time
                        )

                        rate = (
                            processed
                            / elapsed
                            if elapsed > 0
                            else 0
                        )

                        remaining = (
                            EXPECTED
                            - processed
                        )

                        eta_seconds = (
                            remaining / rate
                            if rate > 0
                            else 0
                        )

                        eta_minutes = (
                            eta_seconds / 60
                        )

                        percent = (
                            processed
                            / EXPECTED
                            * 100
                        )

                        avg_candidates = (
                            total_candidates
                            / processed
                        )

                        avg_matches = (
                            total_matches
                            / processed
                        )

                        print(
                            f"[{time.strftime('%H:%M:%S')}] "
                            f"{processed:,}/{EXPECTED:,} "
                            f"({percent:.2f}%) | "
                            f"{rate:.1f} rows/s | "
                            f"ETA {eta_minutes:.1f} min | "
                            f"avg candidates {avg_candidates:.1f} | "
                            f"avg matches {avg_matches:.2f}",
                            flush=True,
                        )

    # -----------------------------------------------------------------------
    # Finished.
    # -----------------------------------------------------------------------

    elapsed = time.time() - start_time

    print()
    print("=" * 60)
    print("PREDICTION COMPLETE")
    print("=" * 60)

    print(
        f"Processed Source1 rows: "
        f"{processed:,}"
    )

    print(
        f"Total final candidates: "
        f"{total_candidates:,}"
    )

    print(
        f"Total predicted matches: "
        f"{total_matches:,}"
    )

    print(
        f"Average candidates / Source1: "
        f"{total_candidates / max(processed, 1):.2f}"
    )

    print(
        f"Average matches / Source1: "
        f"{total_matches / max(processed, 1):.4f}"
    )

    print(
        f"Elapsed: "
        f"{elapsed / 60:.2f} minutes"
    )

    print()
    print(
        f"matching_results.tsv: "
        f"{MATCH_PATH}"
    )

    print(
        f"candidate_pairs.tsv: "
        f"{CANDIDATE_PATH}"
    )

    print()

    # -----------------------------------------------------------------------
    # Basic row-count sanity check.
    # -----------------------------------------------------------------------

    if processed != EXPECTED:

        print(
            "WARNING: processed row count does not match "
            f"expected {EXPECTED:,}."
        )

    else:

        print(
            "Source1 row count is correct."
        )

    print(
        "Next step: run the official submission validator."
    )


# ===========================================================================
# WINDOWS MULTIPROCESSING ENTRY POINT
# ===========================================================================

if __name__ == "__main__":

    # Required for Windows multiprocessing.
    mp.freeze_support()

    main()