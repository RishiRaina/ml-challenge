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


ROOT = Path("student_resource")
TEST = ROOT / "dataset" / "test"
OUTPUT = ROOT / "output"
CACHE = ROOT / "cache"

DB_PATH = CACHE / "er_index_test.sqlite"
MODEL_PATH = CACHE / "lightgbm_matcher.txt"

# Use several workers, but don't hammer the SQLite disk with too many.
WORKERS = max(2, min(6, (os.cpu_count() or 4) - 1))

MODEL = None
CALIBRATOR = None
THRESHOLD = None
CON = None


def init_worker():
    global MODEL, CALIBRATOR, THRESHOLD, CON

    MODEL = lgb.Booster(model_file=str(MODEL_PATH))
    CALIBRATOR = ber.load_calibrator()
    THRESHOLD = ber.load_threshold()

    # Each worker gets its own read-only SQLite connection.
    CON = sqlite3.connect(
        f"file:{DB_PATH}?mode=ro",
        uri=True,
        timeout=60,
    )

    # SQLite performance settings for read workload.
    CON.execute("PRAGMA query_only=ON")
    CON.execute("PRAGMA cache_size=-65536")


def process_row(item):
    global MODEL, CALIBRATOR, THRESHOLD, CON

    s1id, country, name, addr = item

    candidates = ber.candidate_rows(
        CON,
        country,
        name,
        addr,
    )

    if not candidates:
        return s1id, [], []

    features = np.asarray(
        [
            ber.pair_features(
                name,
                addr,
                country,
                c[4],
                c[5],
                c[3],
            )
            for c in candidates
        ],
        dtype=np.float32,
    )

    raw_probs = MODEL.predict(features)

    cal_probs = ber.apply_calibrator(
        CALIBRATOR,
        raw_probs,
    )

    matched = []

    for c, p in zip(candidates, cal_probs):
        strong_name = (
            bool(
                name
                and c[4]
                and max(
                    ber.safe_ratio(name, c[4]),
                    ber.W(name, c[4]),
                    ber.token_set(name, c[4]),
                )
                >= ber.KEEP_NAME
            )
        )

        strong_addr = (
            bool(
                addr
                and c[5]
                and max(
                    ber.safe_ratio(addr, c[5]),
                    ber.W(addr, c[5]),
                    ber.token_set(addr, c[5]),
                )
                >= ber.KEEP_ADDRESS
            )
        )

        if p >= THRESHOLD or (strong_name and strong_addr):
            matched.append(c[1])

    candidate_ids = [c[1] for c in candidates]

    return s1id, matched, candidate_ids


def main():
    print(f"Workers: {WORKERS}")
    print(f"Test index: {DB_PATH}")
    print(f"Model: {MODEL_PATH}")

    OUTPUT.mkdir(parents=True, exist_ok=True)

    match_path = OUTPUT / "matching_results.tsv"
    candidate_path = OUTPUT / "candidate_pairs.tsv"

    # Start fresh.
    match_path.write_text(
        "source1_entity_id\tmatched_entity_ids\n",
        encoding="utf-8",
    )

    candidate_path.write_text(
        "source1_entity_id\tcandidate_entity_ids\n",
        encoding="utf-8",
    )

    source1 = TEST / "test_source1.tsv"

    total = 0
    start = time.time()

    with (
        match_path.open("a", encoding="utf-8", newline="") as mf,
        candidate_path.open("a", encoding="utf-8", newline="") as cf,
    ):
        mw = csv.writer(
            mf,
            delimiter="\t",
            lineterminator="\n",
        )

        cw = csv.writer(
            cf,
            delimiter="\t",
            lineterminator="\n",
        )

        # Stream Source-1 rather than loading all 1.7M rows.
        for chunk in pd.read_csv(
            source1,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            chunksize=8000,
        ):
            jobs = []

            for row in chunk.itertuples(index=False):
                s1id = str(row.entity_id)
                country = str(row.country)

                name = ber.normalize_name(
                    row.business_name
                )

                addr = ber.normalize_address(
                    row.address
                )

                jobs.append(
                    (
                        s1id,
                        country,
                        name,
                        addr,
                    )
                )

            with mp.Pool(
                processes=WORKERS,
                initializer=init_worker,
            ) as pool:
                # Moderate chunks keep memory reasonable while
                # allowing multiple records to run concurrently.
                for s1id, matched, candidates in pool.imap(
                    process_row,
                    jobs,
                    chunksize=32,
                ):
                    mw.writerow(
                        [
                            s1id,
                            ",".join(matched),
                        ]
                    )

                    cw.writerow(
                        [
                            s1id,
                            ",".join(candidates),
                        ]
                    )

                    total += 1

                    if total % 10000 == 0:
                        mf.flush()
                        cf.flush()

                        elapsed = time.time() - start
                        rate = total / max(elapsed, 1)

                        remaining = (
                            1_732_544 - total
                        )

                        eta = remaining / max(rate, 0.001)

                        print(
                            f"Processed {total:,} / 1,732,544 "
                            f"({100 * total / 1_732_544:.2f}%) "
                            f"| {rate:.1f} rows/s "
                            f"| ETA {eta / 60:.1f} min",
                            flush=True,
                        )

    elapsed = time.time() - start

    print()
    print("========================================")
    print("PREDICTION COMPLETE")
    print("========================================")
    print(f"Source-1 records: {total:,}")
    print(f"Time: {elapsed / 60:.2f} minutes")
    print(f"Matches: {match_path}")
    print(f"Candidates: {candidate_path}")


if __name__ == "__main__":
    mp.freeze_support()
    main()