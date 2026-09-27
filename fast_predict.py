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

SOURCE1_TOTAL = 1_732_544

# Your i7-13620H has 16 logical processors.
# Start conservatively because every worker also does RapidFuzz.
WORKERS = max(2, min(8, (os.cpu_count() or 4) - 2))

MODEL = None
CALIBRATOR = None
THRESHOLD = None
CON = None


def init_worker():
    """
    Runs once per worker process.
    """
    global MODEL
    global CALIBRATOR
    global THRESHOLD
    global CON

    MODEL = lgb.Booster(
        model_file=str(MODEL_PATH)
    )

    CALIBRATOR = ber.load_calibrator()
    THRESHOLD = ber.load_threshold()

    # Every worker gets its own read-only SQLite connection.
    CON = sqlite3.connect(
        f"file:{DB_PATH}?mode=ro",
        uri=True,
        timeout=60,
    )

    CON.execute("PRAGMA query_only=ON")

    # 64 MB SQLite page cache per worker.
    CON.execute("PRAGMA cache_size=-65536")


def process_row(job):
    """
    Process one Source-1 entity.
    """

    global MODEL
    global CALIBRATOR
    global THRESHOLD
    global CON

    s1id, country, name, addr = job

    candidates = ber.candidate_rows(
        CON,
        country,
        name,
        addr,
    )

    if not candidates:
        return (
            s1id,
            [],
            [],
        )

    # Build the exact same 22-feature representation
    # used by the trained model.
    features = np.asarray(
        [
            ber.pair_features(
                name,
                addr,
                country,
                candidate[4],
                candidate[5],
                candidate[3],
            )
            for candidate in candidates
        ],
        dtype=np.float32,
    )

    raw_probs = MODEL.predict(
        features
    )

    cal_probs = ber.apply_calibrator(
        CALIBRATOR,
        raw_probs,
    )

    matched = []

    for candidate, probability in zip(
        candidates,
        cal_probs,
    ):
        candidate_name = candidate[4]
        candidate_addr = candidate[5]

        strong_name = (
            bool(
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
                )
                >= ber.KEEP_NAME
            )
        )

        strong_addr = (
            bool(
                addr
                and candidate_addr
                and max(
                    ber.safe_ratio(
                        addr,
                        candidate_addr,
                    ),
                    ber.W(
                        addr,
                        candidate_addr,
                    ),
                    ber.token_set(
                        addr,
                        candidate_addr,
                    ),
                )
                >= ber.KEEP_ADDRESS
            )
        )

        if (
            probability >= THRESHOLD
            or (strong_name and strong_addr)
        ):
            matched.append(
                candidate[1]
            )

    candidate_ids = [
        candidate[1]
        for candidate in candidates
    ]

    return (
        s1id,
        matched,
        candidate_ids,
    )


def main():

    print("=" * 60)
    print("FAST PREDICTION")
    print("=" * 60)

    print(
        f"Workers: {WORKERS}"
    )

    print(
        f"CPU logical processors: "
        f"{os.cpu_count()}"
    )

    print(
        f"Test index: {DB_PATH}"
    )

    print(
        f"Model: {MODEL_PATH}"
    )

    print()

    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Missing test index: {DB_PATH}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing model: {MODEL_PATH}"
        )

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    match_path = (
        OUTPUT / "matching_results.tsv"
    )

    candidate_path = (
        OUTPUT / "candidate_pairs.tsv"
    )

    # Start fresh.
    match_path.write_text(
        "source1_entity_id\tmatched_entity_ids\n",
        encoding="utf-8",
    )

    candidate_path.write_text(
        "source1_entity_id\tcandidate_entity_ids\n",
        encoding="utf-8",
    )

    source1_path = (
        TEST / "test_source1.tsv"
    )

    start_time = time.time()

    processed = 0

    # Keep the pool alive for the ENTIRE prediction.
    with mp.Pool(
        processes=WORKERS,
        initializer=init_worker,
    ) as pool:

        with (
            match_path.open(
                "a",
                encoding="utf-8",
                newline="",
            ) as match_file,

            candidate_path.open(
                "a",
                encoding="utf-8",
                newline="",
            ) as candidate_file,
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

            # Stream Source-1.
            #
            # chunksize here controls how many jobs are
            # dispatched to workers at a time.
            source = pd.read_csv(
                source1_path,
                sep="\t",
                dtype=str,
                keep_default_na=False,
                chunksize=16_000,
            )

            for chunk in source:

                jobs = []

                for row in chunk.itertuples(
                    index=False
                ):

                    s1id = str(
                        row.entity_id
                    )

                    country = str(
                        row.country
                    )

                    name = (
                        ber.normalize_name(
                            row.business_name
                        )
                    )

                    addr = (
                        ber.normalize_address(
                            row.address
                        )
                    )

                    jobs.append(
                        (
                            s1id,
                            country,
                            name,
                            addr,
                        )
                    )

                # map() preserves Source-1 order.
                #
                # chunksize=64 reduces multiprocessing
                # communication overhead.
                results = pool.imap(
                    process_row,
                    jobs,
                    chunksize=64,
                )

                for (
                    s1id,
                    matched,
                    candidates,
                ) in results:

                    match_writer.writerow(
                        [
                            s1id,
                            ",".join(matched),
                        ]
                    )

                    candidate_writer.writerow(
                        [
                            s1id,
                            ",".join(candidates),
                        ]
                    )

                    processed += 1

                    # Flush periodically so the output
                    # files visibly grow during the run.
                    if (
                        processed % 5_000
                        == 0
                    ):

                        match_file.flush()
                        candidate_file.flush()

                        elapsed = (
                            time.time()
                            - start_time
                        )

                        rate = (
                            processed
                            / max(
                                elapsed,
                                0.001,
                            )
                        )

                        remaining = (
                            SOURCE1_TOTAL
                            - processed
                        )

                        eta_seconds = (
                            remaining
                            / max(
                                rate,
                                0.001,
                            )
                        )

                        percent = (
                            100.0
                            * processed
                            / SOURCE1_TOTAL
                        )

                        print(
                            f"[{time.strftime('%H:%M:%S')}] "
                            f"{processed:,}/{SOURCE1_TOTAL:,} "
                            f"({percent:.2f}%) | "
                            f"{rate:.1f} rows/s | "
                            f"ETA "
                            f"{eta_seconds / 60:.1f} min",
                            flush=True,
                        )

    elapsed = (
        time.time()
        - start_time
    )

    print()
    print("=" * 60)
    print("PREDICTION COMPLETE")
    print("=" * 60)

    print(
        f"Processed: "
        f"{processed:,}"
    )

    print(
        f"Time: "
        f"{elapsed / 60:.2f} minutes"
    )

    print(
        f"Matching results:"
        f"\n  {match_path}"
    )

    print(
        f"Candidate pairs:"
        f"\n  {candidate_path}"
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()