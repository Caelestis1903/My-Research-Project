#!/usr/bin/env python3
"""Reproduce the frozen A10/A21 validation AUC analysis from endpoint CSV data.

Only Python's standard library is used. No model, training, network, or repository
module is imported. The default mode authenticates the exact source Git blob;
--skip-source-blob-check is explicitly a local mechanical check.
"""

import argparse
import csv
import hashlib
import math
from pathlib import Path
import statistics


SOURCE_COMMIT = "358d3a465747eda6af1d55acc937759fa651a10b"
SOURCE_BLOB = "d420cdf0043f88b66a6bb886e0c73d72175023a8"
SOURCE_PATH = (
    "artifacts/conditional_history_shared_update_20261122/"
    "tables/A_HISTORY/endpoints.csv"
)
SOURCE_ROWS = 6480
SOURCE_COLUMNS = (
    "dataset", "arm", "seed", "checkpoint_rule", "readout", "split",
    "reuse_status", "epoch", "model_hash", "config", "source_key",
    "source_ref", "trajectory_id", "config_hash", "source_file_sha256",
    "study_id", "cell", "auc", "logloss", "support", "positive", "negative",
    "source_hash", "data_hash", "group_hash", "physical_attempt",
    "owner_generation",
)
SOURCE_CELLS = (
    "d0t0", "d0t1", "d1t0", "d1t1", "d2t0", "d2t1", "macro",
    "domain0", "domain1", "domain2", "task0", "task1", "traffic",
    "task0_pooled", "task1_pooled", "pooled_task_mean", "pooled",
    "pooled_all_entry",
)
DATASETS = ("movielens", "kuairand")
ARMS = ("A10", "A21")
SEEDS = ("2022", "2023", "2024", "2025", "2026")
CHECKPOINT_RULES = ("selected", "E10", "E50")
READOUTS = ("online", "fixed_ema")
SPLIT = "validation"
SIX_CELLS = ("d0t0", "d0t1", "d1t0", "d1t1", "d2t0", "d2t1")
OUTPUT_CELLS = ("macro",) + SIX_CELLS
CONTRASTS = (
    "A21_minus_A10_online",
    "A21_minus_A10_fixed_ema",
    "EMA_gain_A10",
    "EMA_gain_A21",
    "interaction",
)
T_CRITICAL_DF4 = 2.7764451051977987
KEY_COLUMNS = (
    "dataset", "arm", "seed", "checkpoint_rule", "readout", "split", "cell",
)
SEED_FIELDS = (
    "dataset", "seed", "split", "checkpoint_rule", "cell", "contrast",
    "delta_auc", "source_commit", "source_blob",
)
SUMMARY_FIELDS = (
    "dataset", "split", "checkpoint_rule", "cell", "contrast", "n",
    "mean_delta_auc", "sd_delta", "ci95_low", "ci95_high",
    "positive_seeds", "evidence_type",
)


def finite_number(value, field, line):
    """Reject empty, malformed, and non-finite numeric values."""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"row {line}: invalid {field}={value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"row {line}: non-finite {field}={value!r}")
    return number


def count_value(value, field, line):
    """Accept both '27111' and historical CSV values such as '27111.0'."""
    number = finite_number(value, field, line)
    if number < 0 or not number.is_integer():
        raise ValueError(f"row {line}: {field} must be a nonnegative integer")
    return int(number)


def read_endpoints(source, skip_source_blob_check):
    """Read once, authenticate bytes, and require every paired endpoint cell."""
    raw = source.read_bytes()
    actual_blob = hashlib.sha1(
        b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw
    ).hexdigest()
    if not skip_source_blob_check and actual_blob != SOURCE_BLOB:
        raise ValueError(
            f"source Git blob mismatch: expected {SOURCE_BLOB}, got {actual_blob}. "
            f"Use exact bytes of {SOURCE_PATH} at {SOURCE_COMMIT}. "
            "For a transformed local copy, --skip-source-blob-check only permits "
            "a local_mechanical_check; it does not authenticate the source."
        )

    # The frozen source includes a long JSON config field on every row.
    csv.field_size_limit(32 * 1024 * 1024)
    reader = csv.DictReader(raw.decode("utf-8-sig").splitlines(keepends=True))
    header = reader.fieldnames or []
    if len(header) != len(set(header)):
        raise ValueError("duplicate CSV header names")
    required = set(SOURCE_COLUMNS)
    if skip_source_blob_check:
        # The inspected compact copy removes only config, not endpoint data.
        required.remove("config")
    missing_columns = required - set(header)
    extra_columns = set(header) - set(SOURCE_COLUMNS)
    if missing_columns or extra_columns:
        raise ValueError(
            f"CSV schema mismatch: missing={sorted(missing_columns)}, "
            f"unexpected={sorted(extra_columns)}"
        )

    seen_keys = set()
    values = {}
    sample_counts = {}
    row_count = 0
    for row in reader:
        row_count += 1
        line = reader.line_num
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"row {line}: inconsistent CSV field count")
        key = tuple(row[column] for column in KEY_COLUMNS)
        if key in seen_keys:
            raise ValueError(f"row {line}: duplicate endpoint key {key!r}")
        seen_keys.add(key)

        if row["split"] != SPLIT or row["arm"] not in ARMS:
            continue
        for field, allowed in (
            ("dataset", DATASETS), ("seed", SEEDS),
            ("checkpoint_rule", CHECKPOINT_RULES), ("readout", READOUTS),
            ("cell", SOURCE_CELLS),
        ):
            if row[field] not in allowed:
                raise ValueError(
                    f"row {line}: unexpected target {field}={row[field]!r}; "
                    "no target seed or endpoint is silently discarded"
                )
        if row["cell"] not in SIX_CELLS:
            # Recompute macro from six cells; never reuse source aggregate rows.
            continue

        auc = finite_number(row["auc"], "auc", line)
        if not 0 <= auc <= 1:
            raise ValueError(f"row {line}: auc outside [0, 1]")
        logloss = finite_number(row["logloss"], "logloss", line)
        if logloss < 0:
            raise ValueError(f"row {line}: negative logloss")
        support = count_value(row["support"], "support", line)
        positive = count_value(row["positive"], "positive", line)
        negative = count_value(row["negative"], "negative", line)
        if support == 0 or positive == 0 or negative == 0:
            raise ValueError(f"row {line}: AUC requires both classes and support > 0")
        if positive + negative != support:
            raise ValueError(f"row {line}: positive + negative != support")
        count_key = (row["dataset"], row["cell"])
        counts = (support, positive, negative)
        if count_key in sample_counts and sample_counts[count_key] != counts:
            raise ValueError(
                f"row {line}: support/label-count mismatch for {count_key!r}: "
                f"{sample_counts[count_key]!r} != {counts!r}"
            )
        sample_counts[count_key] = counts
        value_key = (
            row["dataset"], row["arm"], row["seed"],
            row["checkpoint_rule"], row["readout"], row["cell"],
        )
        values[value_key] = auc

    expected = {
        (dataset, arm, seed, checkpoint, readout, cell)
        for dataset in DATASETS
        for arm in ARMS
        for seed in SEEDS
        for checkpoint in CHECKPOINT_RULES
        for readout in READOUTS
        for cell in SIX_CELLS
    }
    missing = expected - values.keys()
    unexpected = values.keys() - expected
    if missing or unexpected:
        raise ValueError(
            f"incomplete paired matrix: missing={len(missing)}, "
            f"unexpected={len(unexpected)}; missing examples={sorted(missing)[:8]!r}; "
            f"unexpected examples={sorted(unexpected)[:8]!r}"
        )
    if row_count != SOURCE_ROWS:
        raise ValueError(f"source row count mismatch: expected {SOURCE_ROWS}, got {row_count}")

    for dataset in DATASETS:
        for arm in ARMS:
            for seed in SEEDS:
                for checkpoint in CHECKPOINT_RULES:
                    for readout in READOUTS:
                        endpoint = (dataset, arm, seed, checkpoint, readout)
                        values[endpoint + ("macro",)] = statistics.fmean(
                            values[endpoint + (cell,)] for cell in SIX_CELLS
                        )
    return values, actual_blob, row_count


def build_tables(values, skip_source_blob_check):
    """Compute paired seed deltas before n=5, ddof=1 Student-t summaries."""
    seed_rows = []
    summary_rows = []
    evidence_type = (
        "local_mechanical_check" if skip_source_blob_check else "historical_reanalysis"
    )
    for dataset in DATASETS:
        for checkpoint in CHECKPOINT_RULES:
            for cell in OUTPUT_CELLS:
                paired = {contrast: [] for contrast in CONTRASTS}
                for seed in SEEDS:
                    a10_online = values[(dataset, "A10", seed, checkpoint, "online", cell)]
                    a21_online = values[(dataset, "A21", seed, checkpoint, "online", cell)]
                    a10_ema = values[(dataset, "A10", seed, checkpoint, "fixed_ema", cell)]
                    a21_ema = values[(dataset, "A21", seed, checkpoint, "fixed_ema", cell)]
                    ema_gain_a10 = a10_ema - a10_online
                    ema_gain_a21 = a21_ema - a21_online
                    deltas = (
                        a21_online - a10_online,
                        a21_ema - a10_ema,
                        ema_gain_a10,
                        ema_gain_a21,
                        ema_gain_a21 - ema_gain_a10,
                    )
                    for contrast, delta in zip(CONTRASTS, deltas):
                        paired[contrast].append(delta)
                        seed_rows.append({
                            "dataset": dataset,
                            "seed": seed,
                            "split": SPLIT,
                            "checkpoint_rule": checkpoint,
                            "cell": cell,
                            "contrast": contrast,
                            "delta_auc": delta,
                            # In skip mode these identify the expected historical
                            # source, not authentication of the supplied bytes.
                            "source_commit": SOURCE_COMMIT,
                            "source_blob": SOURCE_BLOB,
                        })
                for contrast in CONTRASTS:
                    deltas = paired[contrast]
                    n = len(deltas)
                    if n != 5:
                        raise ValueError(f"{dataset}/{checkpoint}/{cell}/{contrast}: n={n}, expected 5")
                    mean = statistics.mean(deltas)
                    sd = statistics.stdev(deltas)  # sample SD, divisor n - 1
                    half_width = T_CRITICAL_DF4 * sd / math.sqrt(n)
                    summary_rows.append({
                        "dataset": dataset,
                        "split": SPLIT,
                        "checkpoint_rule": checkpoint,
                        "cell": cell,
                        "contrast": contrast,
                        "n": n,
                        "mean_delta_auc": mean,
                        "sd_delta": sd,
                        "ci95_low": mean - half_width,
                        "ci95_high": mean + half_width,
                        "positive_seeds": sum(delta > 0 for delta in deltas),
                        "evidence_type": evidence_type,
                    })
    if len(seed_rows) != 1050 or len(summary_rows) != 210:
        raise ValueError("unexpected output table size")
    return seed_rows, summary_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="local frozen endpoint CSV")
    parser.add_argument("--output-dir", required=True, type=Path, help="directory for two new CSV files")
    parser.add_argument(
        "--skip-source-blob-check", action="store_true",
        help="mark outputs local_mechanical_check; do not authenticate transformed input bytes",
    )
    args = parser.parse_args()
    try:
        values, actual_blob, row_count = read_endpoints(args.source, args.skip_source_blob_check)
        seed_rows, summary_rows = build_tables(values, args.skip_source_blob_check)
        outputs = (
            (args.output_dir / "validation_interaction_seed.csv", SEED_FIELDS, seed_rows),
            (args.output_dir / "validation_interaction_summary.csv", SUMMARY_FIELDS, summary_rows),
        )
        # Do not overwrite earlier evidence when the command is rerun.
        for path, _, _ in outputs:
            if path.exists():
                raise ValueError(f"output already exists: {path}; choose a new --output-dir")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for path, fields, rows in outputs:
            with path.open("x", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
    except (OSError, UnicodeError, csv.Error, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")

    if args.skip_source_blob_check:
        print("mode=local_mechanical_check; source_authentication=SKIPPED")
        print("source_commit/source_blob columns identify the expected historical source only.")
    else:
        print("mode=historical_reanalysis; source_authentication=EXACT_GIT_BLOB_MATCH")
    print(f"input_git_blob={actual_blob}; source_rows={row_count}; paired_six_cell_rows=720")
    for path, _, rows in outputs:
        print(f"{path}: {len(rows)} rows")


if __name__ == "__main__":
    main()
