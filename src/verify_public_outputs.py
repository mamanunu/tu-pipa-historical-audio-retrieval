#!/usr/bin/env python3
"""Validate the aggregate, public results distributed with this repository."""

from __future__ import annotations

import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def read_csv(name: str) -> list[dict[str, str]]:
    with (RESULTS / name).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def close(actual: str, expected: float, tolerance: float = 1e-4) -> bool:
    return abs(float(actual) - expected) <= tolerance


pca = read_csv("pca_summary.csv")
metrics = read_csv("retrieval_metrics.csv")
differences = read_csv("retrieval_representation_differences.csv")
stability = read_csv("retrieval_top5_stability.csv")
aggregation = read_csv("aggregation_sensitivity_summary.csv")
low_energy = read_csv("sensitivity_low_energy.csv")
overlap = read_csv("sensitivity_overlap.csv")

assert len(pca) == 15
assert len(metrics) == 30
assert len(differences) == 9
assert len(stability) == 9
assert len(aggregation) == 9
assert len(low_energy) == 6
assert len(overlap) == 9
assert {row["component"] for row in pca} == {"PC1", "PC2", "PC3", "PC4", "PC5"}
assert {row["representation"] for row in pca} == {"A", "B", "C"}
assert {row["comparison"] for row in differences} == {"A-B", "A-C", "B-C"}
assert all(row["interval_contains_zero"] == "True" for row in differences)
assert {row["comparison"] for row in stability} == {"A-B", "A-C", "B-C"}
assert all(0.0 <= float(row["mean_top5_overlap"]) <= 1.0 for row in stability)

for table in [pca, metrics, differences, stability, aggregation, low_energy, overlap]:
    for row in table:
        for field, value in row.items():
            if value == "":
                assert field == "query_form" and row.get("task") in {
                    "within_recording_segments", "cross_recording_prototype"
                }
    for row in table:
        for value in row.values():
            try:
                number = float(value)
            except ValueError:
                continue
            assert math.isfinite(number), f"Non-finite number in public results: {value}"

assert {row["representation"] for row in low_energy} == {"A", "B", "C"}
assert {row["query_form"] for row in low_energy} == {"Hua Diao", "Bei Gong Diao"}
assert {row["representation"] for row in overlap} == {"A", "B", "C"}
assert {row["task"] for row in overlap} == {"within_recording_segments", "cross_recording_prototype"}

macro_rows = {
    row["representation"]: row
    for row in metrics
    if row["task"] == "cross_recording_prototype" and row["query_form"] == "macro"
}
assert set(macro_rows) == {"A", "B", "C"}
assert close(macro_rows["A"]["p_at_5"], 0.4849)
assert close(macro_rows["B"]["p_at_5"], 0.4796)
assert close(macro_rows["C"]["p_at_5"], 0.4920)
assert all(close(row["random_p_at_5_baseline"], 0.4906) for row in macro_rows.values())

bc_macro = next(
    row for row in stability
    if row["task"] == "cross_recording_prototype"
    and row["comparison"] == "B-C"
    and row["query_form"] == "macro"
)
assert close(bc_macro["mean_top5_overlap"], 0.9231)
assert close(bc_macro["median_top5_overlap"], 1.0)

assert (RESULTS / "figure_corpus_workflow.png").exists()
assert (RESULTS / "figure_pca_abc.png").exists()

print("Public aggregate result checks passed.")
