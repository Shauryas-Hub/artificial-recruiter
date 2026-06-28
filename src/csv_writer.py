"""
src/csv_writer.py — Write and validate the final submission CSV.

Spec requirements (submission_spec.md):
  - Header: candidate_id,rank,score,reasoning
  - Exactly 100 data rows
  - Ranks 1-100 each exactly once
  - Candidate IDs unique
  - Score non-increasing (float)
  - UTF-8 encoding
"""
from __future__ import annotations
import csv
import io
import os
import re
from pathlib import Path
import pandas as pd

REQUIRED_HEADER = ["candidate_id", "rank", "score", "reasoning"]
CANDIDATE_ID_PATTERN = re.compile(r"^CAND_[0-9]{7}$")


class SubmissionValidationError(Exception):
    pass


def _validate_dataframe(df: pd.DataFrame) -> None:
    """Pre-write validation. Raises SubmissionValidationError on any violation."""
    errors = []

    # Column check
    missing_cols = [c for c in REQUIRED_HEADER if c not in df.columns]
    if missing_cols:
        errors.append(f"Missing columns: {missing_cols}")

    if len(df) != 100:
        errors.append(f"Expected 100 rows, got {len(df)}")

    # Rank uniqueness
    ranks = df["rank"].tolist()
    if sorted(ranks) != list(range(1, 101)):
        errors.append("Ranks must be exactly 1..100, each once")

    # Candidate ID uniqueness & format
    ids = df["candidate_id"].tolist()
    if len(set(ids)) != len(ids):
        errors.append("Duplicate candidate_ids found")
    bad_ids = [i for i in ids if not CANDIDATE_ID_PATTERN.match(str(i))]
    if bad_ids:
        errors.append(f"Invalid candidate_id format: {bad_ids[:5]}")

    # Score monotonicity
    sorted_df = df.sort_values("rank")
    scores = sorted_df["score"].tolist()
    for i in range(len(scores) - 1):
        if scores[i] < scores[i + 1]:
            errors.append(
                f"Score not non-increasing at rank {i+1} ({scores[i]}) -> rank {i+2} ({scores[i+1]})"
            )
            break

    # Reasoning non-empty
    empty_reasoning = df[df["reasoning"].isna() | (df["reasoning"].astype(str).str.strip() == "")]
    if len(empty_reasoning) > 0:
        errors.append(f"{len(empty_reasoning)} rows have empty reasoning")

    if errors:
        raise SubmissionValidationError("\n".join(errors))


def write_submission(df: pd.DataFrame, out_path: str) -> None:
    """
    Write a validated submission CSV.
    Selects and orders the correct 4 columns, validates, then writes UTF-8.
    """
    out_df = df[REQUIRED_HEADER].copy()
    out_df["rank"]  = out_df["rank"].astype(int)
    out_df["score"] = out_df["score"].astype(float)
    out_df = out_df.sort_values("rank").reset_index(drop=True)

    # Pre-write validation
    _validate_dataframe(out_df)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(
        out_path,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
        quoting=csv.QUOTE_MINIMAL,
    )
    print(f"  Written: {out_path} ({len(out_df)} rows, UTF-8)")
