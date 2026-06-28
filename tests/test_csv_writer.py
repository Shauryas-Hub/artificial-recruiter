"""tests/test_csv_writer.py — Unit tests for src/csv_writer.py and src/validate_local.py"""
import sys
import os
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import pytest
from src.csv_writer import write_submission, SubmissionValidationError
from src.validate_local import validate_submission


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_valid_df() -> pd.DataFrame:
    import numpy as np
    rng = np.random.default_rng(0)
    scores = np.linspace(1.0, 0.01, 100)
    return pd.DataFrame({
        "candidate_id": [f"CAND_{str(i).zfill(7)}" for i in range(1, 101)],
        "rank":         list(range(1, 101)),
        "score":        [round(float(s), 4) for s in scores],
        "reasoning":    [f"Reason for candidate {i}" for i in range(1, 101)],
    })


# ---------------------------------------------------------------------------
# write_submission
# ---------------------------------------------------------------------------

def test_write_valid_submission():
    df = _make_valid_df()
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = f.name
    try:
        write_submission(df, path)
        result = pd.read_csv(path)
        assert len(result) == 100
        assert list(result.columns) == ["candidate_id", "rank", "score", "reasoning"]
    finally:
        os.unlink(path)


def test_write_submission_utf8():
    df = _make_valid_df()
    df.loc[0, "reasoning"] = "Café résumé — unicode test ✓"
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = f.name
    try:
        write_submission(df, path)
        with open(path, "r", encoding="utf-8") as fp:
            content = fp.read()
        assert "Café" in content
    finally:
        os.unlink(path)


def test_write_rejects_wrong_row_count():
    df = _make_valid_df().head(99)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = f.name
    try:
        with pytest.raises(SubmissionValidationError, match="99"):
            write_submission(df, path)
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_write_rejects_duplicate_ids():
    df = _make_valid_df()
    df.loc[1, "candidate_id"] = df.loc[0, "candidate_id"]  # duplicate
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = f.name
    try:
        with pytest.raises(SubmissionValidationError):
            write_submission(df, path)
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_write_rejects_non_monotonic_scores():
    df = _make_valid_df()
    # Swap scores so rank 1 < rank 2
    df.loc[df["rank"] == 1, "score"] = 0.1
    df.loc[df["rank"] == 2, "score"] = 0.9
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = f.name
    try:
        with pytest.raises(SubmissionValidationError):
            write_submission(df, path)
    finally:
        if os.path.exists(path):
            os.unlink(path)


# ---------------------------------------------------------------------------
# validate_local
# ---------------------------------------------------------------------------

def test_validate_submission_passes_valid():
    df = _make_valid_df()
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", encoding="utf-8") as f:
        path = f.name
        df.to_csv(f, index=False)
    try:
        errors = validate_submission(path)
        assert errors == [], f"Unexpected errors: {errors}"
    finally:
        os.unlink(path)


def test_validate_submission_detects_wrong_extension():
    errors = validate_submission("submission.json")
    assert any("csv" in e.lower() for e in errors)


def test_validate_submission_detects_duplicate_rank():
    df = _make_valid_df()
    df.loc[1, "rank"] = 1  # duplicate rank 1
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", encoding="utf-8") as f:
        path = f.name
        df.to_csv(f, index=False)
    try:
        errors = validate_submission(path)
        assert any("rank" in e.lower() or "missing" in e.lower() for e in errors)
    finally:
        os.unlink(path)


def test_validate_detects_honeypot_rate_over_10pct():
    df = _make_valid_df()
    hp_ids = set(f"CAND_{str(i).zfill(7)}" for i in range(1, 12))  # 11 honeypots in top-100
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", encoding="utf-8") as f:
        path = f.name
        df.to_csv(f, index=False)
    try:
        errors = validate_submission(path, honeypot_ids=hp_ids)
        assert any("honeypot" in e.lower() or "disqualification" in e.lower() for e in errors)
    finally:
        os.unlink(path)


def test_validate_ok_honeypot_rate_under_10pct():
    df = _make_valid_df()
    hp_ids = set(f"CAND_{str(i).zfill(7)}" for i in range(1, 6))  # 5 = 5% < 10%
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", encoding="utf-8") as f:
        path = f.name
        df.to_csv(f, index=False)
    try:
        errors = validate_submission(path, honeypot_ids=hp_ids)
        assert not any("disqualification" in e.lower() for e in errors)
    finally:
        os.unlink(path)
