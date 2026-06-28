"""tests/test_scorer.py — Unit tests for src/scorer.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest
from src.scorer import compute_raw_scores, minmax_normalize, rank_candidates

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(n: int = 150, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ids = [f"CAND_{str(i).zfill(7)}" for i in range(1, n + 1)]
    return pd.DataFrame({
        "candidate_id":   ids,
        "current_title":  ["ML Engineer"] * n,
        "skill_fit_must": rng.uniform(0.0, 0.4, n),
        "skill_fit_nice": rng.uniform(0.0, 0.3, n),
        "career_fit":     rng.uniform(0.0, 1.0, n),
        "behavioral_fit": rng.uniform(0.1, 1.0, n),
        "trust_penalty":  rng.uniform(0.0, 0.5, n),
        # Extra columns that rank_candidates passes through
        "years_of_experience_stated": rng.uniform(3, 10, n),
        "location": ["Pune"] * n,
        "country":  ["India"] * n,
        "matched_must_skills":  [""] * n,
        "matched_nice_skills":  [""] * n,
        "shipped_evidence_phrase": [""] * n,
        "availability": rng.uniform(0.1, 1.0, n),
        "engagement":   rng.uniform(0.1, 1.0, n),
        "trust_verification": rng.uniform(0.1, 1.0, n),
        "engineering_signal": rng.uniform(0.1, 1.0, n),
        "applied_ml_at_product_years": rng.uniform(0, 8, n),
        "days_since_last_active": rng.integers(0, 365, n),
        "notice_period_days": rng.integers(0, 120, n),
        "response_rate": rng.uniform(0, 1, n),
        "open_to_work_flag": [True] * n,
        "trap_flags": [""] * n,
        "title_alignment": rng.uniform(0.0, 1.0, n),
        "career_fit": rng.uniform(0.0, 1.0, n),
    })


# ---------------------------------------------------------------------------
# compute_raw_scores
# ---------------------------------------------------------------------------

def test_raw_scores_shape():
    df = _make_df(100)
    scores = compute_raw_scores(df)
    assert scores.shape == (100,)


def test_raw_scores_non_negative():
    df = _make_df(200)
    scores = compute_raw_scores(df)
    assert (scores >= 0).all()


def test_raw_scores_at_most_one():
    """Max possible: skill_term=1.0, career=1.0, behavioral=1.0, trust_mult=1.0 => 1.0"""
    df = pd.DataFrame([{
        "candidate_id": "CAND_0000001",
        "skill_fit_must": 1.0,
        "skill_fit_nice": 1.0,
        "career_fit": 1.0,
        "behavioral_fit": 1.0,
        "trust_penalty": 0.0,
        "current_title": "X",
    }])
    scores = compute_raw_scores(df)
    assert scores[0] <= 1.001  # allow tiny float tolerance


def test_raw_score_zero_trust_penalty():
    """Zero trust_penalty => trust_mult = 1.0, no reduction."""
    df = pd.DataFrame([{
        "candidate_id": "CAND_0000001",
        "skill_fit_must": 0.5,
        "skill_fit_nice": 0.3,
        "career_fit": 0.8,
        "behavioral_fit": 0.7,
        "trust_penalty": 0.0,
        "current_title": "X",
    }])
    score_no_penalty = compute_raw_scores(df)[0]
    df["trust_penalty"] = 0.5
    score_with_penalty = compute_raw_scores(df)[0]
    assert score_no_penalty > score_with_penalty


# ---------------------------------------------------------------------------
# minmax_normalize
# ---------------------------------------------------------------------------

def test_minmax_range():
    raw = np.array([0.1, 0.5, 0.3, 0.9, 0.2])
    norm = minmax_normalize(raw)
    assert abs(norm.min()) < 1e-9
    assert abs(norm.max() - 1.0) < 1e-9


def test_minmax_degenerate_all_equal():
    raw = np.full(10, 0.5)
    norm = minmax_normalize(raw)
    assert (norm == 0.5).all()


# ---------------------------------------------------------------------------
# rank_candidates
# ---------------------------------------------------------------------------

def test_rank_candidates_returns_100():
    df = _make_df(500)
    top = rank_candidates(df)
    assert len(top) == 100


def test_rank_candidates_ranks_1_to_100():
    df = _make_df(200)
    top = rank_candidates(df)
    assert sorted(top["rank"].tolist()) == list(range(1, 101))


def test_rank_candidates_score_non_increasing():
    df = _make_df(300)
    top = rank_candidates(df)
    scores = top.sort_values("rank")["score"].tolist()
    for i in range(len(scores) - 1):
        assert scores[i] >= scores[i + 1], (
            f"Score violation at rank {i+1}: {scores[i]} < {scores[i+1]}"
        )


def test_rank_candidates_unique_ids():
    df = _make_df(200)
    top = rank_candidates(df)
    assert top["candidate_id"].nunique() == 100


def test_rank_candidates_tie_break_by_candidate_id():
    """When scores are equal, lower candidate_id must come first (rank is lower)."""
    df = pd.DataFrame([
        {"candidate_id": "CAND_0000010", "skill_fit_must": 0.3, "skill_fit_nice": 0.2,
         "career_fit": 0.8, "behavioral_fit": 0.7, "trust_penalty": 0.0, "current_title": "X"},
        {"candidate_id": "CAND_0000005", "skill_fit_must": 0.3, "skill_fit_nice": 0.2,
         "career_fit": 0.8, "behavioral_fit": 0.7, "trust_penalty": 0.0, "current_title": "X"},
    ] + [
        {"candidate_id": f"CAND_{str(i).zfill(7)}", "skill_fit_must": 0.0,
         "skill_fit_nice": 0.0, "career_fit": 0.0, "behavioral_fit": 0.1,
         "trust_penalty": 0.0, "current_title": "X"}
        for i in range(100, 200)
    ])
    top = rank_candidates(df)
    r5  = top[top["candidate_id"] == "CAND_0000005"]["rank"].iloc[0]
    r10 = top[top["candidate_id"] == "CAND_0000010"]["rank"].iloc[0]
    assert r5 < r10  # CAND_0000005 < CAND_0000010 alphabetically → lower rank number


def test_rank_candidates_best_gets_rank_1():
    df = _make_df(200, seed=7)
    # Force one candidate to be clearly best
    df.loc[0, "skill_fit_must"] = 1.0
    df.loc[0, "career_fit"]     = 1.0
    df.loc[0, "behavioral_fit"] = 1.0
    df.loc[0, "trust_penalty"]  = 0.0
    df.loc[0, "candidate_id"]   = "CAND_9999999"
    top = rank_candidates(df)
    assert top.iloc[0]["candidate_id"] == "CAND_9999999"
    assert top.iloc[0]["rank"] == 1


def test_rank_with_fewer_than_100_candidates():
    """If fewer than 100 survive, return all of them."""
    df = _make_df(50)
    top = rank_candidates(df)
    assert len(top) == 50
