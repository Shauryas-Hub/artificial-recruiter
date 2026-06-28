"""
src/scorer.py — Final hybrid scoring & normalization.

Formula:
  raw = (W_S * skill_must + W_N * skill_nice)
        * career_fit ^ ALPHA
        * behavioral_fit ^ BETA
        * (1 - LAMBDA * trust_penalty)

Then min-max normalize over ALL survivors (not just top-100).
Tie-break: by candidate_id ascending (deterministic).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# FORMULA CONSTANTS — mirror scoring_config.yaml -> scorer
# ---------------------------------------------------------------------------
W_S    = 0.70    # weight on must-have skill fit
W_N    = 0.30    # weight on nice-to-have skill fit
ALPHA  = 1.00    # career_fit gate exponent
BETA   = 0.80    # behavioral_fit exponent (reduced: avoids over-penalizing
                 # strong ML engineers with private/no GitHub)
LAMBDA = 0.50    # trust penalty multiplier
TOP_N  = 100
SCORE_PRECISION = 4


def compute_raw_scores(df: pd.DataFrame) -> np.ndarray:
    """
    Vectorized raw score computation over the full survivor DataFrame.
    Assumes columns: skill_fit_must, skill_fit_nice, career_fit,
                     behavioral_fit, trust_penalty.
    """
    skill_term = (
        W_S * df["skill_fit_must"].to_numpy(dtype=np.float64)
        + W_N * df["skill_fit_nice"].to_numpy(dtype=np.float64)
    )
    career_gate    = np.clip(df["career_fit"].to_numpy(dtype=np.float64), 0.0, 1.0) ** ALPHA
    behavioral_gate = np.clip(df["behavioral_fit"].to_numpy(dtype=np.float64), 0.0, 1.0) ** BETA
    trust_mult     = np.clip(
        1.0 - LAMBDA * df["trust_penalty"].to_numpy(dtype=np.float64),
        0.0, 1.0,
    )
    return skill_term * career_gate * behavioral_gate * trust_mult


def minmax_normalize(raw: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]. If degenerate (all equal), return 0.5."""
    lo, hi = raw.min(), raw.max()
    if hi - lo < 1e-12:
        return np.full_like(raw, 0.5)
    return (raw - lo) / (hi - lo)


def rank_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame of survivor features, return a sorted DataFrame with:
      - _raw_score: raw score before normalization
      - _norm_score: min-max normalized [0,1]
      - rank: 1..100 for the top-100

    Tie-break: (score DESC, candidate_id ASC) — fully deterministic.
    """
    df = df.copy()
    df["_raw_score"] = compute_raw_scores(df)
    df["_norm_score"] = minmax_normalize(df["_raw_score"].to_numpy())

    # Sort: primary by norm_score DESC, secondary by candidate_id ASC (tie-break)
    df_sorted = df.sort_values(
        by=["_norm_score", "candidate_id"],
        ascending=[False, True],
    ).reset_index(drop=True)

    actual_n = min(TOP_N, len(df_sorted))
    top100 = df_sorted.head(actual_n).copy()
    top100["rank"] = range(1, actual_n + 1)
    top100["score"] = top100["_norm_score"].round(SCORE_PRECISION)

# Final monotonicity check: ensure scores are non-increasing after rounding
    # (rounding can create ties which are fine, but never inversions)
    scores = top100["score"].tolist()
    for i in range(len(scores) - 1):
        if scores[i] < scores[i + 1]:
            scores[i + 1] = scores[i]
    top100["score"] = scores

    # Re-sort after rounding so that equal scores are broken by candidate_id ASC
    # This satisfies the validator's tie-break requirement exactly.
    top100 = top100.sort_values(
        by=["score", "candidate_id"],
        ascending=[False, True],
    ).reset_index(drop=True)
    top100["rank"] = range(1, len(top100) + 1)

    return top100
