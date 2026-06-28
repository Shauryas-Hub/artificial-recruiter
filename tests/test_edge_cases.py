"""tests/test_edge_cases.py — Edge cases, failure tests, ranking validation."""
import sys
import math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest
from datetime import date

from src.cleaning import clean_candidate
from src.trap_detector import detect_traps_numeric
from src.behavioral_features import compute_behavioral_features
from src.scorer import compute_raw_scores, rank_candidates


# ---------------------------------------------------------------------------
# Edge: empty skills list
# ---------------------------------------------------------------------------

def test_empty_skills():
    raw = {
        "candidate_id": "CAND_0000001",
        "profile": {
            "anonymized_name": "X", "headline": "X", "summary": "X",
            "location": "Pune", "country": "India",
            "years_of_experience": 5.0, "current_title": "ML Engineer",
            "current_company": "X", "current_company_size": "51-200",
            "current_industry": "Software",
        },
        "career_history": [],
        "education": [],
        "skills": [],
        "redrob_signals": {
            "profile_completeness_score": 30.0,
            "signup_date": "2023-01-01",
            "last_active_date": "2026-01-01",
            "open_to_work_flag": False,
            "profile_views_received_30d": 0,
            "applications_submitted_30d": 0,
            "recruiter_response_rate": 0.0,
            "avg_response_time_hours": 100.0,
            "skill_assessment_scores": {},
            "connection_count": 0,
            "endorsements_received": 0,
            "notice_period_days": 60,
            "expected_salary_range_inr_lpa": {"min": 10.0, "max": 20.0},
            "preferred_work_mode": "hybrid",
            "willing_to_relocate": False,
            "github_activity_score": -1,
            "search_appearance_30d": 0,
            "saved_by_recruiters_30d": 0,
            "interview_completion_rate": 0.5,
            "offer_acceptance_rate": -1,
            "verified_email": False,
            "verified_phone": False,
            "linkedin_connected": False,
        },
    }
    c = clean_candidate(raw)
    assert c.skills == []
    assert c.career_history == []
    trap = detect_traps_numeric(c)
    assert not trap.is_honeypot


# ---------------------------------------------------------------------------
# Edge: github_activity_score = -1 (no github)
# ---------------------------------------------------------------------------

def test_behavioral_github_minus1_neutral():
    """github = None (sentinel -1) must not penalize the score — treated as 0.5."""
    raw = {
        "candidate_id": "CAND_0000002",
        "profile": {
            "anonymized_name": "X", "headline": "X", "summary": "X",
            "location": "Hyderabad", "country": "India",
            "years_of_experience": 6.0, "current_title": "AI Engineer",
            "current_company": "Swiggy", "current_company_size": "5001-10000",
            "current_industry": "Food Delivery",
        },
        "career_history": [],
        "education": [],
        "skills": [],
        "redrob_signals": {
            "profile_completeness_score": 80.0,
            "signup_date": "2023-01-01",
            "last_active_date": "2026-05-01",
            "open_to_work_flag": True,
            "profile_views_received_30d": 20,
            "applications_submitted_30d": 3,
            "recruiter_response_rate": 0.8,
            "avg_response_time_hours": 12.0,
            "skill_assessment_scores": {},
            "connection_count": 300,
            "endorsements_received": 30,
            "notice_period_days": 30,
            "expected_salary_range_inr_lpa": {"min": 25.0, "max": 50.0},
            "preferred_work_mode": "hybrid",
            "willing_to_relocate": True,
            "github_activity_score": -1,   # no GitHub
            "search_appearance_30d": 100,
            "saved_by_recruiters_30d": 10,
            "interview_completion_rate": 0.9,
            "offer_acceptance_rate": -1,
            "verified_email": True,
            "verified_phone": True,
            "linkedin_connected": True,
        },
    }
    c = clean_candidate(raw)
    assert c.signals.github_activity_score is None

    beh = compute_behavioral_features(c, date(2026, 5, 27))
    # behavioral_fit must be > 0 even with no GitHub
    assert beh.behavioral_fit > 0.3
    # engineering_signal with github=None: gh_score=0.5, pvr for 100 views is positive
    assert 0.0 < beh.engineering_signal <= 1.0


# ---------------------------------------------------------------------------
# Edge: zero experience years
# ---------------------------------------------------------------------------

def test_zero_experience_years():
    raw = {
        "candidate_id": "CAND_0000003",
        "profile": {
            "anonymized_name": "X", "headline": "X", "summary": "X",
            "location": "Pune", "country": "India",
            "years_of_experience": 0.0, "current_title": "Intern",
            "current_company": "X", "current_company_size": "1-10",
            "current_industry": "Software",
        },
        "career_history": [],
        "education": [],
        "skills": [],
        "redrob_signals": {
            "profile_completeness_score": 20.0, "signup_date": "2025-01-01",
            "last_active_date": "2026-01-01", "open_to_work_flag": True,
            "profile_views_received_30d": 0, "applications_submitted_30d": 0,
            "recruiter_response_rate": 0.0, "avg_response_time_hours": 200.0,
            "skill_assessment_scores": {}, "connection_count": 0,
            "endorsements_received": 0, "notice_period_days": 0,
            "expected_salary_range_inr_lpa": {"min": 5.0, "max": 10.0},
            "preferred_work_mode": "onsite", "willing_to_relocate": False,
            "github_activity_score": -1, "search_appearance_30d": 0,
            "saved_by_recruiters_30d": 0, "interview_completion_rate": 0.0,
            "offer_acceptance_rate": -1, "verified_email": False,
            "verified_phone": False, "linkedin_connected": False,
        },
    }
    c = clean_candidate(raw)
    assert c.years_of_experience_stated == 0.0
    trap = detect_traps_numeric(c)
    assert not trap.is_honeypot


# ---------------------------------------------------------------------------
# Edge: degenerate scorer (all candidates identical scores)
# ---------------------------------------------------------------------------

def test_scorer_degenerate_all_equal():
    """All candidates with same features: scores normalize to 0.5, ranks by ID."""
    df = pd.DataFrame([
        {"candidate_id": f"CAND_{str(i).zfill(7)}", "skill_fit_must": 0.2,
         "skill_fit_nice": 0.1, "career_fit": 0.5, "behavioral_fit": 0.5,
         "trust_penalty": 0.0, "current_title": "X",
         "matched_must_skills": "", "matched_nice_skills": "",
         "shipped_evidence_phrase": "", "availability": 0.5,
         "engagement": 0.5, "trust_verification": 0.5,
         "engineering_signal": 0.5, "applied_ml_at_product_years": 3.0,
         "days_since_last_active": 30, "notice_period_days": 30,
         "response_rate": 0.5, "open_to_work_flag": True,
         "trap_flags": "", "title_alignment": 0.5,
         "years_of_experience_stated": 5.0, "location": "Pune",
         "country": "India"}
        for i in range(1, 151)
    ])
    top = rank_candidates(df)
    assert len(top) == 100
    # When all equal, must be sorted by candidate_id ascending
    assert top.iloc[0]["candidate_id"] == "CAND_0000001"
    assert top.iloc[99]["candidate_id"] == "CAND_0000100"


# ---------------------------------------------------------------------------
# Edge: candidates exactly at 100 count
# ---------------------------------------------------------------------------

def test_scorer_exactly_100_candidates():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "candidate_id":   [f"CAND_{str(i).zfill(7)}" for i in range(1, 101)],
        "current_title":  ["X"] * 100,
        "skill_fit_must": rng.uniform(0, 0.4, 100),
        "skill_fit_nice": rng.uniform(0, 0.2, 100),
        "career_fit":     rng.uniform(0, 1.0, 100),
        "behavioral_fit": rng.uniform(0.1, 1.0, 100),
        "trust_penalty":  rng.uniform(0, 0.3, 100),
        "matched_must_skills": [""] * 100,
        "matched_nice_skills": [""] * 100,
        "shipped_evidence_phrase": [""] * 100,
        "availability":   rng.uniform(0.1, 1.0, 100),
        "engagement":     rng.uniform(0.1, 1.0, 100),
        "trust_verification": rng.uniform(0.1, 1.0, 100),
        "engineering_signal": rng.uniform(0.1, 1.0, 100),
        "applied_ml_at_product_years": rng.uniform(0, 8, 100),
        "days_since_last_active": rng.integers(0, 200, 100),
        "notice_period_days": rng.integers(0, 90, 100),
        "response_rate":  rng.uniform(0, 1, 100),
        "open_to_work_flag": [True] * 100,
        "trap_flags":     [""] * 100,
        "title_alignment": rng.uniform(0, 1, 100),
        "years_of_experience_stated": rng.uniform(3, 10, 100),
        "location":       ["Pune"] * 100,
        "country":        ["India"] * 100,
    })
    top = rank_candidates(df)
    assert len(top) == 100
    assert sorted(top["rank"].tolist()) == list(range(1, 101))


# ---------------------------------------------------------------------------
# Ranking quality: high-signal candidate beats low-signal
# ---------------------------------------------------------------------------

def test_high_signal_beats_low_signal():
    """A candidate with perfect signals must rank above one with poor signals."""
    df = pd.DataFrame([
        {
            "candidate_id": "CAND_0000001",
            "skill_fit_must": 0.35, "skill_fit_nice": 0.25,
            "career_fit": 0.90, "behavioral_fit": 0.85, "trust_penalty": 0.0,
            "current_title": "ML Engineer", "matched_must_skills": "embeddings~FAISS",
            "matched_nice_skills": "", "shipped_evidence_phrase": "built a ranking system",
            "availability": 0.9, "engagement": 0.8, "trust_verification": 0.9,
            "engineering_signal": 0.7, "applied_ml_at_product_years": 6.0,
            "days_since_last_active": 5, "notice_period_days": 30,
            "response_rate": 0.8, "open_to_work_flag": True, "trap_flags": "",
            "title_alignment": 0.9, "years_of_experience_stated": 7.0,
            "location": "Pune", "country": "India",
        },
        {
            "candidate_id": "CAND_0000002",
            "skill_fit_must": 0.01, "skill_fit_nice": 0.01,
            "career_fit": 0.10, "behavioral_fit": 0.20, "trust_penalty": 0.3,
            "current_title": "Marketing Manager", "matched_must_skills": "",
            "matched_nice_skills": "", "shipped_evidence_phrase": "",
            "availability": 0.2, "engagement": 0.1, "trust_verification": 0.2,
            "engineering_signal": 0.1, "applied_ml_at_product_years": 0.0,
            "days_since_last_active": 300, "notice_period_days": 90,
            "response_rate": 0.1, "open_to_work_flag": False, "trap_flags": "",
            "title_alignment": 0.1, "years_of_experience_stated": 2.0,
            "location": "Chennai", "country": "India",
        },
    ] + [
        {
            "candidate_id": f"CAND_{str(i).zfill(7)}", "skill_fit_must": 0.05,
            "skill_fit_nice": 0.02, "career_fit": 0.3, "behavioral_fit": 0.4,
            "trust_penalty": 0.1, "current_title": "X", "matched_must_skills": "",
            "matched_nice_skills": "", "shipped_evidence_phrase": "",
            "availability": 0.4, "engagement": 0.3, "trust_verification": 0.5,
            "engineering_signal": 0.3, "applied_ml_at_product_years": 1.0,
            "days_since_last_active": 60, "notice_period_days": 60,
            "response_rate": 0.4, "open_to_work_flag": True, "trap_flags": "",
            "title_alignment": 0.3, "years_of_experience_stated": 4.0,
            "location": "Delhi", "country": "India",
        }
        for i in range(100, 200)
    ])
    top = rank_candidates(df)
    rank_good = top[top["candidate_id"] == "CAND_0000001"]["rank"].iloc[0]
    assert rank_good == 1, f"High-signal candidate must be rank 1, got {rank_good}"
    # Low-signal candidate (career_fit=0.10, behavioral=0.20, trust_penalty=0.3)
    # should not appear in top-100 at all when 100 filler candidates outscore it
    bad_in_top = top[top["candidate_id"] == "CAND_0000002"]
    if len(bad_in_top) > 0:
        rank_bad = bad_in_top["rank"].iloc[0]
        assert rank_good < rank_bad, (
            f"High-signal (rank {rank_good}) must beat low-signal (rank {rank_bad})"
        )
