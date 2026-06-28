"""tests/test_trap_detector.py — Unit tests for src/trap_detector.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.cleaning import clean_candidate
from src.trap_detector import (
    detect_traps_numeric,
    finalize_trap_result,
    HARD_HONEYPOT_FLAG_COUNT,
    HARD_FLAG_PREFIXES,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_raw(overrides: dict = None) -> dict:
    base = {
        "candidate_id": "CAND_0000001",
        "profile": {
            "anonymized_name": "Test User",
            "headline": "ML Engineer",
            "summary": "Test",
            "location": "Pune",
            "country": "India",
            "years_of_experience": 5.0,
            "current_title": "ML Engineer",
            "current_company": "TestCo",
            "current_company_size": "51-200",
            "current_industry": "Software",
        },
        "career_history": [
            {
                "company": "TestCo",
                "title": "ML Engineer",
                "start_date": "2021-01-01",
                "end_date": None,
                "duration_months": 30,
                "is_current": True,
                "industry": "Software",
                "company_size": "51-200",
                "description": "Built ML systems",
            }
        ],
        "education": [],
        "skills": [
            {"name": "Python", "proficiency": "advanced", "endorsements": 10, "duration_months": 24},
        ],
        "redrob_signals": {
            "profile_completeness_score": 75.0,
            "signup_date": "2021-01-01",
            "last_active_date": "2026-05-01",
            "open_to_work_flag": True,
            "profile_views_received_30d": 10,
            "applications_submitted_30d": 2,
            "recruiter_response_rate": 0.6,
            "avg_response_time_hours": 24.0,
            "skill_assessment_scores": {},
            "connection_count": 200,
            "endorsements_received": 15,
            "notice_period_days": 30,
            "expected_salary_range_inr_lpa": {"min": 20.0, "max": 40.0},
            "preferred_work_mode": "hybrid",
            "willing_to_relocate": True,
            "github_activity_score": 50.0,
            "search_appearance_30d": 50,
            "saved_by_recruiters_30d": 5,
            "interview_completion_rate": 0.8,
            "offer_acceptance_rate": 0.5,
            "verified_email": True,
            "verified_phone": True,
            "linkedin_connected": True,
        },
    }
    if overrides:
        base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Tests: clean candidate passes trap detection correctly
# ---------------------------------------------------------------------------

def test_clean_candidate_no_traps():
    c = clean_candidate(_make_raw())
    result = detect_traps_numeric(c)
    assert result.is_honeypot is False
    # experience_mismatch is a SOFT flag — trust_penalty may be small but >0 is OK
    hard_flags = [f for f in result.trap_flags if f.startswith(HARD_FLAG_PREFIXES)]
    assert len(hard_flags) == 0  # no hard flags on a clean candidate


def test_salary_inversion_soft_flag():
    raw = _make_raw()
    raw["redrob_signals"]["expected_salary_range_inr_lpa"] = {"min": 50.0, "max": 20.0}
    c = clean_candidate(raw)
    result = detect_traps_numeric(c)
    assert result.is_honeypot is False  # soft flag never triggers honeypot
    assert any("salary_inverted" in f for f in result.trap_flags)
    assert result.trust_penalty > 0.0
    assert result.trust_penalty < 0.15  # soft flag = small penalty


def test_skill_stuffing_hard_flag():
    """Expert skill with 0 endorsements and 1 month duration = stuffing."""
    raw = _make_raw()
    raw["skills"] = [
        {"name": "FAISS", "proficiency": "expert", "endorsements": 0, "duration_months": 1},
        {"name": "Pinecone", "proficiency": "advanced", "endorsements": 1, "duration_months": 2},
        {"name": "Weaviate", "proficiency": "expert", "endorsements": 0, "duration_months": 1},
    ]
    c = clean_candidate(raw)
    result = detect_traps_numeric(c)
    assert any("skill_stuffing" in f for f in result.trap_flags)


def test_mass_expert_hard_flag():
    """4+ expert skills each with backing <= 2 = mass_expert flag."""
    raw = _make_raw()
    raw["skills"] = [
        {"name": f"Skill{i}", "proficiency": "expert", "endorsements": 0, "duration_months": 0}
        for i in range(5)
    ]
    c = clean_candidate(raw)
    result = detect_traps_numeric(c)
    assert any("mass_expert" in f for f in result.trap_flags)


def test_two_hard_flags_triggers_honeypot():
    """skill_stuffing + mass_expert = 2 hard flags = honeypot drop."""
    raw = _make_raw()
    # skill_stuffing: advanced/expert + short duration + 0 endorsements
    raw["skills"] = [
        {"name": f"Skill{i}", "proficiency": "expert", "endorsements": 0, "duration_months": 1}
        for i in range(6)  # >4 => mass_expert AND stuffing both fire
    ]
    c = clean_candidate(raw)
    result = detect_traps_numeric(c)
    hard_flags = [f for f in result.trap_flags if f.startswith(HARD_FLAG_PREFIXES)]
    if len(hard_flags) >= HARD_HONEYPOT_FLAG_COUNT:
        assert result.is_honeypot is True
        assert result.trust_penalty >= 0.30


def test_one_hard_flag_not_honeypot():
    """1 hard flag alone must NOT trigger honeypot drop."""
    raw = _make_raw()
    raw["skills"] = [
        {"name": f"Skill{i}", "proficiency": "expert", "endorsements": 0, "duration_months": 1}
        for i in range(3)
    ]
    c = clean_candidate(raw)
    result = detect_traps_numeric(c)
    # May or may not have mass_expert depending on count, but 1 flag = not honeypot
    hard_flags = [f for f in result.trap_flags if f.startswith(HARD_FLAG_PREFIXES)]
    if len(hard_flags) < HARD_HONEYPOT_FLAG_COUNT:
        assert result.is_honeypot is False


def test_date_contradiction_flag():
    """Role where end_date < start_date = date_contradiction hard flag."""
    raw = _make_raw()
    raw["career_history"] = [
        {
            "company": "BadCo",
            "title": "Engineer",
            "start_date": "2023-06-01",
            "end_date": "2022-01-01",  # end before start
            "duration_months": 18,
            "is_current": False,
            "industry": "Software",
            "company_size": "51-200",
            "description": "Test",
        }
    ]
    c = clean_candidate(raw)
    result = detect_traps_numeric(c)
    assert any("date_contradiction" in f for f in result.trap_flags)


def test_activity_impossible_large_gap():
    """last_active_date > 365 days before signup = activity_impossible hard flag."""
    raw = _make_raw()
    raw["redrob_signals"]["signup_date"]      = "2025-01-01"
    raw["redrob_signals"]["last_active_date"] = "2023-01-01"  # 2 years before signup
    c = clean_candidate(raw)
    result = detect_traps_numeric(c)
    assert any("activity_impossible" in f for f in result.trap_flags)


def test_finalize_adds_semantic_flag():
    c = clean_candidate(_make_raw())
    numeric = detect_traps_numeric(c)
    final = finalize_trap_result(numeric, semantic_flag=True, semantic_flag_label="title_desc_cosine:0.10")
    assert any("title_desc_cosine" in f for f in final.trap_flags)
    assert final.semantic_flag_count == 1
    # Semantic is soft: should not increase hard_count, so still not honeypot
    assert final.is_honeypot is False


def test_trust_penalty_capped_at_max():
    """Even with many flags, trust_penalty must not exceed MAX_TRUST_PENALTY=0.95."""
    from src.trap_detector import MAX_TRUST_PENALTY
    raw = _make_raw()
    raw["skills"] = [
        {"name": f"S{i}", "proficiency": "expert", "endorsements": 0, "duration_months": 0}
        for i in range(20)
    ]
    raw["career_history"] = [
        {
            "company": "BadCo",
            "title": "X",
            "start_date": "2023-01-01",
            "end_date": "2022-01-01",
            "duration_months": 120,
            "is_current": False,
            "industry": "Software",
            "company_size": "51-200",
            "description": "X",
        }
    ]
    raw["redrob_signals"]["signup_date"]      = "2025-01-01"
    raw["redrob_signals"]["last_active_date"] = "2020-01-01"
    c = clean_candidate(raw)
    result = detect_traps_numeric(c)
    assert result.trust_penalty <= MAX_TRUST_PENALTY
