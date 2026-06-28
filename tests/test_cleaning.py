"""tests/test_cleaning.py — Unit tests for src/cleaning.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.cleaning import clean_candidate, PROFICIENCY_ORDINAL


MINIMAL_CANDIDATE = {
    "candidate_id": "CAND_0000001",
    "profile": {
        "anonymized_name": "Test User",
        "headline": "ML Engineer",
        "summary": "Test summary",
        "location": "Pune, Maharashtra",
        "country": "India",
        "years_of_experience": 6.0,
        "current_title": "ML Engineer",
        "current_company": "TestCo",
        "current_company_size": "51-200",
        "current_industry": "Software",
    },
    "career_history": [
        {
            "company": "TestCo",
            "title": "ML Engineer",
            "start_date": "2022-01-01",
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
        {"name": "Python", "proficiency": "advanced", "endorsements": 10, "duration_months": 30},
        {"name": "FAISS", "proficiency": "expert", "endorsements": 5, "duration_months": 24},
    ],
    "redrob_signals": {
        "profile_completeness_score": 80.0,
        "signup_date": "2022-01-01",
        "last_active_date": "2026-05-01",
        "open_to_work_flag": True,
        "profile_views_received_30d": 10,
        "applications_submitted_30d": 2,
        "recruiter_response_rate": 0.7,
        "avg_response_time_hours": 24.0,
        "skill_assessment_scores": {"Python": 85.0},
        "connection_count": 200,
        "endorsements_received": 20,
        "notice_period_days": 30,
        "expected_salary_range_inr_lpa": {"min": 20.0, "max": 40.0},
        "preferred_work_mode": "hybrid",
        "willing_to_relocate": True,
        "github_activity_score": 60.0,
        "search_appearance_30d": 50,
        "saved_by_recruiters_30d": 5,
        "interview_completion_rate": 0.8,
        "offer_acceptance_rate": 0.5,
        "verified_email": True,
        "verified_phone": True,
        "linkedin_connected": True,
    },
}


def test_clean_candidate_basic():
    c = clean_candidate(MINIMAL_CANDIDATE)
    assert c.candidate_id == "CAND_0000001"
    assert c.current_title == "ML Engineer"
    assert c.years_of_experience_stated == 6.0
    assert c.country == "India"


def test_clean_skills_proficiency_ordinal():
    c = clean_candidate(MINIMAL_CANDIDATE)
    assert c.skills[0].name == "Python"
    assert c.skills[0].proficiency_ordinal == PROFICIENCY_ORDINAL["advanced"]
    assert c.skills[1].proficiency_ordinal == PROFICIENCY_ORDINAL["expert"]


def test_clean_signals_sentinel_github_minus1():
    """github_activity_score == -1 should map to None."""
    raw = {**MINIMAL_CANDIDATE}
    raw["redrob_signals"] = {**MINIMAL_CANDIDATE["redrob_signals"], "github_activity_score": -1}
    c = clean_candidate(raw)
    assert c.signals.github_activity_score is None


def test_clean_signals_sentinel_offer_minus1():
    """offer_acceptance_rate == -1 should map to None."""
    raw = {**MINIMAL_CANDIDATE}
    raw["redrob_signals"] = {**MINIMAL_CANDIDATE["redrob_signals"], "offer_acceptance_rate": -1}
    c = clean_candidate(raw)
    assert c.signals.offer_acceptance_rate is None


def test_salary_inversion_flag():
    raw = {**MINIMAL_CANDIDATE}
    raw["redrob_signals"] = {
        **MINIMAL_CANDIDATE["redrob_signals"],
        "expected_salary_range_inr_lpa": {"min": 40.0, "max": 20.0},
    }
    c = clean_candidate(raw)
    assert c.salary_min_gt_max is True


def test_no_salary_inversion_normal():
    c = clean_candidate(MINIMAL_CANDIDATE)
    assert c.salary_min_gt_max is False


def test_missing_candidate_id_raises():
    raw = {**MINIMAL_CANDIDATE}
    raw["candidate_id"] = ""
    with pytest.raises(ValueError):
        clean_candidate(raw)


def test_primary_text_not_empty():
    c = clean_candidate(MINIMAL_CANDIDATE)
    text = c.primary_text()
    assert len(text) > 0
    assert "ML Engineer" in text or "Python" in text


def test_career_months_computed():
    c = clean_candidate(MINIMAL_CANDIDATE)
    # is_current = True => duration_months_computed is None for the current role
    assert c.career_history[0].duration_months_computed is None
    assert c.total_career_months_computed == 30
