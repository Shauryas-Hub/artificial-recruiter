"""
src/cleaning.py — Data cleaning & normalization.

Single responsibility: take a raw candidate dict (one JSONL record) and return
a CleanCandidate dataclass with:
  - Sentinel values resolved: github_activity_score==-1 and offer_acceptance_rate
    ==-1 are mapped to None (MISSING, never averaged in as a real number).
  - Enum -> ordinal normalization (proficiency, company_size, education tier).
  - Dates parsed to datetime.date.
  - Derived fields: days_since_last_active (filled later), total_career_months,
    per-role tenure recomputed from dates (for trap detection).
  - Quality flags: salary_min_gt_max, experience_mismatch.

Determinism: pure function of input + constants. No randomness. No I/O.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

PROFICIENCY_ORDINAL: dict[str, int] = {
    "beginner": 1,
    "intermediate": 2,
    "advanced": 3,
    "expert": 4,
}

EDUCATION_TIER_ORDINAL: dict[str, int] = {
    "tier_1": 4,
    "tier_2": 3,
    "tier_3": 2,
    "tier_4": 1,
    "unknown": 0,
}

COMPANY_SIZE_MIDPOINT: dict[str, int] = {
    "1-10": 5,
    "11-50": 30,
    "51-200": 125,
    "201-500": 350,
    "501-1000": 750,
    "1001-5000": 3000,
    "5001-10000": 7500,
    "10001+": 20000,
}

SENTINEL_MISSING_VALUES: set[float] = {-1.0, -1}
EXPERIENCE_MISMATCH_TOLERANCE_YEARS: float = 2.0


# ---------------------------------------------------------------------------
# DATA STRUCTURES
# ---------------------------------------------------------------------------

@dataclass
class CleanSkill:
    name: str
    proficiency_ordinal: int
    proficiency_raw: str
    endorsements: int
    duration_months: int


@dataclass
class CleanCareerRole:
    company: str
    title: str
    start_date: Optional[date]
    end_date: Optional[date]
    duration_months_stated: int
    duration_months_computed: Optional[int]
    is_current: bool
    industry: str
    company_size: str
    company_size_midpoint: int
    description: str
    tenure_mismatch_months: Optional[int]


@dataclass
class CleanEducation:
    institution: str
    degree: str
    field_of_study: str
    start_year: int
    end_year: int
    grade: Optional[str]
    tier_ordinal: int


@dataclass
class CleanSignals:
    profile_completeness_score: float
    signup_date: Optional[date]
    last_active_date: Optional[date]
    open_to_work_flag: bool
    profile_views_received_30d: int
    applications_submitted_30d: int
    recruiter_response_rate: Optional[float]
    avg_response_time_hours: Optional[float]
    skill_assessment_scores: dict[str, float]
    connection_count: int
    endorsements_received: int
    notice_period_days: int
    expected_salary_min: Optional[float]
    expected_salary_max: Optional[float]
    preferred_work_mode: str
    willing_to_relocate: bool
    github_activity_score: Optional[float]
    search_appearance_30d: int
    saved_by_recruiters_30d: int
    interview_completion_rate: float
    offer_acceptance_rate: Optional[float]
    verified_email: bool
    verified_phone: bool
    linkedin_connected: bool


@dataclass
class CleanCandidate:
    candidate_id: str
    anonymized_name: str
    headline: str
    summary: str
    location: str
    country: str
    years_of_experience_stated: float
    current_title: str
    current_company: str
    current_company_size: str
    current_industry: str
    skills: list[CleanSkill]
    career_history: list[CleanCareerRole]
    education: list[CleanEducation]
    certifications: list[dict]
    languages: list[dict]
    signals: CleanSignals
    total_career_months_computed: int
    total_career_years_computed: float
    experience_mismatch_years: Optional[float]
    salary_min_gt_max: bool
    days_since_last_active: Optional[int] = None

    def primary_text(self) -> str:
        """Concatenated free-text fields for embedding."""
        parts = [self.headline, self.summary]
        parts.extend(r.description for r in self.career_history)
        parts.extend(s.name for s in self.skills)
        return " ".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _parse_date(s) -> Optional[date]:
    if s is None or s == "":
        return None
    try:
        return datetime.strptime(str(s), "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _is_missing_sentinel(v) -> bool:
    try:
        return float(v) in SENTINEL_MISSING_VALUES
    except (TypeError, ValueError):
        return False


def _safe_float(v) -> Optional[float]:
    if v is None or _is_missing_sentinel(v):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _months_between(start: Optional[date], end: Optional[date]) -> Optional[int]:
    if start is None or end is None:
        return None
    if end < start:
        return None
    months = (end.year - start.year) * 12 + (end.month - start.month)
    if end.day > start.day:
        months += 1
    return max(months, 0)


# ---------------------------------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------------------------------

def clean_candidate(raw: dict) -> CleanCandidate:
    """
    Normalize one raw candidate dict into a CleanCandidate.
    Pure & deterministic.
    """
    cid = raw.get("candidate_id", "")
    if not cid:
        raise ValueError("Candidate record missing candidate_id")

    profile = raw.get("profile", {}) or {}
    raw_skills = raw.get("skills", []) or []
    raw_career = raw.get("career_history", []) or []
    raw_edu = raw.get("education", []) or []
    raw_signals = raw.get("redrob_signals", {}) or {}

    # ---- skills ----
    skills: list[CleanSkill] = []
    for s in raw_skills:
        prof_raw = str(s.get("proficiency", "beginner")).lower().strip()
        skills.append(
            CleanSkill(
                name=str(s.get("name", "")).strip(),
                proficiency_ordinal=PROFICIENCY_ORDINAL.get(prof_raw, 0),
                proficiency_raw=prof_raw,
                endorsements=int(s.get("endorsements", 0) or 0),
                duration_months=int(s.get("duration_months", 0) or 0),
            )
        )

    # ---- career history ----
    career: list[CleanCareerRole] = []
    for r in raw_career:
        sd = _parse_date(r.get("start_date"))
        ed = _parse_date(r.get("end_date"))
        stated = int(r.get("duration_months", 0) or 0)
        if r.get("is_current"):
            computed = None
        else:
            computed = _months_between(sd, ed)
        mismatch = None
        if computed is not None:
            mismatch = abs(stated - computed)
        size_raw = str(r.get("company_size", "1-10"))
        career.append(
            CleanCareerRole(
                company=str(r.get("company", "")).strip(),
                title=str(r.get("title", "")).strip(),
                start_date=sd,
                end_date=ed,
                duration_months_stated=stated,
                duration_months_computed=computed,
                is_current=bool(r.get("is_current", False)),
                industry=str(r.get("industry", "")).strip(),
                company_size=size_raw,
                company_size_midpoint=COMPANY_SIZE_MIDPOINT.get(size_raw, 0),
                description=str(r.get("description", "")).strip(),
                tenure_mismatch_months=mismatch,
            )
        )

    # ---- education ----
    edu: list[CleanEducation] = []
    for e in raw_edu:
        tier_raw = str(e.get("tier", "unknown")).lower().strip()
        edu.append(
            CleanEducation(
                institution=str(e.get("institution", "")).strip(),
                degree=str(e.get("degree", "")).strip(),
                field_of_study=str(e.get("field_of_study", "")).strip(),
                start_year=int(e.get("start_year", 0) or 0),
                end_year=int(e.get("end_year", 0) or 0),
                grade=(str(e.get("grade")).strip() if e.get("grade") is not None else None),
                tier_ordinal=EDUCATION_TIER_ORDINAL.get(tier_raw, 0),
            )
        )

    # ---- signals ----
    sal = raw_signals.get("expected_salary_range_inr_lpa", {}) or {}
    sal_min = _safe_float(sal.get("min"))
    sal_max = _safe_float(sal.get("max"))
    salary_inverted = (
        sal_min is not None and sal_max is not None and sal_min > sal_max
    )

    signals = CleanSignals(
        profile_completeness_score=float(raw_signals.get("profile_completeness_score", 0) or 0),
        signup_date=_parse_date(raw_signals.get("signup_date")),
        last_active_date=_parse_date(raw_signals.get("last_active_date")),
        open_to_work_flag=bool(raw_signals.get("open_to_work_flag", False)),
        profile_views_received_30d=int(raw_signals.get("profile_views_received_30d", 0) or 0),
        applications_submitted_30d=int(raw_signals.get("applications_submitted_30d", 0) or 0),
        recruiter_response_rate=_safe_float(raw_signals.get("recruiter_response_rate")),
        avg_response_time_hours=_safe_float(raw_signals.get("avg_response_time_hours")),
        skill_assessment_scores={
            str(k): float(v)
            for k, v in (raw_signals.get("skill_assessment_scores", {}) or {}).items()
            if _safe_float(v) is not None
        },
        connection_count=int(raw_signals.get("connection_count", 0) or 0),
        endorsements_received=int(raw_signals.get("endorsements_received", 0) or 0),
        notice_period_days=int(raw_signals.get("notice_period_days", 0) or 0),
        expected_salary_min=sal_min,
        expected_salary_max=sal_max,
        preferred_work_mode=str(raw_signals.get("preferred_work_mode", "flexible")).lower().strip(),
        willing_to_relocate=bool(raw_signals.get("willing_to_relocate", False)),
        github_activity_score=_safe_float(raw_signals.get("github_activity_score")),
        search_appearance_30d=int(raw_signals.get("search_appearance_30d", 0) or 0),
        saved_by_recruiters_30d=int(raw_signals.get("saved_by_recruiters_30d", 0) or 0),
        interview_completion_rate=float(raw_signals.get("interview_completion_rate", 0) or 0),
        offer_acceptance_rate=_safe_float(raw_signals.get("offer_acceptance_rate")),
        verified_email=bool(raw_signals.get("verified_email", False)),
        verified_phone=bool(raw_signals.get("verified_phone", False)),
        linkedin_connected=bool(raw_signals.get("linkedin_connected", False)),
    )

    # ---- derived ----
    total_months = sum(r.duration_months_stated for r in career)
    total_years = total_months / 12.0
    stated_years = float(profile.get("years_of_experience", 0) or 0)
    exp_mismatch = abs(stated_years - total_years) if total_years > 0 else None

    return CleanCandidate(
        candidate_id=cid,
        anonymized_name=str(profile.get("anonymized_name", "")).strip(),
        headline=str(profile.get("headline", "")).strip(),
        summary=str(profile.get("summary", "")).strip(),
        location=str(profile.get("location", "")).strip(),
        country=str(profile.get("country", "")).strip(),
        years_of_experience_stated=stated_years,
        current_title=str(profile.get("current_title", "")).strip(),
        current_company=str(profile.get("current_company", "")).strip(),
        current_company_size=str(profile.get("current_company_size", "1-10")),
        current_industry=str(profile.get("current_industry", "")).strip(),
        skills=skills,
        career_history=career,
        education=edu,
        certifications=list(raw.get("certifications", []) or []),
        languages=list(raw.get("languages", []) or []),
        signals=signals,
        total_career_months_computed=total_months,
        total_career_years_computed=total_years,
        experience_mismatch_years=exp_mismatch,
        salary_min_gt_max=salary_inverted,
    )
