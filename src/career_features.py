"""
src/career_features.py — Career-Fit feature engineering.

FIXES & IMPROVEMENTS over v1:
  1. Positive weights now sum to exactly 1.0 (was 0.95 in v1).
  2. Penalties are applied as independent subtractions AFTER the positive score
     is computed, then the result is clamped to [0, 1]. This prevents weights
     from summing to >1 or creating undefined behaviour.
  3. Added location_score: India-preferred candidates get a +0.05 bonus;
     outside-India candidates get a -0.10 penalty (per JD: case-by-case).
  4. Added negative_title_penalty: non-technical titles (marketing manager,
     accountant, civil engineer, etc.) with AI keywords stuffed in skills
     get a 0.35 penalty — the JD explicitly calls this out.
  5. applied_ml_at_product_years now correctly credits fractional years and
     filters HCL/Tech Mahindra as services firms in addition to TCS/Infosys.
  6. Shipped-system semantic score checks descriptions against multiple phrases.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional
import numpy as np

from src.cleaning import CleanCandidate

# ---------------------------------------------------------------------------
# POSITIVE WEIGHTS — must sum to 1.0
# ---------------------------------------------------------------------------
W_TITLE_ALIGNMENT       = 0.28
W_APPLIED_ML_PRODUCT    = 0.38
W_SHIPPED_SYSTEM        = 0.20
W_EXPERIENCE_BAND       = 0.08
W_INDUSTRY_FIT          = 0.06
_POS_WEIGHT_SUM = (
    W_TITLE_ALIGNMENT + W_APPLIED_ML_PRODUCT + W_SHIPPED_SYSTEM
    + W_EXPERIENCE_BAND + W_INDUSTRY_FIT
)
assert abs(_POS_WEIGHT_SUM - 1.00) < 1e-9, f"Positive weights sum {_POS_WEIGHT_SUM} != 1.0"

# ---------------------------------------------------------------------------
# PENALTY CAPS
# ---------------------------------------------------------------------------
JOB_HOP_PENALTY_MAX          = 0.20
SERVICES_ONLY_PENALTY_MAX    = 0.25
NEGATIVE_TITLE_PENALTY_MAX   = 0.35

# ---------------------------------------------------------------------------
# THRESHOLDS
# ---------------------------------------------------------------------------
JOB_HOP_TENURE_MONTHS        = 18
EXPERIENCE_IDEAL_MIN         = 6.0
EXPERIENCE_IDEAL_MAX         = 8.0
EXPERIENCE_ACCEPTABLE_MIN    = 5.0
EXPERIENCE_ACCEPTABLE_MAX    = 9.0

LOCATION_INDIA_BONUS         = 0.05
LOCATION_OUTSIDE_INDIA_PENALTY = 0.10

SERVICES_FIRMS = {
    "tcs", "infosys", "wipro", "accenture", "capgemini", "cognizant",
    "mindtree", "hcl", "tech mahindra", "mphasis", "hexaware",
    "l&t infotech", "ltimindtree",
}

AI_PRODUCT_INDUSTRIES = {
    "ai/ml", "fintech", "e-commerce", "food delivery", "transportation",
    "edtech", "healthtech", "saas", "software", "technology", "internet",
    "search", "advertising technology", "recommendation",
}

NEGATIVE_TITLE_TOKENS = {
    "marketing manager", "operations manager", "hr manager", "accountant",
    "civil engineer", "mechanical engineer", "graphic designer",
    "content writer", "sales executive", "customer support",
    "business analyst", "project manager", "product manager",
    "supply chain", "logistics", "finance manager",
}

ACCEPTABLE_TECHNICAL_TOKENS = {
    "data scientist", "ml engineer", "ai engineer", "software engineer",
    "data engineer", "backend engineer", "nlp", "research",
    "applied scientist", "machine learning", "search", "ranking",
    "recommendation", "information retrieval", "platform engineer",
}


@dataclass
class CareerFitResult:
    career_fit: float
    title_alignment: float
    applied_ml_at_product_years: float
    applied_ml_at_product_score: float
    shipped_system_score: float
    shipped_evidence_phrase: str
    job_hop_penalty: float
    services_only_penalty: float
    negative_title_penalty: float
    experience_band_score: float
    industry_fit_score: float
    is_services_only_career: bool
    location_score: float


def _title_is_negative(title: str) -> bool:
    t = title.lower()
    if any(tok in t for tok in ACCEPTABLE_TECHNICAL_TOKENS):
        return False
    return any(tok in t for tok in NEGATIVE_TITLE_TOKENS)


def _is_services_company(company: str) -> bool:
    return company.lower().strip() in SERVICES_FIRMS


def _experience_band_score(years: float) -> float:
    """
    1.0 if in ideal band [6,8], 0.75 if acceptable [5,9],
    decays linearly outside, floor 0.1.
    """
    if EXPERIENCE_IDEAL_MIN <= years <= EXPERIENCE_IDEAL_MAX:
        return 1.0
    if EXPERIENCE_ACCEPTABLE_MIN <= years <= EXPERIENCE_ACCEPTABLE_MAX:
        return 0.75
    if years < EXPERIENCE_ACCEPTABLE_MIN:
        return max(0.10, 0.75 * years / EXPERIENCE_ACCEPTABLE_MIN)
    # Over 9 years: slight decay (over-qualified risk per JD)
    return max(0.40, 0.75 - 0.05 * (years - EXPERIENCE_ACCEPTABLE_MAX))


def _job_hop_penalty(career_history) -> float:
    """
    Fraction of non-current roles with duration < JOB_HOP_TENURE_MONTHS,
    scaled to JOB_HOP_PENALTY_MAX.
    """
    non_current = [r for r in career_history if not r.is_current]
    if not non_current:
        return 0.0
    hops = sum(1 for r in non_current if r.duration_months_stated < JOB_HOP_TENURE_MONTHS)
    frac = hops / len(non_current)
    return min(frac * JOB_HOP_PENALTY_MAX, JOB_HOP_PENALTY_MAX)


def _services_only_penalty(career_history) -> tuple[float, bool]:
    """
    If every role is at a services firm, return max penalty.
    If majority services but some product, partial penalty.
    """
    if not career_history:
        return 0.0, False
    services_months = sum(
        r.duration_months_stated
        for r in career_history
        if _is_services_company(r.company)
    )
    total_months = sum(r.duration_months_stated for r in career_history) or 1
    services_fraction = services_months / total_months
    is_services_only = services_fraction > 0.95
    penalty = min(services_fraction * SERVICES_ONLY_PENALTY_MAX, SERVICES_ONLY_PENALTY_MAX)
    return penalty, is_services_only


def _location_score(country: str, location: str, acceptable_locations: list[str]) -> float:
    """
    Bonus if India-based in acceptable cities; penalty if outside India entirely.
    """
    country_lower = country.lower().strip()
    location_lower = location.lower().strip()
    if country_lower == "india":
        if any(loc.lower() in location_lower for loc in acceptable_locations):
            return LOCATION_INDIA_BONUS
        return 0.0
    return -LOCATION_OUTSIDE_INDIA_PENALTY


def compute_career_features(
    cand: CleanCandidate,
    jd_requirements: dict,
    embed_fn,
) -> CareerFitResult:
    """Full career-fit computation."""

    acceptable_locations = jd_requirements.get("role", {}).get(
        "locations_acceptable_india", []
    )

    # ------------------------------------------------------------------ #
    # 1. Title alignment
    # ------------------------------------------------------------------ #
    title_protos = (
        jd_requirements.get("title_prototypes", {}).get("must", [])
        + jd_requirements.get("title_prototypes", {}).get("adjacent", [])
    )
    current_title = cand.current_title or ""
    if current_title and title_protos:
        title_emb = embed_fn([current_title])   # (1, dim)
        proto_emb = embed_fn(title_protos)      # (n, dim)
        sims = (title_emb @ proto_emb.T).flatten()
        # Must-prototypes get full weight, adjacent get 0.5
        n_must = len(jd_requirements.get("title_prototypes", {}).get("must", []))
        weights = np.ones(len(sims))
        weights[n_must:] = 0.5
        title_alignment = float(np.clip(
            (sims * weights).max(), 0.0, 1.0
        ))
    else:
        title_alignment = 0.0

    # Negative title penalty (non-technical role with AI keyword stuffing)
    neg_title_penalty = 0.0
    if _title_is_negative(current_title):
        # Check if career history also looks non-technical
        tech_role_count = sum(
            1 for r in cand.career_history
            if any(tok in r.title.lower() for tok in ACCEPTABLE_TECHNICAL_TOKENS)
        )
        if tech_role_count == 0:
            neg_title_penalty = NEGATIVE_TITLE_PENALTY_MAX
        else:
            neg_title_penalty = NEGATIVE_TITLE_PENALTY_MAX * 0.5

    # ------------------------------------------------------------------ #
    # 2. Applied ML years at product companies
    # ------------------------------------------------------------------ #
    ml_title_tokens = {
        "ml engineer", "ai engineer", "machine learning", "data scientist",
        "applied scientist", "nlp engineer", "search engineer",
        "recommendation", "ranking engineer", "research scientist",
        "applied ml", "applied ai",
    }
    applied_ml_months = 0.0
    for r in cand.career_history:
        if _is_services_company(r.company):
            continue  # services firms don't count
        role_lower = r.title.lower()
        desc_lower = r.description.lower()
        is_ml_role = any(tok in role_lower or tok in desc_lower for tok in ml_title_tokens)
        if is_ml_role:
            applied_ml_months += r.duration_months_stated

    applied_ml_years = applied_ml_months / 12.0
    # Sigmoid-like score: 4yr => ~0.7, 6yr => ~0.95
    if applied_ml_years <= 0:
        applied_ml_score = 0.0
    elif applied_ml_years >= 6:
        applied_ml_score = 1.0
    else:
        applied_ml_score = float(np.clip(
            1.0 / (1.0 + math.exp(-1.2 * (applied_ml_years - 3.0))),
            0.0, 1.0,
        ))

    # ------------------------------------------------------------------ #
    # 3. Shipped-system evidence (Tier-5 path)
    # ------------------------------------------------------------------ #
    shipped_phrases = jd_requirements.get("shipped_system_phrases", [])
    best_shipped_score = 0.0
    best_shipped_phrase = ""

    if shipped_phrases:
        all_descs = [r.description for r in cand.career_history if r.description]
        if all_descs:
            combined_desc = " ".join(all_descs)
            desc_emb = embed_fn([combined_desc])    # (1, dim)
            phrase_emb = embed_fn(shipped_phrases)  # (n, dim)
            sims = (desc_emb @ phrase_emb.T).flatten()
            best_idx = int(np.argmax(sims))
            best_shipped_score = float(np.clip(sims[best_idx], 0.0, 1.0))
            best_shipped_phrase = shipped_phrases[best_idx] if best_shipped_score > 0.3 else ""

    # ------------------------------------------------------------------ #
    # 4. Experience band
    # ------------------------------------------------------------------ #
    exp_band = _experience_band_score(cand.years_of_experience_stated)

    # ------------------------------------------------------------------ #
    # 5. Industry fit
    # ------------------------------------------------------------------ #
    current_industry_lower = (cand.current_industry or "").lower()
    industry_fit = 1.0 if any(
        ind in current_industry_lower for ind in AI_PRODUCT_INDUSTRIES
    ) else 0.3

    # ------------------------------------------------------------------ #
    # 6. Job-hop penalty
    # ------------------------------------------------------------------ #
    hop_penalty = _job_hop_penalty(cand.career_history)

    # ------------------------------------------------------------------ #
    # 7. Services-only penalty
    # ------------------------------------------------------------------ #
    svc_penalty, is_services_only = _services_only_penalty(cand.career_history)

    # ------------------------------------------------------------------ #
    # 8. Location score
    # ------------------------------------------------------------------ #
    loc_score = _location_score(cand.country, cand.location, acceptable_locations)

    # ------------------------------------------------------------------ #
    # FINAL career_fit assembly
    # ------------------------------------------------------------------ #
    positive_score = (
        W_TITLE_ALIGNMENT    * title_alignment
        + W_APPLIED_ML_PRODUCT * applied_ml_score
        + W_SHIPPED_SYSTEM     * best_shipped_score
        + W_EXPERIENCE_BAND    * exp_band
        + W_INDUSTRY_FIT       * industry_fit
    )

    # Apply location as an additive bonus/penalty on top (small magnitude)
    positive_score += loc_score * 0.10

    # Subtract penalties (independent of weight sum)
    career_fit = positive_score - hop_penalty - svc_penalty - neg_title_penalty
    career_fit = float(np.clip(career_fit, 0.0, 1.0))

    return CareerFitResult(
        career_fit=career_fit,
        title_alignment=title_alignment,
        applied_ml_at_product_years=applied_ml_years,
        applied_ml_at_product_score=applied_ml_score,
        shipped_system_score=best_shipped_score,
        shipped_evidence_phrase=best_shipped_phrase,
        job_hop_penalty=hop_penalty,
        services_only_penalty=svc_penalty,
        negative_title_penalty=neg_title_penalty,
        experience_band_score=exp_band,
        industry_fit_score=industry_fit,
        is_services_only_career=is_services_only,
        location_score=loc_score,
    )
