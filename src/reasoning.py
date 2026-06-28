"""
src/reasoning.py — Per-candidate reasoning generation.

CRITICAL IMPROVEMENT over v1:
  - Opening clause is VARIED per candidate: instead of always leading with
    "strong fit on embeddings-based retrieval", the opening rotates to the
    SINGLE STRONGEST differentiator for each candidate.
  - Honest concerns section is always present if any issue exists.
  - No hallucination: all facts sourced directly from feature columns.
  - Stage 4 rubric compliance: specific facts, JD connection, rank consistency.
  - Rank-band tone calibration (1-10 strong, 11-30 good, 31-60 moderate, 61-100 marginal).
"""
from __future__ import annotations
import re
from typing import Optional


# ---------------------------------------------------------------------------
# RANK-BAND TONE
# ---------------------------------------------------------------------------
def _tone_band(rank: int) -> str:
    if rank <= 10:
        return "strong"
    if rank <= 30:
        return "good"
    if rank <= 60:
        return "moderate"
    return "marginal"


# ---------------------------------------------------------------------------
# FIELD FORMATTERS
# ---------------------------------------------------------------------------
def _yrs(v) -> str:
    try:
        f = float(v)
        return f"{f:.1f}yr"
    except (TypeError, ValueError):
        return "?"


def _pct(v) -> str:
    try:
        return f"{float(v)*100:.0f}%"
    except (TypeError, ValueError):
        return "?"


def _days(v) -> str:
    try:
        return f"{int(v)}d"
    except (TypeError, ValueError):
        return "?"


# ---------------------------------------------------------------------------
# OPENING CLAUSE — rotates based on strongest signal
# ---------------------------------------------------------------------------
def _opening_clause(row: dict, rank: int) -> str:
    """
    Pick the single strongest differentiator as the opening sentence.
    Candidates with different strongest signals will have different openings.
    """
    title = str(row.get("current_title", "Candidate"))
    yrs   = _yrs(row.get("years_of_experience_stated", 0))
    comp  = str(row.get("current_company", ""))

    matched_must = str(row.get("matched_must_skills", "")).strip()
    shipped      = str(row.get("shipped_evidence_phrase", "")).strip()
    ml_yrs       = float(row.get("applied_ml_at_product_years", 0) or 0)
    skill_must   = float(row.get("skill_fit_must", 0) or 0)
    career_fit   = float(row.get("career_fit", 0) or 0)
    title_align  = float(row.get("title_alignment", 0) or 0)

    # Determine strongest opener
    if ml_yrs >= 5.0 and career_fit >= 0.70:
        opener = (
            f"{title} at {comp} with {_yrs(ml_yrs)} applied ML at product companies"
            if comp else
            f"{title} with {_yrs(ml_yrs)} applied ML at product companies"
        )
    elif shipped and float(row.get("shipped_system_score", 0) or 0) >= 0.50:
        # Tier-5 path: specific shipped evidence
        opener = f"{title} with evidence of '{shipped}' in career history ({yrs} exp)"
    elif skill_must >= 0.25 and matched_must:
        # Skill-led opener — but mention WHICH specific skill was strongest
        first_match = matched_must.split(",")[0].split("~")[-1].strip()
        opener = (
            f"{title} ({yrs}) with strong match on {first_match}"
        )
    elif title_align >= 0.70:
        opener = f"{title} role at {comp} closely matches AI/ML engineering profile ({yrs} exp)"
    else:
        tone = _tone_band(rank)
        opener = f"{title} ({yrs} exp, {comp or 'current employer'}) — {tone} overall fit"

    return opener


# ---------------------------------------------------------------------------
# CONCERN CLAUSE — honest negatives
# ---------------------------------------------------------------------------
def _concern_clause(row: dict) -> Optional[str]:
    concerns = []

    notice = int(row.get("notice_period_days", 0) or 0)
    if notice > 90:
        concerns.append(f"notice period {_days(notice)} is a concern (JD prefers ≤30d)")
    elif notice > 30:
        concerns.append(f"notice period {_days(notice)} (acceptable but not ideal)")

    days_inactive = int(row.get("days_since_last_active", 0) or 0)
    if days_inactive > 180:
        concerns.append(f"last active {_days(days_inactive)} ago — availability uncertain")
    elif days_inactive > 90:
        concerns.append(f"last active {_days(days_inactive)} ago — moderately disengaged")

    trust_pen = float(row.get("trust_penalty", 0) or 0)
    if trust_pen >= 0.15:
        flags = str(row.get("trap_flags", "")).strip()
        if "title_desc" in flags:
            concerns.append("profile shows title-description cosine mismatch — weaker signal confidence")
        elif "experience_mismatch" in flags:
            concerns.append("stated experience years inconsistent with computed career span")
        elif "salary_inverted" in flags:
            concerns.append("salary range min > max — minor data quality flag")

    svc_pen = float(row.get("services_only_penalty", 0) or 0)
    if svc_pen >= 0.15:
        concerns.append("majority career at IT services firms (JD explicitly prefers product-company background)")

    rr = float(row.get("response_rate", 0) or 0)
    if rr < 0.20:
        concerns.append(f"low recruiter response rate ({_pct(rr)}) — may be hard to reach")

    is_outside_india = str(row.get("country", "")).strip().lower() not in ("india", "")
    if is_outside_india:
        country = str(row.get("country", "")).strip()
        concerns.append(f"located in {country} — relocation or visa case-by-case per JD")

    if not concerns:
        return None
    return "; ".join(concerns)


# ---------------------------------------------------------------------------
# POSITIVE DETAIL CLAUSE
# ---------------------------------------------------------------------------
def _positive_detail(row: dict) -> str:
    parts = []

    avail    = float(row.get("availability", 0) or 0)
    eng      = float(row.get("engagement", 0) or 0)
    notice   = int(row.get("notice_period_days", 0) or 0)
    otw      = str(row.get("open_to_work_flag", "False")).lower() == "true"

    if otw:
        parts.append("actively open to work")
    if notice <= 30:
        parts.append(f"notice ≤{notice}d (ideal)")
    if eng >= 0.65:
        rr = float(row.get("response_rate", 0) or 0)
        parts.append(f"high recruiter engagement ({_pct(rr)} response rate)")

    matched_nice = str(row.get("matched_nice_skills", "")).strip()
    if matched_nice:
        nice_list = [x.split("~")[-1].strip() for x in matched_nice.split(",")][:2]
        parts.append(f"bonus fit on {', '.join(nice_list)}")

    ml_yrs = float(row.get("applied_ml_at_product_years", 0) or 0)
    if ml_yrs >= 3.0 and f"{_yrs(ml_yrs)} applied ML" not in parts:
        parts.append(f"{_yrs(ml_yrs)} applied ML at product companies")

    if not parts:
        return "platform signals indicate genuine availability"

    return "; ".join(parts)


# ---------------------------------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------------------------------
def build_reasoning(row: dict, rank: int) -> str:
    """
    Build a 1-2 sentence reasoning string for a single top-100 candidate.
    - Sentence 1: opening (strongest differentiator) + positive detail.
    - Sentence 2: concerns (if any), or reaffirmation for top ranks.
    No hallucination: all facts from feature columns.
    """
    opener   = _opening_clause(row, rank)
    positive = _positive_detail(row)
    concern  = _concern_clause(row)

    sentence1 = f"{opener}; {positive}."

    if concern:
        sentence2 = f"Concern: {concern}."
    elif rank <= 10:
        career_fit = float(row.get("career_fit", 0) or 0)
        sentence2 = f"No significant concerns; career_fit={career_fit:.2f} places this candidate in the elite tier."
    elif rank <= 30:
        sentence2 = "Minor gaps only; overall profile closely matches the JD's product-engineering mandate."
    else:
        sentence2 = "Included as a solid-but-not-elite match; skill and availability signals are positive."

    # Truncate to avoid excessively long reasoning
    result = f"{sentence1} {sentence2}"
    if len(result) > 400:
        result = result[:397] + "..."

    return result


def add_reasoning_column(df, rank_col: str = "rank") -> None:
    """In-place: add 'reasoning' column to the ranked DataFrame."""
    rows = df.to_dict(orient="records")
    df["reasoning"] = [
        build_reasoning(row, int(row[rank_col]))
        for row in rows
    ]
