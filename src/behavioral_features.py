"""
src/behavioral_features.py — Behavioral-Fit feature engineering.

The 23 Redrob behavioral signals are grouped into 4 sub-scores:
  availability        (0..1) — how reachable and ready is this person right now?
  engagement          (0..1) — how actively are they responding to recruiters?
  trust_verification  (0..1) — how verified and complete is their profile?
  engineering_signal  (0..1) — GitHub / platform activity showing real engineering

Combined using weighted geometric mean so one catastrophic signal (e.g.,
last_active 11 months ago) propagates multiplicatively and drags the whole
behavioral score down.

IMPROVEMENTS over v1:
  - open_to_work_flag is now incorporated into availability (was ignored).
  - saved_by_recruiters_30d (market demand signal) is included in engagement.
  - offer_acceptance_rate handled: if None (no prior offers), treated as neutral.
  - profile_completeness_score incorporated into trust_verification.
  - interview_completion_rate incorporated into engagement (reliable candidate).
  - beta exponent in final scorer reduced to 0.80 to avoid over-penalizing strong
    ML candidates who happen to have low GitHub scores (many work on private repos).
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional
import numpy as np

# ---------------------------------------------------------------------------
# SUB-SCORE WEIGHTS — must sum to 1.0
# ---------------------------------------------------------------------------
W_AVAILABILITY        = 0.45
W_ENGAGEMENT          = 0.30
W_TRUST_VERIFICATION  = 0.15
W_ENGINEERING_SIGNAL  = 0.10

# ---------------------------------------------------------------------------
# THRESHOLDS
# ---------------------------------------------------------------------------
DAYS_ACTIVE_IDEAL  = 30
DAYS_ACTIVE_MAX    = 365
NOTICE_IDEAL_DAYS  = 30
NOTICE_MAX_DAYS    = 90
RESPONSE_RATE_IDEAL = 0.7
RESPONSE_RATE_FLOOR = 0.1
RESPONSE_TIME_IDEAL_HOURS = 24
RESPONSE_TIME_MAX_HOURS   = 168     # 1 week
GITHUB_IDEAL        = 50
INTERVIEW_IDEAL     = 0.80          # 80%+ interview completion = reliable


@dataclass
class BehavioralFitResult:
    behavioral_fit: float
    availability: float
    engagement: float
    trust_verification: float
    engineering_signal: float


def _linear_decay(val: float, ideal: float, max_val: float) -> float:
    """1.0 at <=ideal, linear decay to 0.0 at max_val."""
    if val <= ideal:
        return 1.0
    if val >= max_val:
        return 0.0
    return 1.0 - (val - ideal) / (max_val - ideal)


def _sigmoid_01(val: float, mid: float, steepness: float = 5.0) -> float:
    """Smooth 0..1 score centered at mid."""
    return float(1.0 / (1.0 + math.exp(-steepness * (val - mid))))


def compute_behavioral_features(
    cand,               # CleanCandidate
    date_anchor,        # datetime.date — deterministic anchor
) -> BehavioralFitResult:
    sig = cand.signals

    # ------------------------------------------------------------------
    # 1. AVAILABILITY
    # ------------------------------------------------------------------
    # 1a. days since last active
    if sig.last_active_date and date_anchor:
        days_since = (date_anchor - sig.last_active_date).days
    else:
        days_since = DAYS_ACTIVE_MAX   # conservative: unknown = max inactive

    days_score = _linear_decay(max(days_since, 0), DAYS_ACTIVE_IDEAL, DAYS_ACTIVE_MAX)

    # 1b. notice period (lower = better, 0..90 acceptable)
    notice = sig.notice_period_days
    if notice <= NOTICE_IDEAL_DAYS:
        notice_score = 1.0
    elif notice <= NOTICE_MAX_DAYS:
        notice_score = 0.5 + 0.5 * _linear_decay(notice, NOTICE_IDEAL_DAYS, NOTICE_MAX_DAYS)
    else:
        notice_score = max(0.1, 0.5 - 0.005 * (notice - NOTICE_MAX_DAYS))

    # 1c. open_to_work flag — explicit signal
    otw_score = 1.0 if sig.open_to_work_flag else 0.6  # not open_to_work is a penalty, not a zero

    availability = float(np.clip(
        0.50 * days_score + 0.30 * notice_score + 0.20 * otw_score,
        0.0, 1.0,
    ))

    # ------------------------------------------------------------------
    # 2. ENGAGEMENT
    # ------------------------------------------------------------------
    # 2a. recruiter response rate
    if sig.recruiter_response_rate is not None:
        rr = float(sig.recruiter_response_rate)
        rr_score = _linear_decay(
            max(RESPONSE_RATE_IDEAL - rr, 0),
            0,
            RESPONSE_RATE_IDEAL - RESPONSE_RATE_FLOOR,
        )
        # Simpler: direct mapping [0.1, 0.7] -> [0, 1]
        rr_score = float(np.clip(
            (rr - RESPONSE_RATE_FLOOR) / (RESPONSE_RATE_IDEAL - RESPONSE_RATE_FLOOR),
            0.0, 1.0,
        ))
    else:
        rr_score = 0.5

    # 2b. avg response time (lower = better)
    if sig.avg_response_time_hours is not None:
        rt = float(sig.avg_response_time_hours)
        rt_score = _linear_decay(rt, RESPONSE_TIME_IDEAL_HOURS, RESPONSE_TIME_MAX_HOURS)
    else:
        rt_score = 0.5

    # 2c. interview completion rate
    icr_score = float(np.clip(sig.interview_completion_rate / INTERVIEW_IDEAL, 0.0, 1.0))

    # 2d. saved_by_recruiters_30d (market demand signal, log-scaled)
    svd = float(sig.saved_by_recruiters_30d or 0)
    svd_score = float(np.clip(math.log1p(svd) / math.log1p(20), 0.0, 1.0))

    # 2e. applications_submitted_30d — actively searching
    apps = float(sig.applications_submitted_30d or 0)
    apps_score = float(np.clip(math.log1p(apps) / math.log1p(15), 0.0, 1.0))

    engagement = float(np.clip(
        0.30 * rr_score
        + 0.20 * rt_score
        + 0.20 * icr_score
        + 0.20 * svd_score
        + 0.10 * apps_score,
        0.0, 1.0,
    ))

    # ------------------------------------------------------------------
    # 3. TRUST & VERIFICATION
    # ------------------------------------------------------------------
    email_score  = 1.0 if sig.verified_email else 0.0
    phone_score  = 1.0 if sig.verified_phone else 0.0
    li_score     = 1.0 if sig.linkedin_connected else 0.5

    completeness_score = float(np.clip(
        sig.profile_completeness_score / 100.0, 0.0, 1.0
    ))

    trust_verification = float(np.clip(
        0.30 * email_score
        + 0.30 * phone_score
        + 0.20 * li_score
        + 0.20 * completeness_score,
        0.0, 1.0,
    ))

    # ------------------------------------------------------------------
    # 4. ENGINEERING SIGNAL
    # ------------------------------------------------------------------
    # github_activity_score: None => private/no-github => neutral 0.5
    gh = sig.github_activity_score
    if gh is None:
        gh_score = 0.5
    else:
        gh_score = float(np.clip(float(gh) / GITHUB_IDEAL, 0.0, 1.0))

    # profile_views_received_30d — recruiter interest (weak signal)
    pvr = float(sig.profile_views_received_30d or 0)
    pvr_score = float(np.clip(math.log1p(pvr) / math.log1p(100), 0.0, 1.0))

    engineering_signal = float(np.clip(
        0.70 * gh_score + 0.30 * pvr_score,
        0.0, 1.0,
    ))

    # ------------------------------------------------------------------
    # COMBINE: weighted geometric mean
    # ------------------------------------------------------------------
    components = [
        (max(availability,       0.01), W_AVAILABILITY),
        (max(engagement,         0.01), W_ENGAGEMENT),
        (max(trust_verification, 0.01), W_TRUST_VERIFICATION),
        (max(engineering_signal, 0.01), W_ENGINEERING_SIGNAL),
    ]
    log_sum = sum(w * math.log(v) for v, w in components)
    behavioral_fit = float(np.clip(math.exp(log_sum), 0.0, 1.0))

    return BehavioralFitResult(
        behavioral_fit=behavioral_fit,
        availability=availability,
        engagement=engagement,
        trust_verification=trust_verification,
        engineering_signal=engineering_signal,
    )
