"""
src/trap_detector.py — Honeypot & trap detection.

Single responsibility: given a CleanCandidate, return a TrapResult containing:
  - is_honeypot: bool       -> hard drop (>= HARD_HONEYPOT_FLAG_COUNT)
  - trust_penalty: float    -> in [0,1]
  - trap_flags: list[str]   -> human-readable

HARD flags (genuine impossibilities — count toward honeypot drop):
  skill_stuffing, mass_expert, date_contradiction, impossible_tenure,
  activity_impossible (gap > 365d)

SOFT flags (common noise — nudge trust_penalty only, never trigger drop):
  salary_inverted, experience_mismatch, title_desc_cosine

The semantic title-vs-description flag is added in a second pass in
precompute.py once embeddings are computed.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from src.cleaning import CleanCandidate

# ---------------------------------------------------------------------------
# THRESHOLDS — mirror scoring_config.yaml -> trap_detection
# ---------------------------------------------------------------------------
TENURE_MISMATCH_MONTHS = 3
PROF_MIN_ORDINAL = 3
PROF_MAX_DURATION_MONTHS = 3
PROF_MAX_ENDORSEMENTS = 2
EXPERIENCE_MISMATCH_YEARS = 2.0
MASS_EXPERT_THRESHOLD = 4
MASS_EXPERT_MAX_BACKING = 2
PER_FLAG_PENALTY = 0.15
MAX_TRUST_PENALTY = 0.95
HARD_HONEYPOT_FLAG_COUNT = 2
ACTIVITY_IMPOSSIBLE_MIN_GAP_DAYS = 365
IMPOSSIBLE_TENURE_GAP_MONTHS = 12

HARD_FLAG_PREFIXES = (
    "skill_stuffing",
    "mass_expert",
    "date_contradiction",
    "impossible_tenure",
    "activity_impossible",
)


@dataclass
class TrapResult:
    is_honeypot: bool
    trust_penalty: float
    trap_flags: list[str] = field(default_factory=list)
    numeric_flag_count: int = 0
    semantic_flag_count: int = 0

    def effective_flag_count(self) -> int:
        return self.numeric_flag_count + self.semantic_flag_count


def _backing_score(skill) -> float:
    return skill.endorsements + skill.duration_months / 12.0


def detect_traps_numeric(c: CleanCandidate) -> TrapResult:
    """First-pass detector: pure numeric/structural flags (no model required)."""
    flags: list[str] = []

    # (a) tenure mismatch: stated duration != computed from dates
    tenure_mismatches = [
        r.tenure_mismatch_months
        for r in c.career_history
        if r.tenure_mismatch_months is not None
        and r.tenure_mismatch_months > TENURE_MISMATCH_MONTHS
    ]
    if tenure_mismatches:
        flags.append(f"tenure_mismatch:{max(tenure_mismatches)}mo")

    # (b) proficiency-vs-duration stuffing: claimed expert/advanced with no backing
    stuffed = [
        s.name
        for s in c.skills
        if s.proficiency_ordinal >= PROF_MIN_ORDINAL
        and s.duration_months <= PROF_MAX_DURATION_MONTHS
        and s.endorsements <= PROF_MAX_ENDORSEMENTS
    ]
    if stuffed:
        flags.append(f"skill_stuffing:{len(stuffed)}skills")

    # (c) experience mismatch (informational flag from cleaning)
    if c.experience_mismatch_years is not None and c.experience_mismatch_years > EXPERIENCE_MISMATCH_YEARS:
        flags.append(f"experience_mismatch:{c.experience_mismatch_years:.1f}yr")

    # (d) mass-expert anomaly
    weak_experts = [
        s for s in c.skills
        if s.proficiency_ordinal >= 4 and _backing_score(s) <= MASS_EXPERT_MAX_BACKING
    ]
    if len(weak_experts) >= MASS_EXPERT_THRESHOLD:
        flags.append(f"mass_expert:{len(weak_experts)}")

    # (e) activity impossible: last_active before signup by a large gap
    sig = c.signals
    if sig.signup_date and sig.last_active_date and sig.last_active_date < sig.signup_date:
        gap_days = (sig.signup_date - sig.last_active_date).days
        if gap_days > ACTIVITY_IMPOSSIBLE_MIN_GAP_DAYS:
            flags.append(f"activity_impossible:{gap_days}d")

    # (e2) date contradiction: career role with end < start
    end_before_start = [
        r for r in c.career_history
        if r.start_date and r.end_date and r.end_date < r.start_date
    ]
    if end_before_start:
        flags.append(f"date_contradiction:{len(end_before_start)}roles_end_before_start")

    # (e3) impossible tenure: stated duration wildly inconsistent with span
    impossible_tenures = [
        r for r in c.career_history
        if r.tenure_mismatch_months is not None
        and r.tenure_mismatch_months > IMPOSSIBLE_TENURE_GAP_MONTHS
    ]
    if impossible_tenures:
        flags.append(
            f"impossible_tenure:{len(impossible_tenures)}roles:"
            f"{max(r.tenure_mismatch_months for r in impossible_tenures)}mo"
        )

    # (f) salary inversion (minor soft flag)
    if c.salary_min_gt_max:
        flags.append("salary_inverted")

    # ---- finalize ----
    hard_flags = [f for f in flags if f.startswith(HARD_FLAG_PREFIXES)]
    soft_flags = [f for f in flags if not f.startswith(HARD_FLAG_PREFIXES)]
    hard_count = len(hard_flags)

    penalty = min(
        MAX_TRUST_PENALTY,
        hard_count * PER_FLAG_PENALTY + len(soft_flags) * 0.03,
    )
    is_honeypot = hard_count >= HARD_HONEYPOT_FLAG_COUNT

    return TrapResult(
        is_honeypot=is_honeypot,
        trust_penalty=penalty,
        trap_flags=flags,
        numeric_flag_count=len(flags),
        semantic_flag_count=0,
    )


def finalize_trap_result(
    numeric_result: TrapResult,
    semantic_flag: bool = False,
    semantic_flag_label: str = "",
) -> TrapResult:
    """
    Second pass: incorporate the semantic title-vs-description mismatch flag.
    Semantic flag is SOFT — does NOT count toward honeypot drop threshold.
    """
    flags = list(numeric_result.trap_flags)
    extra_penalty = 0.0
    sem_count = 0
    if semantic_flag:
        flags.append(semantic_flag_label or "title_desc_semantic_mismatch")
        extra_penalty = PER_FLAG_PENALTY
        sem_count = 1

    hard_flags = [f for f in flags if f.startswith(HARD_FLAG_PREFIXES)]
    hard_count = len(hard_flags)
    soft_numeric = [
        f for f in numeric_result.trap_flags
        if not f.startswith(HARD_FLAG_PREFIXES)
    ]
    penalty = min(
        MAX_TRUST_PENALTY,
        hard_count * PER_FLAG_PENALTY + len(soft_numeric) * 0.03 + extra_penalty,
    )
    is_honeypot = hard_count >= HARD_HONEYPOT_FLAG_COUNT

    return TrapResult(
        is_honeypot=is_honeypot,
        trust_penalty=penalty,
        trap_flags=flags,
        numeric_flag_count=numeric_result.numeric_flag_count,
        semantic_flag_count=sem_count,
    )
