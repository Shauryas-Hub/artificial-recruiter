"""
src/skill_features.py — Skill-Fit feature engineering.

IMPROVEMENTS over v1:
  - Keyword alias scan layer: exact substring match in skill name gets a
    keyword_boost multiplier on top of the cosine score. This ensures that
    a candidate who literally lists "Pinecone" or "FAISS" is not undersold
    by a 0.45-threshold cosine gate.
  - Description scan: career descriptions are also scanned for must-have
    keywords, contributing a weaker "evidence_from_description" signal.
    This implements the Tier-5 path more completely.
  - trust_formula uses +1 inside log to prevent log(0) with 0 endorsements.
  - cosine_match_threshold lowered to 0.42 to catch more semantic variants.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from src.cleaning import CleanCandidate

# ---------------------------------------------------------------------------
# CONSTANTS — mirror scoring_config.yaml -> skill_fit
# ---------------------------------------------------------------------------
COSINE_MATCH_THRESHOLD = 0.42
TRUST_FLOOR = 0.05
ASSESSMENT_CREDIBILITY_THRESHOLD = 50
ASSESSMENT_CREDIBILITY_PENALTY = 0.4
MUST_HAVE_WEIGHT = 0.70
NICE_TO_HAVE_WEIGHT = 0.30
KEYWORD_BOOST = 0.15          # additive boost when exact alias found in skill name
DESC_KEYWORD_BOOST = 0.08     # weaker boost when keyword found in description


@dataclass
class SkillFitResult:
    skill_fit_must: float
    skill_fit_nice: float
    skill_fit_combined: float
    matched_must_skills: list[str] = field(default_factory=list)
    matched_nice_skills: list[str] = field(default_factory=list)
    top_skill_name: str = ""
    assessment_credibility_hits: int = 0
    keyword_boost_applied: bool = False


def trust_multiplier(proficiency_ordinal: int, duration_months: int, endorsements: int) -> float:
    """
    Trust multiplier: proficiency * log1p(duration) * log1p(endorsements + 1)
    FIX: endorsements+1 inside log prevents log(1)=0 killing trust for 0-endorsement
    legitimate skills.
    """
    val = (
        max(proficiency_ordinal, 0)
        * math.log1p(max(duration_months, 0))
        * math.log1p(max(endorsements, 0) + 1)
    )
    return max(val, TRUST_FLOOR)


def _keyword_match(text: str, aliases: list[str]) -> bool:
    """Case-insensitive substring match of any alias in text."""
    text_lower = text.lower()
    return any(alias.lower() in text_lower for alias in aliases)


def _desc_keyword_coverage(career_history, aliases: list[str]) -> float:
    """
    Fraction of career descriptions that mention any alias.
    Returns 0..1 proportional signal.
    """
    if not career_history or not aliases:
        return 0.0
    hits = sum(
        1 for r in career_history
        if _keyword_match(r.description, aliases)
    )
    return min(1.0, hits / max(len(career_history), 1))


def compute_skill_features(
    cand: CleanCandidate,
    jd_requirements: dict,
    embed_fn,
) -> SkillFitResult:
    """
    Full skill-feature computation using an embedding function.
    embed_fn returns L2-normalized row vectors (dot product == cosine).
    """
    skills = [s for s in cand.skills if s.name]
    n_skills = len(skills)

    if n_skills == 0:
        return SkillFitResult(0.0, 0.0, 0.0)

    skill_texts = [s.name for s in skills]
    cand_emb = embed_fn(skill_texts)   # (n_skills, dim)
    trust_weights = np.array([
        trust_multiplier(s.proficiency_ordinal, s.duration_months, s.endorsements)
        for s in skills
    ], dtype=np.float64)

    cred_inc = [0]

    def _coverage(requirements_list: list[dict]) -> tuple[float, list[str], bool]:
        """
        Return (weighted_mean_coverage, matched_names, keyword_boost_applied).
        """
        if not requirements_list:
            return 0.0, [], False
        coverages: list[float] = []
        matched: list[str] = []
        any_keyword_boost = False

        for req in requirements_list:
            aliases = [req["name"]] + list(req.get("aliases", []))
            req_emb = embed_fn(aliases)    # (n_aliases, dim)

            # Semantic cosine coverage
            sim = cand_emb @ req_emb.T    # (n_skills, n_aliases)
            best_per_skill = sim.max(axis=1)   # (n_skills,)
            mask = best_per_skill >= COSINE_MATCH_THRESHOLD

            # Keyword alias exact scan over skill names (boost layer)
            skill_keyword_hits = np.array([
                _keyword_match(s.name, req.get("aliases", []))
                for s in skills
            ], dtype=bool)
            # Also count name itself
            skill_keyword_hits |= np.array([
                _keyword_match(s.name, [req["name"]])
                for s in skills
            ], dtype=bool)

            # Combine: semantic mask OR keyword hit
            combined_mask = mask | skill_keyword_hits

            if not combined_mask.any():
                # Try description scan as fallback (weak signal)
                desc_cov = _desc_keyword_coverage(
                    cand.career_history, req.get("aliases", [])
                )
                coverages.append(min(DESC_KEYWORD_BOOST, desc_cov * DESC_KEYWORD_BOOST))
                continue

            # Credibility check on best matching skill
            best_skill_local = int(np.argmax(
                (best_per_skill * combined_mask.astype(float)) * trust_weights
            ))
            best_skill_name = skills[best_skill_local].name
            assessment = cand.signals.skill_assessment_scores.get(best_skill_name)

            contribution = float((trust_weights * combined_mask * np.maximum(best_per_skill, 0.01)).sum())
            total = float(trust_weights.sum())
            cov = contribution / total if total > 0 else 0.0

            # Keyword boost additive
            if skill_keyword_hits.any():
                cov = min(1.0, cov + KEYWORD_BOOST)
                any_keyword_boost = True

            # Assessment credibility penalty
            if assessment is not None and assessment < ASSESSMENT_CREDIBILITY_THRESHOLD:
                cov *= ASSESSMENT_CREDIBILITY_PENALTY
                cred_inc[0] += 1

            coverages.append(min(cov, 1.0))
            matched.append(f"{req['name']}~{best_skill_name}")

        mean_cov = float(np.mean(coverages)) if coverages else 0.0
        return mean_cov, matched, any_keyword_boost

    must_cov, matched_must, kb_must = _coverage(
        jd_requirements.get("must_have_skills", [])
    )
    nice_cov, matched_nice, kb_nice = _coverage(
        jd_requirements.get("nice_to_have_skills", [])
    )

    combined = MUST_HAVE_WEIGHT * must_cov + NICE_TO_HAVE_WEIGHT * nice_cov

    # Top skill: highest trust weight
    top_skill_name = skills[int(np.argmax(trust_weights))].name if n_skills > 0 else ""

    return SkillFitResult(
        skill_fit_must=must_cov,
        skill_fit_nice=nice_cov,
        skill_fit_combined=combined,
        matched_must_skills=matched_must,
        matched_nice_skills=matched_nice,
        top_skill_name=top_skill_name,
        assessment_credibility_hits=cred_inc[0],
        keyword_boost_applied=(kb_must or kb_nice),
    )
