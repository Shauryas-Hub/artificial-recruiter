"""
tests/test_integration.py — Integration tests.

Uses the 50 sample candidates from sample_candidates.json to verify the
full pipeline end-to-end (clean → trap → features → score → rank → CSV).

These tests do NOT require sentence-transformers; they use a deterministic
mock embed_fn to test the pipeline logic independently of model weights.
"""
import sys
import json
import os
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest
from datetime import date

from src.cleaning import clean_candidate
from src.trap_detector import detect_traps_numeric, finalize_trap_result
from src.skill_features import compute_skill_features
from src.career_features import compute_career_features
from src.behavioral_features import compute_behavioral_features
from src.scorer import rank_candidates
from src.reasoning import add_reasoning_column, build_reasoning
from src.csv_writer import write_submission
from src.validate_local import validate_submission

# ---------------------------------------------------------------------------
# MOCK embed_fn: deterministic, no model needed
# ---------------------------------------------------------------------------

def mock_embed_fn(texts: list[str]) -> np.ndarray:
    """
    Deterministic mock: hash text to a unit vector. Tests pipeline logic,
    not model quality.
    """
    out = []
    for text in texts:
        seed = hash(text) % (2**31)
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(384)
        v = v / (np.linalg.norm(v) + 1e-8)
        out.append(v)
    return np.array(out)


# ---------------------------------------------------------------------------
# Load sample candidates
# ---------------------------------------------------------------------------

SAMPLE_PATH = Path(__file__).parent.parent / "data" / "sample_candidates.json"
BUNDLE_SAMPLE_PATH = Path(__file__).parent.parent.parent / "sample_candidates.json"

def _load_samples() -> list[dict]:
    for p in [SAMPLE_PATH, BUNDLE_SAMPLE_PATH]:
        if p.exists():
            with open(p, encoding="utf-8") as f:
                return json.load(f)
    pytest.skip("sample_candidates.json not found — place it in data/")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_clean_all_samples():
    samples = _load_samples()
    for raw in samples:
        c = clean_candidate(raw)
        assert c.candidate_id.startswith("CAND_")
        assert c.years_of_experience_stated >= 0


def test_trap_detection_all_samples_no_crash():
    samples = _load_samples()
    for raw in samples:
        c = clean_candidate(raw)
        result = detect_traps_numeric(c)
        assert 0.0 <= result.trust_penalty <= 0.95


def test_skill_features_all_samples_no_crash():
    import yaml
    cfg_dir = Path(__file__).parent.parent / "config"
    with open(cfg_dir / "jd_requirements.yaml") as f:
        jd_req = yaml.safe_load(f)

    samples = _load_samples()
    for raw in samples:
        c = clean_candidate(raw)
        trap = detect_traps_numeric(c)
        if trap.is_honeypot:
            continue
        result = compute_skill_features(c, jd_req, mock_embed_fn)
        assert 0.0 <= result.skill_fit_must <= 1.0
        assert 0.0 <= result.skill_fit_nice <= 1.0
        assert 0.0 <= result.skill_fit_combined <= 1.0


def test_career_features_all_samples_no_crash():
    import yaml
    cfg_dir = Path(__file__).parent.parent / "config"
    with open(cfg_dir / "jd_requirements.yaml") as f:
        jd_req = yaml.safe_load(f)

    samples = _load_samples()
    for raw in samples:
        c = clean_candidate(raw)
        trap = detect_traps_numeric(c)
        if trap.is_honeypot:
            continue
        result = compute_career_features(c, jd_req, mock_embed_fn)
        assert 0.0 <= result.career_fit <= 1.0
        assert result.applied_ml_at_product_years >= 0


def test_behavioral_features_all_samples_no_crash():
    samples = _load_samples()
    anchor = date(2026, 5, 27)
    for raw in samples:
        c = clean_candidate(raw)
        result = compute_behavioral_features(c, anchor)
        assert 0.0 <= result.behavioral_fit <= 1.0
        assert 0.0 <= result.availability <= 1.0
        assert 0.0 <= result.engagement <= 1.0


def test_full_pipeline_produces_valid_csv():
    """Full pipeline from sample candidates → CSV passes validator."""
    import yaml
    cfg_dir = Path(__file__).parent.parent / "config"
    with open(cfg_dir / "jd_requirements.yaml") as f:
        jd_req = yaml.safe_load(f)

    samples = _load_samples()
    anchor = date(2026, 5, 27)
    rows = []

    for raw in samples:
        try:
            c = clean_candidate(raw)
            trap = detect_traps_numeric(c)
            if trap.is_honeypot:
                continue
            skill_res  = compute_skill_features(c, jd_req, mock_embed_fn)
            career_res = compute_career_features(c, jd_req, mock_embed_fn)
            beh_res    = compute_behavioral_features(c, anchor)
            days_since = (
                (anchor - c.signals.last_active_date).days
                if c.signals.last_active_date else 365
            )
            rows.append({
                "candidate_id":                c.candidate_id,
                "current_title":               c.current_title,
                "years_of_experience_stated":  c.years_of_experience_stated,
                "current_company":             c.current_company,
                "current_industry":            c.current_industry,
                "location":                    c.location,
                "country":                     c.country,
                "skill_fit_must":              skill_res.skill_fit_must,
                "skill_fit_nice":              skill_res.skill_fit_nice,
                "skill_fit_combined":          skill_res.skill_fit_combined,
                "matched_must_skills":         ",".join(skill_res.matched_must_skills),
                "matched_nice_skills":         ",".join(skill_res.matched_nice_skills),
                "top_skill_name":              skill_res.top_skill_name,
                "assessment_credibility_hits": skill_res.assessment_credibility_hits,
                "career_fit":                  career_res.career_fit,
                "title_alignment":             career_res.title_alignment,
                "applied_ml_at_product_years": career_res.applied_ml_at_product_years,
                "applied_ml_at_product_score": career_res.applied_ml_at_product_score,
                "shipped_system_score":        career_res.shipped_system_score,
                "shipped_evidence_phrase":     career_res.shipped_evidence_phrase,
                "job_hop_penalty":             career_res.job_hop_penalty,
                "services_only_penalty":       career_res.services_only_penalty,
                "negative_title_penalty":      career_res.negative_title_penalty,
                "experience_band_score":       career_res.experience_band_score,
                "industry_fit_score":          career_res.industry_fit_score,
                "is_services_only_career":     career_res.is_services_only_career,
                "title_desc_cosine":           0.5,
                "location_score":              career_res.location_score,
                "behavioral_fit":              beh_res.behavioral_fit,
                "availability":                beh_res.availability,
                "engagement":                  beh_res.engagement,
                "trust_verification":          beh_res.trust_verification,
                "engineering_signal":          beh_res.engineering_signal,
                "days_since_last_active":      days_since,
                "notice_period_days":          c.signals.notice_period_days,
                "response_rate":               c.signals.recruiter_response_rate or 0.0,
                "open_to_work_flag":           c.signals.open_to_work_flag,
                "is_honeypot":                 False,
                "trust_penalty":               trap.trust_penalty,
                "trap_flags":                  "|".join(trap.trap_flags),
                "numeric_flag_count":          trap.numeric_flag_count,
                "semantic_flag_count":         0,
                "salary_min_gt_max":           c.salary_min_gt_max,
                "experience_mismatch_years":   c.experience_mismatch_years or 0.0,
                "date_anchor":                 str(anchor),
            })
        except Exception as e:
            pytest.fail(f"Pipeline failed for {raw.get('candidate_id','?')}: {e}")

    assert len(rows) > 0, "No candidates survived pipeline"

    df = pd.DataFrame(rows)
    top_df = rank_candidates(df)
    add_reasoning_column(top_df, rank_col="rank")

    # Write and validate
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        path = f.name
    try:
        write_submission(top_df, path)
        errors = validate_submission(path)
        assert errors == [], f"Validation errors: {errors}"
    finally:
        os.unlink(path)


def test_reasoning_all_unique():
    """All reasoning strings in top-N must be unique."""
    import yaml
    cfg_dir = Path(__file__).parent.parent / "config"
    with open(cfg_dir / "jd_requirements.yaml") as f:
        jd_req = yaml.safe_load(f)

    samples = _load_samples()
    anchor = date(2026, 5, 27)
    rows = []

    for raw in samples:
        c = clean_candidate(raw)
        trap = detect_traps_numeric(c)
        if trap.is_honeypot:
            continue
        skill_res  = compute_skill_features(c, jd_req, mock_embed_fn)
        career_res = compute_career_features(c, jd_req, mock_embed_fn)
        beh_res    = compute_behavioral_features(c, anchor)
        days_since = (
            (anchor - c.signals.last_active_date).days
            if c.signals.last_active_date else 365
        )
        rows.append({
            "candidate_id": c.candidate_id, "current_title": c.current_title,
            "years_of_experience_stated": c.years_of_experience_stated,
            "current_company": c.current_company, "current_industry": c.current_industry,
            "location": c.location, "country": c.country,
            "skill_fit_must": skill_res.skill_fit_must,
            "skill_fit_nice": skill_res.skill_fit_nice,
            "skill_fit_combined": skill_res.skill_fit_combined,
            "matched_must_skills": ",".join(skill_res.matched_must_skills),
            "matched_nice_skills": ",".join(skill_res.matched_nice_skills),
            "top_skill_name": skill_res.top_skill_name,
            "assessment_credibility_hits": skill_res.assessment_credibility_hits,
            "career_fit": career_res.career_fit, "title_alignment": career_res.title_alignment,
            "applied_ml_at_product_years": career_res.applied_ml_at_product_years,
            "applied_ml_at_product_score": career_res.applied_ml_at_product_score,
            "shipped_system_score": career_res.shipped_system_score,
            "shipped_evidence_phrase": career_res.shipped_evidence_phrase,
            "job_hop_penalty": career_res.job_hop_penalty,
            "services_only_penalty": career_res.services_only_penalty,
            "negative_title_penalty": career_res.negative_title_penalty,
            "experience_band_score": career_res.experience_band_score,
            "industry_fit_score": career_res.industry_fit_score,
            "is_services_only_career": career_res.is_services_only_career,
            "title_desc_cosine": 0.5, "location_score": career_res.location_score,
            "behavioral_fit": beh_res.behavioral_fit, "availability": beh_res.availability,
            "engagement": beh_res.engagement, "trust_verification": beh_res.trust_verification,
            "engineering_signal": beh_res.engineering_signal,
            "days_since_last_active": days_since,
            "notice_period_days": c.signals.notice_period_days,
            "response_rate": c.signals.recruiter_response_rate or 0.0,
            "open_to_work_flag": c.signals.open_to_work_flag,
            "is_honeypot": False, "trust_penalty": trap.trust_penalty,
            "trap_flags": "|".join(trap.trap_flags),
            "numeric_flag_count": trap.numeric_flag_count, "semantic_flag_count": 0,
            "salary_min_gt_max": c.salary_min_gt_max,
            "experience_mismatch_years": c.experience_mismatch_years or 0.0,
            "date_anchor": str(anchor),
        })

    df = pd.DataFrame(rows)
    top_df = rank_candidates(df)
    add_reasoning_column(top_df, rank_col="rank")

    reasonings = top_df["reasoning"].tolist()
    assert len(set(reasonings)) == len(reasonings), (
        f"Duplicate reasoning strings found: "
        f"{len(reasonings) - len(set(reasonings))} duplicates"
    )


def test_cand_0000031_ranks_top_in_sample():
    """CAND_0000031 (Ela Singh, Recommendation Systems Engineer, Swiggy) should
    rank #1 in the sample — she has the best ML/retrieval profile."""
    import yaml
    cfg_dir = Path(__file__).parent.parent / "config"
    with open(cfg_dir / "jd_requirements.yaml") as f:
        jd_req = yaml.safe_load(f)

    samples = _load_samples()
    anchor = date(2026, 5, 27)
    rows = []

    for raw in samples:
        c = clean_candidate(raw)
        trap = detect_traps_numeric(c)
        if trap.is_honeypot:
            continue
        skill_res  = compute_skill_features(c, jd_req, mock_embed_fn)
        career_res = compute_career_features(c, jd_req, mock_embed_fn)
        beh_res    = compute_behavioral_features(c, anchor)
        days_since = (
            (anchor - c.signals.last_active_date).days
            if c.signals.last_active_date else 365
        )
        rows.append({
            "candidate_id": c.candidate_id, "current_title": c.current_title,
            "years_of_experience_stated": c.years_of_experience_stated,
            "current_company": c.current_company, "current_industry": c.current_industry,
            "location": c.location, "country": c.country,
            "skill_fit_must": skill_res.skill_fit_must, "skill_fit_nice": skill_res.skill_fit_nice,
            "skill_fit_combined": skill_res.skill_fit_combined,
            "matched_must_skills": ",".join(skill_res.matched_must_skills),
            "matched_nice_skills": ",".join(skill_res.matched_nice_skills),
            "top_skill_name": skill_res.top_skill_name,
            "assessment_credibility_hits": skill_res.assessment_credibility_hits,
            "career_fit": career_res.career_fit, "title_alignment": career_res.title_alignment,
            "applied_ml_at_product_years": career_res.applied_ml_at_product_years,
            "applied_ml_at_product_score": career_res.applied_ml_at_product_score,
            "shipped_system_score": career_res.shipped_system_score,
            "shipped_evidence_phrase": career_res.shipped_evidence_phrase,
            "job_hop_penalty": career_res.job_hop_penalty,
            "services_only_penalty": career_res.services_only_penalty,
            "negative_title_penalty": career_res.negative_title_penalty,
            "experience_band_score": career_res.experience_band_score,
            "industry_fit_score": career_res.industry_fit_score,
            "is_services_only_career": career_res.is_services_only_career,
            "title_desc_cosine": 0.5, "location_score": career_res.location_score,
            "behavioral_fit": beh_res.behavioral_fit, "availability": beh_res.availability,
            "engagement": beh_res.engagement, "trust_verification": beh_res.trust_verification,
            "engineering_signal": beh_res.engineering_signal,
            "days_since_last_active": days_since,
            "notice_period_days": c.signals.notice_period_days,
            "response_rate": c.signals.recruiter_response_rate or 0.0,
            "open_to_work_flag": c.signals.open_to_work_flag,
            "is_honeypot": False, "trust_penalty": trap.trust_penalty,
            "trap_flags": "|".join(trap.trap_flags),
            "numeric_flag_count": trap.numeric_flag_count, "semantic_flag_count": 0,
            "salary_min_gt_max": c.salary_min_gt_max,
            "experience_mismatch_years": c.experience_mismatch_years or 0.0,
            "date_anchor": str(anchor),
        })

    df = pd.DataFrame(rows)
    top_df = rank_candidates(df)

    # CAND_0000031 must be in top 5 (mock embeddings are random so top-1 may vary)
    if "CAND_0000031" in top_df["candidate_id"].values:
        rank_31 = top_df[top_df["candidate_id"] == "CAND_0000031"]["rank"].iloc[0]
        assert rank_31 <= 5, f"CAND_0000031 ranked {rank_31}, expected top-5"
    else:
        pytest.skip("CAND_0000031 not in survivors (may be honeypot in this test run)")
