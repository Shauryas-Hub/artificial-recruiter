"""
scripts/precompute.py — Offline feature store builder.

Run once (no time limit). Produces:
  artifacts/candidate_features.parquet  — 99K+ rows x 42 cols
  artifacts/honeypots_excluded.csv      — dropped honeypots
  artifacts/feature_store_schema.json   — metadata

Usage:
  python scripts/precompute.py --candidates data/candidates.jsonl \
                                --out-dir artifacts/

This script uses sentence-transformers (network allowed, no ranking constraints).
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from datetime import date
from pathlib import Path

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.cleaning import clean_candidate
from src.trap_detector import detect_traps_numeric, finalize_trap_result
from src.skill_features import compute_skill_features
from src.career_features import compute_career_features
from src.behavioral_features import compute_behavioral_features

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
EMBEDDING_MODEL    = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE         = 256
TITLE_DESC_COSINE_THRESHOLD = 0.15   # below => semantic mismatch flag

ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"
CONFIG_DIR    = Path(__file__).parent.parent / "config"


def load_config(name: str) -> dict:
    import yaml  # type: ignore
    with open(CONFIG_DIR / name, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class EmbedFn:
    """Wrapper around SentenceTransformer for batch embedding."""

    def __init__(self, model: SentenceTransformer):
        self._model = model
        self._cache: dict[str, np.ndarray] = {}

    def __call__(self, texts: list[str]) -> np.ndarray:
        unique = list(dict.fromkeys(t for t in texts if t))
        missing = [t for t in unique if t not in self._cache]
        if missing:
            vecs = self._model.encode(
                missing,
                batch_size=BATCH_SIZE,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            for t, v in zip(missing, vecs):
                self._cache[t] = v
        out = np.array([self._cache.get(t, np.zeros(384)) for t in texts])
        return out


def batch_encode_unique_texts(
    model: SentenceTransformer,
    texts: list[str],
) -> dict[str, np.ndarray]:
    """Batch-encode all unique texts; return mapping text -> vector."""
    unique = list(dict.fromkeys(t for t in texts if t))
    print(f"      Encoding {len(unique)} unique texts in batches of {BATCH_SIZE}...")
    t0 = time.time()
    vecs = model.encode(
        unique,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    print(f"      Encoded {len(unique)} unique texts in {time.time()-t0:.1f}s")
    return dict(zip(unique, vecs))


def main():
    parser = argparse.ArgumentParser(description="Redrob Ranker — Precompute Feature Store")
    parser.add_argument(
        "--candidates",
        default=str(Path(__file__).parent.parent / "data" / "candidates.jsonl"),
        help="Path to candidates.jsonl or .jsonl.gz",
    )
    parser.add_argument(
        "--out-dir",
        default=str(ARTIFACTS_DIR),
        help="Output directory for artifacts",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t_total = time.time()

    # ------------------------------------------------------------------ #
    # [1/7] Load configs
    # ------------------------------------------------------------------ #
    print("[1/7] Loading configs...")
    jd_req     = load_config("jd_requirements.yaml")
    scoring_cfg = load_config("scoring_config.yaml")

    # ------------------------------------------------------------------ #
    # [2/7] Load embedding model
    # ------------------------------------------------------------------ #
    print(f"[2/7] Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"      Model loaded. dim={model.get_sentence_embedding_dimension()}")

    # ------------------------------------------------------------------ #
    # [3/7] Stream & clean candidates; collect texts for batch embed
    # ------------------------------------------------------------------ #
    print("[3/7] Streaming candidates (cleaning + trap detection + text collection)...")
    from src.io_utils import stream_candidates

    cleaned_candidates = []
    numeric_trap_results = []
    all_texts: list[str] = []
    title_texts: list[str] = []
    desc_texts: list[str] = []

    for i, raw in enumerate(stream_candidates(args.candidates), 1):
        try:
            c = clean_candidate(raw)
            trap = detect_traps_numeric(c)
            cleaned_candidates.append(c)
            numeric_trap_results.append(trap)
            all_texts.append(c.primary_text())
            title_texts.append(c.current_title or "")
            # Concatenated descriptions for semantic title-desc check
            desc_combined = " ".join(r.description for r in c.career_history if r.description)
            desc_texts.append(desc_combined)
        except Exception as e:
            print(f"  WARNING: failed to clean candidate {raw.get('candidate_id','?')}: {e}")
            continue

        if i % 20000 == 0:
            print(f"      ...{i} candidates cleaned ({len(set(all_texts))} unique texts so far)")

    print(f"      Total cleaned: {len(cleaned_candidates)}")

    # Determine date anchor
    last_active_dates = [
        c.signals.last_active_date
        for c in cleaned_candidates
        if c.signals.last_active_date is not None
    ]
    date_anchor = max(last_active_dates) if last_active_dates else date.today()
    print(f"      Date anchor (max last_active_date): {date_anchor}")

    # ------------------------------------------------------------------ #
    # [4/7] Batch encode all unique texts
    # ------------------------------------------------------------------ #
    print(f"[4/7] Batch encoding texts...")
    all_unique_texts = list(set(t for t in all_texts + title_texts + desc_texts if t))
    text_to_vec = batch_encode_unique_texts(model, all_unique_texts)

    def embed_fn(texts: list[str]) -> np.ndarray:
        return np.array([
            text_to_vec.get(t, np.zeros(384)) for t in texts
        ])

    # ------------------------------------------------------------------ #
    # [5/7] Compute features per candidate
    # ------------------------------------------------------------------ #
    print("[5/7] Computing skill/career/behavioral features...")
    t0 = time.time()

    rows: list[dict] = []
    honeypots: list[dict] = []

    for i, (c, numeric_trap) in enumerate(zip(cleaned_candidates, numeric_trap_results), 1):
        try:
            # Semantic title-description mismatch check
            t_emb = text_to_vec.get(c.current_title or "", np.zeros(384))
            d_emb = text_to_vec.get(
                " ".join(r.description for r in c.career_history if r.description),
                np.zeros(384),
            )
            title_desc_cos = float(np.dot(t_emb, d_emb))
            sem_flag = title_desc_cos < TITLE_DESC_COSINE_THRESHOLD and c.current_title != ""

            trap = finalize_trap_result(
                numeric_trap,
                semantic_flag=sem_flag,
                semantic_flag_label=f"title_desc_cosine:{title_desc_cos:.3f}",
            )

            # Honeypot hard drop
            if trap.is_honeypot:
                honeypots.append({
                    "candidate_id": c.candidate_id,
                    "current_title": c.current_title,
                    "trap_flags": "|".join(trap.trap_flags),
                    "trust_penalty": trap.trust_penalty,
                })
                continue

            # Skill features
            skill_res = compute_skill_features(c, jd_req, embed_fn)

            # Career features
            career_res = compute_career_features(c, jd_req, embed_fn)

            # Behavioral features
            beh_res = compute_behavioral_features(c, date_anchor)

            days_since = (
                (date_anchor - c.signals.last_active_date).days
                if c.signals.last_active_date
                else 365
            )

            rows.append({
                "candidate_id":                c.candidate_id,
                "current_title":               c.current_title,
                "years_of_experience_stated":  c.years_of_experience_stated,
                "current_company":             c.current_company,
                "current_industry":            c.current_industry,
                "location":                    c.location,
                "country":                     c.country,
                # Skill
                "skill_fit_must":              skill_res.skill_fit_must,
                "skill_fit_nice":              skill_res.skill_fit_nice,
                "skill_fit_combined":          skill_res.skill_fit_combined,
                "matched_must_skills":         ",".join(skill_res.matched_must_skills),
                "matched_nice_skills":         ",".join(skill_res.matched_nice_skills),
                "top_skill_name":              skill_res.top_skill_name,
                "assessment_credibility_hits": skill_res.assessment_credibility_hits,
                # Career
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
                "title_desc_cosine":           title_desc_cos,
                "location_score":              career_res.location_score,
                # Behavioral
                "behavioral_fit":              beh_res.behavioral_fit,
                "availability":                beh_res.availability,
                "engagement":                  beh_res.engagement,
                "trust_verification":          beh_res.trust_verification,
                "engineering_signal":          beh_res.engineering_signal,
                "days_since_last_active":      days_since,
                "notice_period_days":          c.signals.notice_period_days,
                "response_rate":               c.signals.recruiter_response_rate or 0.0,
                "open_to_work_flag":           c.signals.open_to_work_flag,
                # Trust
                "is_honeypot":                 False,
                "trust_penalty":               trap.trust_penalty,
                "trap_flags":                  "|".join(trap.trap_flags),
                "numeric_flag_count":          trap.numeric_flag_count,
                "semantic_flag_count":         trap.semantic_flag_count,
                # Quality flags
                "salary_min_gt_max":           c.salary_min_gt_max,
                "experience_mismatch_years":   c.experience_mismatch_years or 0.0,
                "date_anchor":                 str(date_anchor),
            })

        except Exception as e:
            print(f"  WARNING: feature computation failed for {c.candidate_id}: {e}")
            continue

        if i % 5000 == 0:
            elapsed = time.time() - t0
            rate = i / elapsed
            eta = (len(cleaned_candidates) - i) / rate if rate > 0 else 0
            print(f"      ...{i}/{len(cleaned_candidates)} processed ({rate:.0f}/s, ETA {eta:.0f}s)")

    elapsed = time.time() - t0
    print(f"      Feature computation done in {elapsed:.1f}s")
    print(f"      Survivors: {len(rows)} | Honeypots dropped: {len(honeypots)}")

    # ------------------------------------------------------------------ #
    # [6/7] Write feature store
    # ------------------------------------------------------------------ #
    print(f"[6/7] Writing feature store...")
    df = pd.DataFrame(rows)
    parquet_path = out_dir / "candidate_features.parquet"
    df.to_parquet(str(parquet_path), index=False)
    print(f"      Wrote {len(df)} rows x {len(df.columns)} cols → {parquet_path}")

    hp_path = out_dir / "honeypots_excluded.csv"
    pd.DataFrame(honeypots).to_csv(str(hp_path), index=False)
    print(f"[6/7] Logged {len(honeypots)} honeypots → {hp_path}")

    # ------------------------------------------------------------------ #
    # [7/7] Schema
    # ------------------------------------------------------------------ #
    schema = {
        "n_candidates_input": len(cleaned_candidates) + len(honeypots),
        "n_survivors": len(rows),
        "n_honeypots_dropped": len(honeypots),
        "date_anchor": str(date_anchor),
        "embedding_model": EMBEDDING_MODEL,
        "columns": {col: str(df[col].dtype) for col in df.columns},
    }
    schema_path = out_dir / "feature_store_schema.json"
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)
    print(f"[7/7] Schema → {schema_path}")

    total_time = time.time() - t_total
    print(f"\nDONE in {total_time:.1f}s")
    print(f"  Feature store: {parquet_path}")
    print(f"  Honeypots log: {hp_path}")
    print(f"  Schema:        {schema_path}")


if __name__ == "__main__":
    main()
