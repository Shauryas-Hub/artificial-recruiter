"""
scripts/rank.py — Ranking step (≤5 min, CPU only, no network).

Loads the pre-built feature store parquet, computes scores, selects top-100,
generates reasoning, writes submission CSV.

Usage:
  python scripts/rank.py --feature-store artifacts/candidate_features.parquet \
                          --out submission/team_redrob.csv

  # Or with raw candidates (triggers inline feature computation):
  python scripts/rank.py --candidates data/candidates.jsonl \
                          --out submission/team_redrob.csv
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.scorer import rank_candidates
from src.reasoning import add_reasoning_column
from src.csv_writer import write_submission

FEATURE_STORE_DEFAULT = str(
    Path(__file__).parent.parent / "artifacts" / "candidate_features.parquet"
)
OUTPUT_DEFAULT = str(
    Path(__file__).parent.parent / "submission" / "team_redrob.csv"
)


def run_from_feature_store(feature_store_path: str, out_path: str) -> None:
    t0 = time.time()
    print(f"[1/4] Loading feature store: {feature_store_path}")
    df = pd.read_parquet(feature_store_path)
    print(f"      Loaded {len(df)} survivors x {len(df.columns)} cols in {time.time()-t0:.1f}s")

    print("[2/4] Scoring & ranking...")
    top100 = rank_candidates(df)
    print(f"      Ranked. Top score: {top100['score'].iloc[0]:.4f}")

    print("[3/4] Generating reasoning...")
    add_reasoning_column(top100, rank_col="rank")

    print("[4/4] Writing submission CSV...")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    write_submission(top100, out_path)

    total = time.time() - t0
    print(f"\nDone in {total:.1f}s  →  {out_path}")
    print(f"Top-5 candidates:")
    for _, row in top100.head(5).iterrows():
        print(
            f"  #{int(row['rank'])}: {row['candidate_id']} | "
            f"{row.get('current_title','?')} | score={row['score']:.4f}"
        )


def run_from_candidates(candidates_path: str, out_path: str) -> None:
    """
    Inline feature computation for small samples (e.g., sandbox demo).
    Not intended for the full 100K pool within the 5-min budget.
    For full pool: run precompute.py first, then use --feature-store.
    """
    print("[inline] Loading embedding model (requires network on first run)...")
    from sentence_transformers import SentenceTransformer
    from src.cleaning import clean_candidate
    from src.trap_detector import detect_traps_numeric, finalize_trap_result
    from src.skill_features import compute_skill_features
    from src.career_features import compute_career_features
    from src.behavioral_features import compute_behavioral_features
    from src.io_utils import stream_candidates
    import yaml
    import numpy as np
    from datetime import date

    cfg_dir = Path(__file__).parent.parent / "config"
    with open(cfg_dir / "jd_requirements.yaml") as f:
        jd_req = yaml.safe_load(f)

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    cache: dict[str, np.ndarray] = {}

    def embed_fn(texts):
        missing = [t for t in texts if t not in cache]
        if missing:
            vecs = model.encode(missing, normalize_embeddings=True, show_progress_bar=False)
            for t, v in zip(missing, vecs):
                cache[t] = v
        return np.array([cache.get(t, np.zeros(384)) for t in texts])

    rows = []
    honeypot_count = 0
    anchor = date.today()

    for raw in stream_candidates(candidates_path):
        try:
            c = clean_candidate(raw)
            trap = detect_traps_numeric(c)
            if trap.is_honeypot:
                honeypot_count += 1
                continue
            skill_res  = compute_skill_features(c, jd_req, embed_fn)
            career_res = compute_career_features(c, jd_req, embed_fn)
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
            print(f"  WARNING: {raw.get('candidate_id','?')}: {e}")

    print(f"  Processed {len(rows)} candidates ({honeypot_count} honeypots dropped)")
    df = pd.DataFrame(rows)
    top100 = rank_candidates(df)
    add_reasoning_column(top100, rank_col="rank")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    write_submission(top100, out_path)
    print(f"  Written: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Redrob Ranker — Ranking Step")
    parser.add_argument("--feature-store", default=FEATURE_STORE_DEFAULT,
                        help="Path to precomputed parquet feature store")
    parser.add_argument("--candidates", default=None,
                        help="Path to candidates.jsonl (fallback: inline compute, slower)")
    parser.add_argument("--out", default=OUTPUT_DEFAULT,
                        help="Output CSV path")
    args = parser.parse_args()

    if args.candidates and not Path(args.feature_store).exists():
        print(f"Feature store not found. Running inline computation from {args.candidates}...")
        run_from_candidates(args.candidates, args.out)
    else:
        if not Path(args.feature_store).exists():
            print(f"ERROR: Feature store not found: {args.feature_store}")
            print("Run: python scripts/precompute.py --candidates data/candidates.jsonl")
            sys.exit(1)
        run_from_feature_store(args.feature_store, args.out)


if __name__ == "__main__":
    main()
