"""
sandbox/app.py — Streamlit sandbox demo for Redrob Ranker.

Accepts a small candidate sample (JSON upload or text paste),
runs the full ranking pipeline, displays ranked results.

Deploy to HuggingFace Spaces (Streamlit SDK) or Streamlit Cloud.
"""
from __future__ import annotations
import json
import sys
import io
import tempfile
from pathlib import Path

import streamlit as st
import pandas as pd

# Make src importable
sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Redrob AI Ranker — Demo",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Redrob AI Candidate Ranker")
st.caption("Intelligent Candidate Discovery & Ranking — Hackathon Demo")

# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
**Architecture**
- Phase 1 (Offline): Embedding model encodes skills + career text → feature store
- Phase 2 (Real-time): Pure arithmetic ranking over pre-computed features

**Scoring Formula**
```
raw = (0.7×skill_must + 0.3×skill_nice)
    × career_fit^1.0
    × behavioral_fit^0.8
    × (1 − 0.5×trust_penalty)
```

**Key signals**
- Semantic skill match (4 must-have, 5 nice-to-have)
- Applied ML years at product companies
- Shipped-system evidence (Tier-5 path)
- Behavioral availability & engagement
- Honeypot & trap detection
    """)
    st.divider()
    st.markdown("**Compute constraints met:**")
    st.success("✅ ≤5 min ranking step")
    st.success("✅ CPU only")
    st.success("✅ No network during ranking")
    st.success("✅ ≤16 GB RAM")

# ---------------------------------------------------------------------------
# INPUT
# ---------------------------------------------------------------------------
st.header("1. Upload Candidates")

col1, col2 = st.columns([1, 1])

with col1:
    uploaded = st.file_uploader(
        "Upload candidates JSON/JSONL (≤500 candidates for demo)",
        type=["json", "jsonl"],
        help="Upload a JSON array or JSONL file of candidate records."
    )

with col2:
    st.markdown("**Or paste a JSON array:**")
    pasted = st.text_area(
        "Paste candidate JSON array",
        height=150,
        placeholder='[{"candidate_id": "CAND_0000001", ...}, ...]',
    )

# ---------------------------------------------------------------------------
# LOAD CANDIDATES
# ---------------------------------------------------------------------------
candidates_raw = []

if uploaded is not None:
    try:
        content = uploaded.read().decode("utf-8")
        # Try JSON array first, then JSONL
        try:
            data = json.loads(content)
            if isinstance(data, list):
                candidates_raw = data
            else:
                candidates_raw = [data]
        except json.JSONDecodeError:
            for line in content.splitlines():
                line = line.strip()
                if line:
                    candidates_raw.append(json.loads(line))
        st.success(f"Loaded {len(candidates_raw)} candidates from upload.")
    except Exception as e:
        st.error(f"Failed to parse upload: {e}")

elif pasted.strip():
    try:
        candidates_raw = json.loads(pasted.strip())
        if not isinstance(candidates_raw, list):
            candidates_raw = [candidates_raw]
        st.success(f"Loaded {len(candidates_raw)} candidates from paste.")
    except Exception as e:
        st.error(f"Invalid JSON: {e}")

# Limit for demo
if len(candidates_raw) > 500:
    st.warning(f"Demo limited to 500 candidates. Showing first 500 of {len(candidates_raw)}.")
    candidates_raw = candidates_raw[:500]

# ---------------------------------------------------------------------------
# RUN RANKING
# ---------------------------------------------------------------------------
if candidates_raw and st.button("🚀 Run Ranking", type="primary"):
    from datetime import date
    import numpy as np
    import yaml

    cfg_dir = Path(__file__).parent.parent / "config"

    with st.spinner("Loading model & computing features..."):
        try:
            from sentence_transformers import SentenceTransformer
            from src.cleaning import clean_candidate
            from src.trap_detector import detect_traps_numeric, finalize_trap_result
            from src.skill_features import compute_skill_features
            from src.career_features import compute_career_features
            from src.behavioral_features import compute_behavioral_features
            from src.scorer import rank_candidates
            from src.reasoning import add_reasoning_column

            with open(cfg_dir / "jd_requirements.yaml") as f:
                jd_req = yaml.safe_load(f)

            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            cache: dict = {}

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
            errors = []

            progress = st.progress(0, text="Processing candidates...")
            n = len(candidates_raw)

            for idx, raw in enumerate(candidates_raw):
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
                    errors.append(f"{raw.get('candidate_id','?')}: {e}")

                progress.progress((idx + 1) / n, text=f"Processing {idx+1}/{n}...")

            progress.empty()

            if errors:
                st.warning(f"{len(errors)} candidates failed to process: {errors[:3]}")

            if not rows:
                st.error("No candidates survived processing. Check your input data.")
                st.stop()

            df = pd.DataFrame(rows)
            top_n = min(100, len(df))

            # Rank
            from src.scorer import compute_raw_scores, minmax_normalize
            import numpy as np
            df["_raw_score"] = compute_raw_scores(df)
            df["_norm_score"] = minmax_normalize(df["_raw_score"].to_numpy())
            df_sorted = df.sort_values(["_norm_score", "candidate_id"], ascending=[False, True]).reset_index(drop=True)
            top_df = df_sorted.head(top_n).copy()
            top_df["rank"] = range(1, top_n + 1)
            top_df["score"] = top_df["_norm_score"].round(4)
            add_reasoning_column(top_df, rank_col="rank")

            st.success(f"✅ Ranked {len(rows)} candidates ({honeypot_count} honeypots dropped)")

        except Exception as e:
            st.error(f"Ranking failed: {e}")
            st.exception(e)
            st.stop()

    # ------------------------------------------------------------------
    # RESULTS DISPLAY
    # ------------------------------------------------------------------
    st.header("2. Results")

    # Summary metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Processed", len(candidates_raw))
    m2.metric("Survivors", len(rows))
    m3.metric("Honeypots Dropped", honeypot_count)
    m4.metric("Top N Returned", len(top_df))

    st.divider()

    # Top candidates table
    display_cols = [
        "rank", "candidate_id", "current_title", "years_of_experience_stated",
        "location", "country", "score",
        "skill_fit_must", "career_fit", "behavioral_fit", "trust_penalty",
        "applied_ml_at_product_years", "notice_period_days",
    ]
    available = [c for c in display_cols if c in top_df.columns]
    st.dataframe(
        top_df[available].style.format({
            "score": "{:.4f}",
            "skill_fit_must": "{:.3f}",
            "career_fit": "{:.3f}",
            "behavioral_fit": "{:.3f}",
            "trust_penalty": "{:.2f}",
            "applied_ml_at_product_years": "{:.1f}",
        }),
        use_container_width=True,
        height=500,
    )

    # Top-10 reasoning
    st.subheader("Top-10 Reasoning")
    for _, row in top_df.head(10).iterrows():
        with st.expander(
            f"#{int(row['rank'])}: {row['candidate_id']} — {row.get('current_title','?')} "
            f"(score={row['score']:.4f})"
        ):
            st.write(row.get("reasoning", "No reasoning generated."))
            score_cols = ["skill_fit_must", "career_fit", "behavioral_fit", "trust_penalty"]
            st.json({c: round(float(row[c]), 4) for c in score_cols if c in row})

    # Download
    st.divider()
    csv_buf = io.StringIO()
    out_df = top_df[["candidate_id", "rank", "score", "reasoning"]].copy()
    out_df.to_csv(csv_buf, index=False)
    st.download_button(
        label="⬇️ Download submission.csv",
        data=csv_buf.getvalue().encode("utf-8"),
        file_name="team_redrob.csv",
        mime="text/csv",
    )

elif not candidates_raw:
    st.info("Upload candidate data above to get started.")
