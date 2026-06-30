# Redrob AI Candidate Ranker

**Redrob Hackathon — Intelligent Candidate Discovery & Ranking Challenge**

> Ranks 100,000 candidates for a Senior AI Engineer role the way a great recruiter would — not by keyword matching, but by genuinely understanding who fits for the role.

---

## Architecture

```
Phase 1 — Precompute 1
  candidates.jsonl
       │
       ▼
  [Cleaning]        ← normalize sentinel values, parse dates, compute tenure
       │
       ▼
  [Trap Detector]   ← numeric: skill_stuffing, mass_expert, date_contradiction,
       │              impossible_tenure, activity_impossible
       │              semantic: title-vs-description cosine mismatch
       │              honeypots (≥2 hard flags) → excluded
       ▼
  [Embedding]       ← sentence-transformers/all-MiniLM-L6-v2 (dim=384)
       │              batched, cached — runs once on all unique texts
       ▼
  [Feature Store]   ← skill_fit + career_fit + behavioral_fit per survivor
  candidate_features.parquet (99K+ rows × 42 cols)


Phase 2 — Rank (≤5 min, CPU only, no network)
  candidate_features.parquet
       │
       ▼
  [Scorer]          ← raw = (0.7×skill_must + 0.3×skill_nice)
       │                    × career_fit^1.0
       │                    × behavioral_fit^0.8
       │                    × (1 − 0.5×trust_penalty)
       │              min-max normalize over all survivors
       │              sort by (score DESC, candidate_id ASC)
       ▼
  [Reasoning]       ← varied opening clause per candidate's strongest signal
       │              honest concerns always included when present
       │              no LLM — all facts from feature columns
       ▼
  submission/team_redrob.csv
```

---

## Scoring Formula

| Component | Role | Weight |
|---|---|---|
| `skill_fit_must` | Semantic coverage of 4 must-have JD skills | W_S = 0.70 |
| `skill_fit_nice` | Semantic coverage of 5 nice-to-have skills | W_N = 0.30 |
| `career_fit` | Title + ML years at product cos + shipped evidence − penalties | exponent α = 1.0 |
| `behavioral_fit` | Geometric mean of availability, engagement, trust, GitHub | exponent β = 0.80 |
| `trust_penalty` | Trap flags (0.15 per hard flag, capped at 0.95) | λ = 0.50 |

### Career-fit sub-weights (sum = 1.0)

| Signal | Weight |
|---|---|
| Applied ML years at product companies | 0.38 |
| Title alignment (semantic) | 0.28 |
| Shipped-system evidence (Tier-5 path) | 0.20 |
| Experience band (ideal 6-8yr) | 0.08 |
| Industry fit | 0.06 |

Penalties subtracted independently: job-hop (max 0.20), services-only (max 0.25), non-technical title (max 0.35).

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Place the candidate pool

```bash
cp /path/to/candidates.jsonl data/candidates.jsonl
# or gzipped:
cp /path/to/candidates.jsonl.gz data/candidates.jsonl.gz
```

### 3. Precompute feature store (one-time, ~7 min, network OK)

```bash
python scripts/precompute.py --candidates data/candidates.jsonl --out-dir artifacts/
```

Produces:
- `artifacts/candidate_features.parquet`
- `artifacts/honeypots_excluded.csv`
- `artifacts/feature_store_schema.json`

### 4. Rank (≤5 sec, CPU only, no network)

```bash
python scripts/rank.py --feature-store artifacts/candidate_features.parquet \
                        --out submission/team_redrob.csv
```

### 5. Validate submission

```bash
python validate_submission.py submission/team_redrob.csv
# or with honeypot cross-check:
python -m src.validate_local submission/team_redrob.csv artifacts/honeypots_excluded.csv
```

---

## Reproduce Command (for Stage 3)

```bash
python scripts/rank.py --feature-store artifacts/candidate_features.parquet --out submission/team_redrob.csv
```

Pre-computation (required first, allowed to exceed 5 min):
```bash
python scripts/precompute.py --candidates data/candidates.jsonl
```

---

## Run Tests

```bash
# All tests (no model required — uses mock embeddings)
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## Project Structure

```
redrob-ranker/
├── config/
│   ├── jd_requirements.yaml       ← JD signals, must/nice skills, disqualifiers
│   └── scoring_config.yaml        ← all weights, thresholds, constants
├── src/
│   ├── cleaning.py                ← raw → CleanCandidate (sentinel resolution, dates)
│   ├── trap_detector.py           ← honeypot & trust-penalty detection
│   ├── skill_features.py          ← semantic skill coverage + keyword boost
│   ├── career_features.py         ← title, ML years, shipped evidence, penalties
│   ├── behavioral_features.py     ← 23 signals → 4 sub-scores → geometric combine
│   ├── scorer.py                  ← final formula + min-max normalize + rank
│   ├── reasoning.py               ← varied per-candidate reasoning (no LLM)
│   ├── csv_writer.py              ← validated CSV output
│   ├── io_utils.py                ← streaming JSONL reader 
│   └── validate_local.py          ← pre-submission validator
├── scripts/
│   ├── precompute.py              ← Phase 1: offline feature store builder
│   └── rank.py                    ← Phase 2: fast ranking from feature store
├── sandbox/
│   └── app.py                     ← Streamlit demo 
├── tests/
│   ├── test_cleaning.py
│   ├── test_trap_detector.py
│   ├── test_scorer.py
│   ├── test_csv_writer.py
│   ├── test_edge_cases.py
│   └── test_integration.py
├── artifacts/                     ← generated by precompute.py 
├── data/                          ← place candidates.jsonl here
├── submission/
│   └── team_redrob.csv            ← final submission
├── requirements.txt
├── submission_metadata.yaml       ← FILL IN before portal submission
└── validate_submission.py         ← official hackathon validator 
```

---

## Key Design Decisions

### Why two phases?
The submission spec requires ranking to complete in ≤5 minutes on CPU with no network. By precomputing all embeddings and features offline, the ranking step is pure pandas/numpy arithmetic — runs in <10 seconds even for 100K candidates.

### Why multiplicative gating?
A candidate who is outstanding on skills but unreachable (last active 11 months ago, notice period 150 days) should not rank in the top 10. Multiplicative scoring ensures all three gates (skill, career, behavioral) must be healthy for a high final score.

### Why β = 0.80 for behavioral_fit?
Many strong ML engineers work on private proprietary repos — their GitHub score is 0 (sentinel). A β < 1.0 prevents this from eliminating otherwise excellent candidates while still letting strong behavioral signals boost rankings.

### Why varied reasoning openers?
The Stage 4 evaluation explicitly checks for templated reasoning. The reasoning module selects the single strongest differentiator per candidate as the opening clause (ML years, shipped evidence, specific skill match, or title alignment), ensuring surface variation across all 100 rows.

### Why `all-MiniLM-L6-v2`?
Fast, CPU-friendly, good multilingual coverage, 384 dimensions. Produces high-quality semantic similarity for the skill/career matching use case. The compute constraints (no GPU, ≤16 GB) make this the right size.

---
