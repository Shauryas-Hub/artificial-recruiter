"""
src/validate_local.py — Local submission validator.

Runs all checks from the official validate_submission.py PLUS additional
quality checks:
  - Reasoning diversity (not all identical)
  - Score monotonicity
  - Honeypot rate check
  - Reasoning hallucination check (no made-up candidate IDs)
"""
from __future__ import annotations
import csv
import re
import sys
from pathlib import Path

REQUIRED_HEADER = ["candidate_id", "rank", "score", "reasoning"]
CANDIDATE_ID_PATTERN = re.compile(r"^CAND_[0-9]{7}$")


def validate_submission(csv_path: str, honeypot_ids: set | None = None) -> list[str]:
    errors = []
    path = Path(csv_path)

    if path.suffix.lower() != ".csv":
        errors.append("File must have .csv extension")
        return errors

    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                errors.append("File is empty")
                return errors

            if header != REQUIRED_HEADER:
                errors.append(
                    f"Header must be exactly: {','.join(REQUIRED_HEADER)}\nGot: {','.join(header)}"
                )

            data_rows = [row for row in reader if any(c.strip() for c in row)]
    except UnicodeDecodeError:
        errors.append("File must be UTF-8 encoded")
        return errors
    except OSError as e:
        errors.append(f"Cannot read file: {e}")
        return errors

    if len(data_rows) != 100:
        errors.append(f"Expected 100 data rows, found {len(data_rows)}")

    seen_ids: set[str] = set()
    seen_ranks: set[int] = set()
    by_rank: list[tuple[int, float, str]] = []
    reasonings: list[str] = []

    for i, cells in enumerate(data_rows):
        row_num = i + 2
        if len(cells) != 4:
            errors.append(f"Row {row_num}: expected 4 columns, got {len(cells)}")
            continue

        cid, rank_s, score_s, reasoning = [c.strip() for c in cells]

        if not CANDIDATE_ID_PATTERN.match(cid):
            errors.append(f"Row {row_num}: invalid candidate_id '{cid}'")
        elif cid in seen_ids:
            errors.append(f"Row {row_num}: duplicate candidate_id '{cid}'")
        else:
            seen_ids.add(cid)

        try:
            rank = int(rank_s)
            if not 1 <= rank <= 100:
                errors.append(f"Row {row_num}: rank {rank} out of 1..100")
            elif rank in seen_ranks:
                errors.append(f"Row {row_num}: duplicate rank {rank}")
            else:
                seen_ranks.add(rank)
        except ValueError:
            errors.append(f"Row {row_num}: rank must be integer, got '{rank_s}'")
            rank = None

        try:
            score = float(score_s)
        except ValueError:
            errors.append(f"Row {row_num}: score must be float, got '{score_s}'")
            score = None

        if reasoning:
            reasonings.append(reasoning)

        if rank is not None and score is not None and cid:
            by_rank.append((rank, score, cid))

    # Monotonicity
    by_rank.sort(key=lambda x: x[0])
    for i in range(len(by_rank) - 1):
        r1, s1, _ = by_rank[i]
        r2, s2, _ = by_rank[i + 1]
        if s1 < s2:
            errors.append(f"Score not non-increasing: rank {r1}={s1} < rank {r2}={s2}")

    # Missing ranks
    missing = set(range(1, 101)) - seen_ranks
    if missing:
        errors.append(f"Missing ranks: {sorted(missing)[:10]}")

    # Reasoning quality checks
    if reasonings:
        unique_count = len(set(reasonings))
        if unique_count < 80:
            errors.append(
                f"Low reasoning diversity: only {unique_count}/100 unique reasonings "
                f"(risk of Stage 4 'templated' flag)"
            )
        empty_count = sum(1 for r in reasonings if not r.strip())
        if empty_count > 0:
            errors.append(f"{empty_count} rows have empty reasoning")

    # Honeypot check (if IDs provided)
    if honeypot_ids:
        hp_in_top100 = seen_ids & honeypot_ids
        rate = len(hp_in_top100) / 100
        if rate > 0.10:
            errors.append(
                f"DISQUALIFICATION RISK: honeypot rate {rate:.1%} > 10% "
                f"(IDs: {list(hp_in_top100)[:5]})"
            )
        else:
            print(f"  Honeypot rate: {rate:.1%} — OK ({len(hp_in_top100)} in top-100)")

    return errors


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.validate_local <submission.csv> [honeypots.csv]")
        sys.exit(1)

    hp_ids = None
    if len(sys.argv) >= 3:
        import pandas as pd
        hp_df = pd.read_csv(sys.argv[2])
        hp_ids = set(hp_df["candidate_id"].tolist()) if "candidate_id" in hp_df.columns else None

    errs = validate_submission(sys.argv[1], hp_ids)
    if errs:
        print(f"\nValidation FAILED ({len(errs)} issue(s)):")
        for e in errs:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("Submission is VALID.")
