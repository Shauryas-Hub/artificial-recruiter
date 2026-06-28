"""
src/io_utils.py — I/O utilities for streaming JSONL (gzipped or plain).
"""
from __future__ import annotations
import gzip
import json
from pathlib import Path
from typing import Iterator


def stream_candidates(path: str) -> Iterator[dict]:
    """
    Stream raw candidate dicts from a .jsonl or .jsonl.gz file.
    Memory-efficient: one record at a time.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Candidate file not found: {path}")

    if p.suffix == ".gz":
        opener = lambda: gzip.open(str(p), "rt", encoding="utf-8")
    else:
        opener = lambda: open(str(p), "r", encoding="utf-8")

    with opener() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    # Skip malformed lines but don't crash
                    print(f"  WARNING: skipping malformed JSONL line: {e}")
