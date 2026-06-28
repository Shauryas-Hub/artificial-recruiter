# data/

Place the candidate pool here before running precompute.py:

```bash
# Gzipped JSONL (preferred, ~52 MB):
cp /path/to/candidates.jsonl.gz data/

# Or plain JSONL (~465 MB):
cp /path/to/candidates.jsonl data/
```

Then run:
```bash
python scripts/precompute.py --candidates data/candidates.jsonl.gz
# or
python scripts/precompute.py --candidates data/candidates.jsonl
```

The sample candidates file for testing can also be placed here:
```bash
cp /path/to/sample_candidates.json data/
```
 