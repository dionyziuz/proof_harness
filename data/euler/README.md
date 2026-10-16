# Project Euler Dataset

This directory contains computational number theory and algorithmic discrete mathematics problems from Project Euler.

## Files
- `euler_200.json`: 200 curated core benchmark problems (Project Euler 1–200) for standard solver evaluation and regression testing.
- `euler_800_1000.json`: High-difficulty research-level problems (Project Euler 800–1000) requiring advanced analytic number theory, generating functions, and sieve methods.
- `euler_full_1000.json`: Complete dataset of 1,000 Project Euler problems with problem titles, descriptions, and answer keys.

## Evaluation
Run evaluation via:
```bash
prime-math bench --dataset euler
```
Or with custom limits:
```bash
python3 -m prime_math.evaluator --limit 10 --workers 2
```
