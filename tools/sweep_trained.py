#!/usr/bin/env python3
"""Benchmark the exported two-layer reference at three sequential-host batch sizes."""
import csv
import subprocess
import tempfile
from pathlib import Path

MODEL = Path('results/trained/trained.kan')
OUT = Path('results/bench_trained.csv')


def main():
    if not MODEL.is_file():
        raise SystemExit('Run make check before the trained-model benchmark')
    records = []
    with tempfile.TemporaryDirectory() as directory:
        for batch in (1, 16, 256):
            local = Path(directory) / f'bench_{batch}.csv'
            subprocess.run(['./build/kan_bench', '--model', str(MODEL), '--batch', str(batch),
                            '--rounds', '9', '--reps', '12', '--output', str(local)], check=True)
            with local.open(newline='') as f:
                records.extend(list(csv.DictReader(f)))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print('Wrote', OUT)


if __name__ == '__main__':
    main()
