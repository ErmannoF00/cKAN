#!/usr/bin/env python3
"""Mechanical source-of-truth benchmark summary from CSV."""
import argparse
import csv
import json
import platform
import subprocess
from pathlib import Path


def command(args):
    try:
        return subprocess.run(args, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return 'unavailable'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='results/bench_host.csv')
    ap.add_argument('--output', default='results/bench_summary.json')
    args = ap.parse_args()
    with open(args.input, newline='') as f:
        rows = list(csv.DictReader(f))
    sets = {}
    for r in rows:
        key = (r['geometry'], r['knots'], r['batch'])
        sets.setdefault(key, {})[r['engine']] = r
    entries = []
    for (geometry, knots, batch), engines in sorted(sets.items()):
        if len(engines) != 3:
            raise ValueError(f'missing engine for {geometry}, K={knots}, batch={batch}')
        baseline = float(engines['baseline']['median_ns_per_sample'])
        shared = float(engines['shared']['median_ns_per_sample'])
        packed = float(engines['packed']['median_ns_per_sample'])
        entries.append(dict(geometry=geometry, knots=int(knots), batch=int(batch),
                            baseline_ns=baseline, shared_ns=shared, packed_ns=packed,
                            shared_speedup=baseline/shared, packed_speedup=baseline/packed,
                            extra_packed_bytes=int(engines['packed']['packing_bytes']),
                            median_prep_baseline_us=float(engines['baseline']['prepare_us']),
                            median_prep_packed_us=float(engines['packed']['prepare_us'])))
    report = {
        'source_csv': args.input,
        'os': platform.platform(),
        'machine': platform.machine(),
        'processor': command(['bash', '-c', 'lscpu | grep -m1 "Model name" || true']),
        'g++': command(['g++', '--version']).splitlines()[0],
        'cmake': command(['cmake', '--version']).splitlines()[0],
        'protocol': 'one thread, C++17 release, 2 warmup batches; medians of independent timed blocks',
        'caveat': 'Ephemeral hosted CPU results; not transferably predictive of memory energy or FPGA execution.',
        'rows': entries,
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    print(f'Wrote {args.output} with {len(entries)} matched sets')


if __name__ == '__main__':
    main()
