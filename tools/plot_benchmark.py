#!/usr/bin/env python3
"""Generate plot exclusively from recorded benchmark CSV (no simulated points)."""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='results/bench_host.csv')
    ap.add_argument('--output', default='results/bench_host.png')
    args = ap.parse_args()
    with open(args.input, newline='') as file:
        rows = list(csv.DictReader(file))
    shapes = sorted({(int(r['inputs']), int(r['knots'])) for r in rows})
    engines = ['baseline', 'shared', 'packed']
    data = {(int(r['inputs']), int(r['knots']), r['engine']):
            float(r['median_ns_per_sample']) for r in rows}
    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    for e in engines:
        vals = [data.get((w, k, e), float('nan')) for w, k in shapes]
        ax.plot(range(len(shapes)), vals, marker='o', label=e)
    ax.set_xticks(range(len(shapes)))
    ax.set_xticklabels([f'{w}x{w}\nK={k}' for w, k in shapes])
    ax.set_ylabel('Median host inference ns/sample (lower is better)')
    ax.set_xlabel('One-layer width and number of shared knots')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    plt.close(fig)
    print(args.output)


if __name__ == '__main__':
    main()
