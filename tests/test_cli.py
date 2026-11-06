#!/usr/bin/env python3
"""Subprocess-level CKAN1 CSV and CLI contract tests (no third-party packages)."""
import argparse
import subprocess
import tempfile
from pathlib import Path


def execute(program, args, success=True):
    p = subprocess.run([str(program), *map(str, args)], text=True,
                       capture_output=True, check=False)
    if (p.returncode == 0) != success:
        raise AssertionError(f'exit={p.returncode}, stderr={p.stderr}, args={args}')
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--binary', required=True, type=Path)
    ap.add_argument('--model', required=True, type=Path)
    opts = ap.parse_args()
    program = opts.binary.resolve()
    model = opts.model.resolve()
    first = execute(program, ['--model', model, '--input', '0.4,-0.2', '--engine', 'packed'])
    assert float(first.stdout.strip()) == 1.0
    with tempfile.TemporaryDirectory() as directory:
        data = Path(directory) / 'x.csv'
        data.write_bytes(b'0.4,-0.2\r\n-0.4,0.2\r\n')
        results = []
        for engine in ('baseline', 'shared', 'packed'):
            target = Path(directory) / f'{engine}.csv'
            execute(program, ['--model', model, '--batch', data,
                              '--output', target, '--engine', engine, '--repeat', '3'])
            values = target.read_text().strip().splitlines()
            assert len(values) == 2
            assert float(values[0]) == 1.0
            results.append(values)
        assert results[0] == results[1] == results[2]
        data.write_text('0.4,-0.2,9\n')
        execute(program, ['--model', model, '--batch', data], success=False)
        data.write_text('nan,1\n')
        execute(program, ['--model', model, '--batch', data], success=False)
        data.write_text('0.4,-0.2\n')
        execute(program, ['--model', model, '--batch', data, '--input', '0,0'], success=False)
        execute(program, ['--model', 'missing-no-such-model.kan', '--input', '0,0'], success=False)
    print('CLI CSV/CRLF, repeat, three-way parity, invalid input: PASS')


if __name__ == '__main__':
    main()
