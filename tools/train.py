#!/usr/bin/env python3
"""Train the CKAN1 uniform-grid piecewise-linear KAN and compare identical C++ backends.

Reference training basis is triangular and valid only for a uniform grid.
Dataset is synthetic; no external data and no assertion of superiority over MLP.
"""
import argparse
import csv
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn


def target(x):
    return torch.sin(math.pi * x[:, 0:1]) + 0.4 * x[:, 1:2].square() - 0.2 * x[:, 0:1] * x[:, 1:2]


def dataset(seed, n):
    generator = torch.Generator().manual_seed(seed)
    x = (2.0 * torch.rand((n, 2), generator=generator) - 1.0).to(torch.float32)
    return x, target(x)


class PWLKANLayer(nn.Module):
    def __init__(self, inputs, outputs, knots=13, low=-2.0, high=2.0):
        super().__init__()
        grid = torch.linspace(low, high, knots)
        self.register_buffer('knots', grid)
        self.skip = nn.Parameter(torch.empty(outputs, inputs))
        self.coefficients = nn.Parameter(torch.empty(outputs, inputs, knots))
        self.bias = nn.Parameter(torch.zeros(outputs))
        nn.init.normal_(self.skip, std=0.15 / math.sqrt(inputs))
        nn.init.normal_(self.coefficients, std=0.15 / math.sqrt(inputs * knots))

    def forward(self, x):
        # Triangular basis for uniform knots. Spline argument clamps, skip does not.
        delta = (self.knots[1] - self.knots[0]).item()
        z = x.clamp(min=self.knots[0].item(), max=self.knots[-1].item())
        basis = (1.0 - (z.unsqueeze(-1) - self.knots).abs() / delta).clamp(min=0.0)
        spline = torch.einsum('bik,oik->bo', basis, self.coefficients)
        return spline + nn.functional.linear(x, self.skip, self.bias)


class PWLKAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([PWLKANLayer(2, 4), PWLKANLayer(4, 1)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def export_text(model, path):
    with torch.no_grad(), Path(path).open('w', encoding='ascii', newline='\n') as out:
        out.write(f'CKAN 1\nLAYERS {len(model.layers)}\n')
        for layer in model.layers:
            I = layer.skip.shape[1]
            O = layer.skip.shape[0]
            K = layer.knots.numel()
            out.write(f'LAYER {I} {O} {K}\n')
            for tag, data in [('KNOTS', layer.knots), ('BIAS', layer.bias),
                              ('SKIP', layer.skip), ('COEFF', layer.coefficients)]:
                vals = data.detach().cpu().contiguous().numpy().astype(np.float32).ravel()
                out.write(tag + ' ' + ' '.join(format(float(v), '.9g') for v in vals) + '\n')


def train_model(model, x, y, steps, lr):
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(steps):
        optim.zero_grad(set_to_none=True)
        loss = (model(x) - y).square().mean()
        loss.backward()
        optim.step()
    return model.eval()


def mse(model, x, y):
    with torch.no_grad():
        return float((model(x) - y).square().mean().item())


def parity(check_exe, model_path, x_test, reference, out_dir):
    model_input = out_dir / 'heldout_inputs.csv'
    with model_input.open('w', newline='') as f:
        writer = csv.writer(f)
        for v in x_test.detach().cpu().numpy():
            writer.writerow([format(float(q), '.9g') for q in v])
    result = {}
    for backend in ['baseline', 'shared', 'packed']:
        output_path = out_dir / f'heldout_{backend}.csv'
        command = [str(check_exe), '--model', str(model_path), '--batch', str(model_input),
                   '--output', str(output_path), '--engine', backend]
        subprocess.run(command, check=True, capture_output=True, text=True)
        output = np.loadtxt(output_path, delimiter=',', dtype=np.float32).reshape(-1, 1)
        if output.shape != reference.shape:
            raise RuntimeError(f'{backend} shape mismatch: {output.shape} vs {reference.shape}')
        result[backend] = {'max_abs_error': float(np.max(np.abs(output - reference))),
                           'mean_abs_error': float(np.mean(np.abs(output - reference)))}
        if result[backend]['max_abs_error'] > 3e-5:
            raise RuntimeError(f'{backend} violates reference numerical tolerance: {result[backend]}')
    for backend in ['shared', 'packed']:
        a = np.loadtxt(out_dir / 'heldout_baseline.csv', delimiter=',', dtype=np.float32)
        b = np.loadtxt(out_dir / f'heldout_{backend}.csv', delimiter=',', dtype=np.float32)
        if not np.array_equal(a, b):
            raise RuntimeError(f'C++ {backend} is not exact versus C++ baseline')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=650)
    parser.add_argument('--out', type=Path, default=Path('results/trained'))
    parser.add_argument('--infer', type=Path, default=Path('build/kan_infer'))
    parser.add_argument('--lr', type=float, default=0.016)
    args = parser.parse_args()
    if args.steps < 1 or args.steps > 100000:
        parser.error('steps must be in 1..100000')
    torch.set_num_threads(1)
    torch.manual_seed(11)
    np.random.seed(11)
    x_train, y_train = dataset(202, 1024)
    x_test, y_test = dataset(701, 256)
    args.out.mkdir(parents=True, exist_ok=True)
    kan = train_model(PWLKAN(), x_train, y_train, args.steps, args.lr)
    with torch.no_grad():
        torch_ref = kan(x_test).numpy().astype(np.float32)
    trained = args.out / 'trained.kan'
    export_text(kan, trained)
    # Equal *trainable* parameter count: 2->43->1 tanh MLP has 173 params.
    torch.manual_seed(11)
    mlp = nn.Sequential(nn.Linear(2, 43), nn.Tanh(), nn.Linear(43, 1))
    mlp = train_model(mlp, x_train, y_train, args.steps, args.lr)
    affine = nn.Linear(2, 1)
    with torch.no_grad():
        X = torch.cat([x_train, torch.ones_like(x_train[:, :1])], dim=1)
        coef = torch.linalg.lstsq(X, y_train).solution
        pred_affine = torch.cat([x_test, torch.ones_like(x_test[:, :1])], dim=1) @ coef
        affine_mse = (pred_affine - y_test).square().mean().item()
    n_trainable = sum(p.numel() for p in kan.parameters())
    mlp_trainable = sum(p.numel() for p in mlp.parameters())
    assert n_trainable == mlp_trainable == 173
    tests = parity(args.infer.resolve(), trained.resolve(), x_test, torch_ref, args.out)
    report = {
        'dataset': 'seed202 1024 uniform [-1,1]^2 training; seed701 256 held-out',
        'target': 'sin(pi*x0) + 0.4*x1^2 - 0.2*x0*x1',
        'architecture': 'piecewise-linear 2-4-1, 13 fixed uniform knots in [-2,2] per layer',
        'steps': args.steps, 'lr': args.lr, 'trainable_params': n_trainable,
        'mlp_trainable_params': mlp_trainable,
        'kan_heldout_mse': mse(kan, x_test, y_test),
        'mlp_heldout_mse': mse(mlp, x_test, y_test),
        'affine_heldout_mse': float(affine_mse),
        'torch_version': torch.__version__, 'numpy_version': np.__version__,
        'python_version': sys.version, 'host': platform.platform(),
        'cpp_parity': tests,
        'cpp_engines_bitwise_equal_on_256_samples': True,
        'caveat': 'Synthetic 2D function, not task-general model quality or hardware speedup.'
    }
    (args.out / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k in
                      ('steps', 'kan_heldout_mse', 'mlp_heldout_mse', 'affine_heldout_mse', 'cpp_parity')}, indent=2))


if __name__ == '__main__':
    main()
