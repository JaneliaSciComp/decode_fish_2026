#!/usr/bin/env python
"""Evaluate DECODE-FISH predictions against public benchmark GT.

Supports GT formats:
  - fish_spots: CSV with axis-0/1/2 columns (Z, Y, X in pixels)
  - rsfish: .loc file (space-delimited X, Y, Z, intensity in pixels)

Wraps evaluate_msp300.py logic with GT format conversion.

Usage:
    python evaluate_benchmark.py --pred_csv predictions.csv --gt_csv gt.csv --gt_format fish_spots
    python evaluate_benchmark.py --pred_csv predictions.csv --gt_loc gt.loc --gt_format rsfish
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from decode_fish.funcs.matching import matching


def load_gt_fish_spots(csv_path, pixel_size=100.0):
    """Load FISH_spots GT (axis-0=Z, axis-1=Y, axis-2=X in pixels)."""
    df = pd.read_csv(csv_path)
    gt_df = pd.DataFrame({
        'frame_idx': 0,
        'x': df['axis-2'].values * pixel_size,
        'y': df['axis-1'].values * pixel_size,
        'z': df['axis-0'].values * pixel_size,
        'code_inds': 0,
    })
    return gt_df


def load_gt_rsfish(loc_path, pixel_size=100.0):
    """Load RS-FISH .loc GT (space-delimited: X, Y, Z, intensity in pixels)."""
    data = np.loadtxt(str(loc_path))
    gt_df = pd.DataFrame({
        'frame_idx': 0,
        'x': data[:, 0] * pixel_size,
        'y': data[:, 1] * pixel_size,
        'z': data[:, 2] * pixel_size,
        'code_inds': 0,
    })
    return gt_df


def load_predictions(csv_path):
    df = pd.read_csv(csv_path)
    if 'code_inds' not in df.columns:
        df['code_inds'] = 0
    if 'frame_idx' not in df.columns and 'frame' in df.columns:
        df['frame_idx'] = df['frame']
    # Filter to frame 0 only
    df = df[df['frame_idx'] == 0].reset_index(drop=True)
    return df


def evaluate_at_threshold(gt_df, pred_df, threshold, tolerance=1000):
    filtered = pred_df[pred_df['prob'] >= threshold].reset_index(drop=True)
    if len(filtered) == 0:
        return {
            'threshold': threshold, 'n_pred': 0, 'n_gt': len(gt_df),
            'TP': 0, 'FP': 0, 'FN': len(gt_df),
            'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'jaccard': 0.0,
            'rmse_vol': np.nan, 'rmse_x': np.nan, 'rmse_y': np.nan, 'rmse_z': np.nan,
        }
    perf, match_df, shift = matching(
        gt_df, filtered, tolerance=tolerance, print_res=False, match_genes=False
    )
    p, r = perf['precision'], perf['recall']
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {
        'threshold': threshold, 'n_pred': len(filtered), 'n_gt': len(gt_df),
        'TP': perf['n_matches'], 'FP': len(filtered) - perf['n_matches'],
        'FN': len(gt_df) - perf['n_matches'],
        'precision': p, 'recall': r, 'f1': f1, 'jaccard': perf['jaccard'],
        'rmse_vol': perf['rmse_vol'], 'rmse_x': perf['rmse_x'],
        'rmse_y': perf['rmse_y'], 'rmse_z': perf['rmse_z'],
    }


def sweep_thresholds(gt_df, pred_df, tolerance=1000):
    thresholds = np.arange(0.05, 1.001, 0.025)
    return pd.DataFrame([evaluate_at_threshold(gt_df, pred_df, t, tolerance) for t in thresholds])


def plot_results(results_df, output_dir, name, n_gt):
    # PR curve
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.plot(results_df['recall'], results_df['precision'], 'b.-', linewidth=1.5, markersize=4)
    best_idx = results_df['f1'].idxmax()
    best = results_df.iloc[best_idx]
    ax.plot(best['recall'], best['precision'], 'r*', markersize=15,
            label=f'Best F1={best["f1"]:.3f} (t={best["threshold"]:.3f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'{name}: Precision-Recall')
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(results_df['threshold'], results_df['n_pred'], 'b.-', linewidth=1.5)
    ax.axhline(y=n_gt, color='red', linestyle='--', label=f'GT ({n_gt})')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Count')
    ax.set_title(f'{name}: Detection Count')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    valid = results_df['rmse_vol'].notna()
    ax.plot(results_df.loc[valid, 'threshold'], results_df.loc[valid, 'rmse_vol'], 'b.-', linewidth=1.5, label='RMSE_vol')
    ax.plot(results_df.loc[valid, 'threshold'], results_df.loc[valid, 'rmse_x'], 'r.--', linewidth=1, alpha=0.7, label='x')
    ax.plot(results_df.loc[valid, 'threshold'], results_df.loc[valid, 'rmse_y'], 'g.--', linewidth=1, alpha=0.7, label='y')
    ax.plot(results_df.loc[valid, 'threshold'], results_df.loc[valid, 'rmse_z'], 'm.--', linewidth=1, alpha=0.7, label='z')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('RMSE (nm)')
    ax.set_title(f'{name}: Localization RMSE')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / f'{name}_results.png', dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Evaluate on public benchmarks')
    parser.add_argument('--pred_csv', required=True)
    parser.add_argument('--gt_csv', default=None, help='FISH_spots CSV path')
    parser.add_argument('--gt_loc', default=None, help='RS-FISH .loc path')
    parser.add_argument('--gt_format', required=True, choices=['fish_spots', 'rsfish'])
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--name', default='benchmark')
    parser.add_argument('--tolerance', type=float, default=1000)
    parser.add_argument('--pixel_size', type=float, default=100.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load GT
    if args.gt_format == 'fish_spots':
        gt_df = load_gt_fish_spots(args.gt_csv, args.pixel_size)
    elif args.gt_format == 'rsfish':
        gt_df = load_gt_rsfish(args.gt_loc, args.pixel_size)

    pred_df = load_predictions(args.pred_csv)
    n_gt = len(gt_df)

    print(f"=== {args.name} ===")
    print(f"GT: {n_gt} emitters")
    print(f"Predictions: {len(pred_df)}")
    if len(pred_df) > 0:
        print(f"Prob range: [{pred_df['prob'].min():.4f}, {pred_df['prob'].max():.4f}]")

    # Sweep
    results_df = sweep_thresholds(gt_df, pred_df, args.tolerance)
    best_idx = results_df['f1'].idxmax()
    best = results_df.iloc[best_idx]

    print(f"\nBest F1={best['f1']:.4f} at t={best['threshold']:.3f}")
    print(f"  P={best['precision']:.4f}  R={best['recall']:.4f}  RMSE={best['rmse_vol']:.1f}nm")
    print(f"  Detections: {int(best['n_pred'])} (GT: {n_gt})")

    # Save
    results_df.to_csv(output_dir / 'eval_results.csv', index=False)
    summary = {
        'name': args.name,
        'n_gt': n_gt,
        'n_total_pred': len(pred_df),
        'tolerance': args.tolerance,
        'best_threshold': float(best['threshold']),
        'best_f1': float(best['f1']),
        'best_precision': float(best['precision']),
        'best_recall': float(best['recall']),
        'best_jaccard': float(best['jaccard']),
        'best_rmse_vol': float(best['rmse_vol']) if not np.isnan(best['rmse_vol']) else None,
        'best_n_pred': int(best['n_pred']),
    }
    with open(output_dir / 'eval_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    plot_results(results_df, output_dir, args.name, n_gt)
    print(f"\nSaved to {output_dir}/")


if __name__ == '__main__':
    main()
