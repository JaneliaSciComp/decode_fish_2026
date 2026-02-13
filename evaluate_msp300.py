#!/usr/bin/env python
"""Evaluate DECODE-FISH predictions against msp300 ground truth.

Sweeps probability thresholds, computes precision/recall/F1/RMSE via matching(),
generates diagnostic plots, and saves results.

Usage:
    python evaluate_msp300.py --pred_csv models/msp300_5k/predictions.csv
    python evaluate_msp300.py --pred_csv predictions.csv --ref_csv example/msp300_predictions.csv
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
import tifffile

# Add decode_fish to path
sys.path.insert(0, str(Path(__file__).parent))
from decode_fish.funcs.matching import matching


def load_gt(gt_npy, pixel_size=100.0):
    """Load ground truth from .npy file (z,y,x in pixels) -> DataFrame in nm."""
    zyx = np.load(gt_npy)  # (N, 3) in pixels, columns = z, y, x
    gt_df = pd.DataFrame({
        'frame_idx': np.zeros(len(zyx), dtype=int),
        'x': zyx[:, 2] * pixel_size,  # x in nm
        'y': zyx[:, 1] * pixel_size,  # y in nm
        'z': zyx[:, 0] * pixel_size,  # z in nm
        'code_inds': np.zeros(len(zyx), dtype=int),
    })
    return gt_df


def load_predictions(csv_path, gt_frames=None):
    """Load predictions CSV, add code_inds if missing.

    If gt_frames is provided, filter to only keep predictions whose frame_idx
    is in gt_frames. This handles e.g. the reference CSV where frame 2 contains
    GT positions embedded as predictions.
    """
    df = pd.read_csv(csv_path)
    if 'code_inds' not in df.columns:
        df['code_inds'] = 0
    if 'frame_idx' not in df.columns and 'frame' in df.columns:
        df['frame_idx'] = df['frame']
    if gt_frames is not None:
        n_before = len(df)
        df = df[df['frame_idx'].isin(gt_frames)].reset_index(drop=True)
        if len(df) < n_before:
            print(f"  Filtered predictions: {n_before} -> {len(df)} (keeping frames {sorted(gt_frames)})")
    return df


def evaluate_at_threshold(gt_df, pred_df, threshold, tolerance=1000):
    """Filter predictions by threshold and run matching."""
    filtered = pred_df[pred_df['prob'] >= threshold].reset_index(drop=True)
    n_pred = len(filtered)

    if n_pred == 0:
        return {
            'threshold': threshold,
            'n_pred': 0,
            'n_gt': len(gt_df),
            'TP': 0, 'FP': 0, 'FN': len(gt_df),
            'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'jaccard': 0.0,
            'rmse_vol': np.nan, 'rmse_x': np.nan, 'rmse_y': np.nan, 'rmse_z': np.nan,
        }

    perf, match_df, shift = matching(
        gt_df, filtered, tolerance=tolerance, print_res=False, match_genes=False
    )

    precision = perf['precision']
    recall = perf['recall']
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'threshold': threshold,
        'n_pred': n_pred,
        'n_gt': len(gt_df),
        'TP': perf['n_matches'],
        'FP': n_pred - perf['n_matches'],
        'FN': len(gt_df) - perf['n_matches'],
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'jaccard': perf['jaccard'],
        'rmse_vol': perf['rmse_vol'],
        'rmse_x': perf['rmse_x'],
        'rmse_y': perf['rmse_y'],
        'rmse_z': perf['rmse_z'],
    }


def sweep_thresholds(gt_df, pred_df, tolerance=1000):
    """Sweep thresholds from 0.05 to 1.0."""
    thresholds = np.arange(0.05, 1.001, 0.025)
    results = []
    for t in thresholds:
        res = evaluate_at_threshold(gt_df, pred_df, t, tolerance)
        results.append(res)
    return pd.DataFrame(results)


def plot_pr_curve(results_df, output_dir, label='predictions', ref_results_df=None):
    """Plot precision-recall curve with F1 iso-lines."""
    fig, ax = plt.subplots(figsize=(8, 7))

    # F1 iso-lines
    for f1_val in [0.2, 0.4, 0.6, 0.8, 0.9]:
        r_vals = np.linspace(0.01, 1.0, 200)
        p_vals = f1_val * r_vals / (2 * r_vals - f1_val)
        valid = (p_vals > 0) & (p_vals <= 1)
        ax.plot(r_vals[valid], p_vals[valid], '--', color='gray', alpha=0.3, linewidth=0.8)
        # Label the iso-line
        idx = np.argmin(np.abs(r_vals - 0.95))
        if valid[idx]:
            ax.annotate(f'F1={f1_val}', (r_vals[idx], p_vals[idx]),
                       fontsize=7, color='gray', alpha=0.5)

    # Main PR curve
    ax.plot(results_df['recall'], results_df['precision'], 'b.-', linewidth=1.5,
            markersize=4, label=label, zorder=3)

    # Mark optimal F1 point
    best_idx = results_df['f1'].idxmax()
    best = results_df.iloc[best_idx]
    ax.plot(best['recall'], best['precision'], 'r*', markersize=15, zorder=5,
            label=f'Best F1={best["f1"]:.3f} (t={best["threshold"]:.3f})')

    # Reference if provided
    if ref_results_df is not None:
        ax.plot(ref_results_df['recall'], ref_results_df['precision'], 'g.-',
                linewidth=1.5, markersize=4, label='reference', alpha=0.7, zorder=2)
        ref_best_idx = ref_results_df['f1'].idxmax()
        ref_best = ref_results_df.iloc[ref_best_idx]
        ax.plot(ref_best['recall'], ref_best['precision'], 'g*', markersize=12, zorder=4,
                label=f'Ref best F1={ref_best["f1"]:.3f} (t={ref_best["threshold"]:.3f})')

    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve', fontsize=14)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower left', fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / 'pr_curve.png', dpi=150)
    plt.close(fig)


def plot_count_vs_threshold(results_df, output_dir, n_gt, ref_results_df=None):
    """Plot detection count vs threshold with GT line."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(results_df['threshold'], results_df['n_pred'], 'b.-', linewidth=1.5,
            markersize=4, label='predictions')
    ax.axhline(y=n_gt, color='red', linestyle='--', linewidth=1.5, label=f'GT count ({n_gt})')

    if ref_results_df is not None:
        ax.plot(ref_results_df['threshold'], ref_results_df['n_pred'], 'g.-',
                linewidth=1.5, markersize=4, label='reference', alpha=0.7)

    ax.set_xlabel('Probability Threshold', fontsize=12)
    ax.set_ylabel('Detection Count', fontsize=12)
    ax.set_title('Detection Count vs Threshold', fontsize=14)
    ax.set_yscale('log')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / 'count_vs_threshold.png', dpi=150)
    plt.close(fig)


def plot_rmse_vs_threshold(results_df, output_dir, ref_results_df=None):
    """Plot RMSE (total + per-axis) vs threshold."""
    fig, ax = plt.subplots(figsize=(8, 5))

    valid = results_df['rmse_vol'].notna()
    ax.plot(results_df.loc[valid, 'threshold'], results_df.loc[valid, 'rmse_vol'],
            'b.-', linewidth=1.5, label='RMSE_vol')
    ax.plot(results_df.loc[valid, 'threshold'], results_df.loc[valid, 'rmse_x'],
            'r.--', linewidth=1, alpha=0.7, label='RMSE_x')
    ax.plot(results_df.loc[valid, 'threshold'], results_df.loc[valid, 'rmse_y'],
            'g.--', linewidth=1, alpha=0.7, label='RMSE_y')
    ax.plot(results_df.loc[valid, 'threshold'], results_df.loc[valid, 'rmse_z'],
            'm.--', linewidth=1, alpha=0.7, label='RMSE_z')

    if ref_results_df is not None:
        ref_valid = ref_results_df['rmse_vol'].notna()
        ax.plot(ref_results_df.loc[ref_valid, 'threshold'],
                ref_results_df.loc[ref_valid, 'rmse_vol'],
                'c.-', linewidth=1.5, alpha=0.6, label='Ref RMSE_vol')

    ax.set_xlabel('Probability Threshold', fontsize=12)
    ax.set_ylabel('RMSE (nm)', fontsize=12)
    ax.set_title('Localization RMSE vs Threshold', fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / 'rmse_vs_threshold.png', dpi=150)
    plt.close(fig)


def plot_xy_overlay(gt_df, pred_df, optimal_threshold, output_dir, image_path=None):
    """Max-Z projection with GT circles and prediction crosses at optimal threshold."""
    filtered = pred_df[pred_df['prob'] >= optimal_threshold].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 10))

    # Background image (max projection) if available
    if image_path is not None and Path(image_path).exists():
        img = tifffile.imread(str(image_path))
        if img.ndim == 3:
            max_proj = img.max(axis=0)
        else:
            max_proj = img
        ax.imshow(max_proj, cmap='gray', vmin=np.percentile(max_proj, 1),
                  vmax=np.percentile(max_proj, 99.5))

    # GT positions (convert nm back to pixels for overlay, px_size=100)
    gt_x_px = gt_df['x'] / 100.0
    gt_y_px = gt_df['y'] / 100.0
    ax.scatter(gt_x_px, gt_y_px, s=80, facecolors='none', edgecolors='lime',
               linewidths=1.2, label=f'GT ({len(gt_df)})', zorder=3)

    # Predictions
    pred_x_px = filtered['x'] / 100.0
    pred_y_px = filtered['y'] / 100.0
    ax.scatter(pred_x_px, pred_y_px, s=30, marker='x', color='red',
               linewidths=1.0, label=f'Pred ({len(filtered)}, t={optimal_threshold:.3f})',
               zorder=4)

    ax.set_title(f'XY Overlay (threshold={optimal_threshold:.3f})', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    fig.tight_layout()
    fig.savefig(output_dir / 'xy_overlay.png', dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Evaluate DECODE-FISH predictions on msp300')
    parser.add_argument('--pred_csv', required=True, help='Path to predictions CSV')
    parser.add_argument('--gt_npy', default=None, help='Path to GT .npy (default: example/msp300_zyx.npy)')
    parser.add_argument('--output_dir', default=None, help='Output directory (default: same dir as pred_csv + /eval)')
    parser.add_argument('--ref_csv', default=None, help='Path to reference predictions CSV')
    parser.add_argument('--image_path', default=None, help='Path to image TIF for overlay')
    parser.add_argument('--tolerance', type=float, default=1000, help='Matching tolerance in nm')
    parser.add_argument('--pixel_size', type=float, default=100.0, help='Pixel size in nm')
    args = parser.parse_args()

    base_dir = Path(__file__).parent

    # Resolve defaults
    gt_npy = Path(args.gt_npy) if args.gt_npy else base_dir / 'example' / 'msp300_zyx.npy'
    pred_csv = Path(args.pred_csv)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = pred_csv.parent / 'eval'
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.image_path:
        image_path = Path(args.image_path)
    else:
        image_path = base_dir / 'example' / 'msp300_smFISH_3.tif'

    print(f"GT: {gt_npy}")
    print(f"Predictions: {pred_csv}")
    print(f"Output: {output_dir}")
    print(f"Tolerance: {args.tolerance} nm")

    # Load data
    gt_df = load_gt(gt_npy, pixel_size=args.pixel_size)
    gt_frames = set(gt_df['frame_idx'].unique())
    pred_df = load_predictions(pred_csv, gt_frames=gt_frames)
    n_gt = len(gt_df)
    print(f"\nGT emitters: {n_gt}")
    print(f"Total predictions: {len(pred_df)}")
    if len(pred_df) > 0:
        print(f"Prob range: [{pred_df['prob'].min():.4f}, {pred_df['prob'].max():.4f}]")

    # Sweep thresholds
    print("\nSweeping thresholds...")
    results_df = sweep_thresholds(gt_df, pred_df, tolerance=args.tolerance)

    # Find best F1
    best_idx = results_df['f1'].idxmax()
    best = results_df.iloc[best_idx]
    print(f"\n=== Best F1 ===")
    print(f"  Threshold: {best['threshold']:.3f}")
    print(f"  F1: {best['f1']:.4f}")
    print(f"  Precision: {best['precision']:.4f}")
    print(f"  Recall: {best['recall']:.4f}")
    print(f"  Jaccard: {best['jaccard']:.4f}")
    print(f"  RMSE_vol: {best['rmse_vol']:.1f} nm")
    print(f"  Detections: {int(best['n_pred'])} (GT: {n_gt})")

    # Reference evaluation
    ref_results_df = None
    if args.ref_csv:
        ref_csv = Path(args.ref_csv)
        if ref_csv.exists():
            print(f"\nEvaluating reference: {ref_csv}")
            ref_df = load_predictions(ref_csv, gt_frames=gt_frames)
            print(f"Reference predictions: {len(ref_df)}")
            ref_results_df = sweep_thresholds(gt_df, ref_df, tolerance=args.tolerance)
            ref_best_idx = ref_results_df['f1'].idxmax()
            ref_best = ref_results_df.iloc[ref_best_idx]
            print(f"  Ref best F1: {ref_best['f1']:.4f} at t={ref_best['threshold']:.3f}")
            ref_results_df.to_csv(output_dir / 'ref_eval_results.csv', index=False)

    # Save results
    results_df.to_csv(output_dir / 'eval_results.csv', index=False)

    summary = {
        'pred_csv': str(pred_csv),
        'gt_npy': str(gt_npy),
        'n_gt': n_gt,
        'n_total_pred': len(pred_df),
        'tolerance': args.tolerance,
        'best_threshold': float(best['threshold']),
        'best_f1': float(best['f1']),
        'best_precision': float(best['precision']),
        'best_recall': float(best['recall']),
        'best_jaccard': float(best['jaccard']),
        'best_rmse_vol': float(best['rmse_vol']) if not np.isnan(best['rmse_vol']) else None,
        'best_rmse_x': float(best['rmse_x']) if not np.isnan(best['rmse_x']) else None,
        'best_rmse_y': float(best['rmse_y']) if not np.isnan(best['rmse_y']) else None,
        'best_rmse_z': float(best['rmse_z']) if not np.isnan(best['rmse_z']) else None,
        'best_n_pred': int(best['n_pred']),
    }
    with open(output_dir / 'eval_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Plots
    print("\nGenerating plots...")
    plot_pr_curve(results_df, output_dir, label='predictions', ref_results_df=ref_results_df)
    plot_count_vs_threshold(results_df, output_dir, n_gt, ref_results_df=ref_results_df)
    plot_rmse_vs_threshold(results_df, output_dir, ref_results_df=ref_results_df)
    plot_xy_overlay(gt_df, pred_df, best['threshold'], output_dir, image_path=image_path)

    print(f"\nResults saved to {output_dir}/")
    print("  eval_results.csv  — full threshold sweep")
    print("  eval_summary.json — best metrics")
    print("  pr_curve.png      — precision-recall curve")
    print("  count_vs_threshold.png")
    print("  rmse_vs_threshold.png")
    print("  xy_overlay.png")


if __name__ == '__main__':
    main()
