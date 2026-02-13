#!/usr/bin/env python
"""Evaluate DECODE-FISH predictions on C. elegans (no position GT).

Sweeps probability thresholds, plots count curves, spatial density,
and image overlays at selected thresholds.

Usage:
    python evaluate_celegans.py --pred_csv models/celegans_5k/predictions.csv
    python evaluate_celegans.py --pred_csv predictions.csv --image_path example/N2_702_highSNR.tif
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile


def load_predictions(csv_path):
    """Load predictions CSV."""
    df = pd.read_csv(csv_path)
    if 'frame_idx' not in df.columns and 'frame' in df.columns:
        df['frame_idx'] = df['frame']
    return df


def count_at_thresholds(pred_df, thresholds):
    """Count detections at each threshold, optionally per frame."""
    results = []
    frames = sorted(pred_df['frame_idx'].unique())
    for t in thresholds:
        filtered = pred_df[pred_df['prob'] >= t]
        row = {'threshold': t, 'total': len(filtered)}
        for f in frames:
            row[f'frame_{f}'] = len(filtered[filtered['frame_idx'] == f])
        results.append(row)
    return pd.DataFrame(results)


def plot_count_vs_threshold(count_df, output_dir, n_frames):
    """Plot detection count vs threshold."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(count_df['threshold'], count_df['total'], 'b.-', linewidth=2,
            markersize=5, label='Total')

    if n_frames > 1:
        for col in count_df.columns:
            if col.startswith('frame_'):
                ax.plot(count_df['threshold'], count_df[col], '--', linewidth=1,
                        alpha=0.5, markersize=3, label=col)

    ax.set_xlabel('Probability Threshold', fontsize=12)
    ax.set_ylabel('Detection Count', fontsize=12)
    ax.set_title('Detection Count vs Threshold', fontsize=14)
    ax.set_yscale('log')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / 'count_vs_threshold.png', dpi=150)
    plt.close(fig)


def plot_prob_histogram(pred_df, output_dir):
    """Plot histogram of prediction probabilities."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(pred_df['prob'], bins=50, color='steelblue', edgecolor='black', alpha=0.8)
    ax.set_xlabel('Probability', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Probability Distribution', fontsize=14)
    ax.axvline(x=0.5, color='red', linestyle='--', label='Default threshold (0.5)')
    ax.axvline(x=0.9, color='orange', linestyle='--', label='High threshold (0.9)')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / 'prob_histogram.png', dpi=150)
    plt.close(fig)


def plot_spatial_density(pred_df, output_dir, threshold=0.5):
    """Plot 2D spatial density of detections."""
    filtered = pred_df[pred_df['prob'] >= threshold]
    if len(filtered) == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    scatter = ax.scatter(filtered['x'] / 100.0, filtered['y'] / 100.0,
                         c=filtered['prob'], cmap='hot', s=5, alpha=0.7,
                         vmin=threshold, vmax=1.0)
    plt.colorbar(scatter, ax=ax, label='Probability')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_title(f'Spatial Density (threshold={threshold}, n={len(filtered)})')
    ax.set_aspect('equal')
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(output_dir / f'spatial_density_t{threshold:.2f}.png', dpi=150)
    plt.close(fig)


def plot_overlays(pred_df, image_path, output_dir, thresholds=(0.5, 0.7, 0.9, 0.95)):
    """Max-projection overlays at multiple thresholds."""
    img = tifffile.imread(str(image_path))

    # Handle multi-frame: use first frame for single-image, or iterate
    if img.ndim == 3:
        max_proj = img.max(axis=0)
        frame_idx = 0
    elif img.ndim == 2:
        max_proj = img
        frame_idx = 0
    else:
        # 4D: take first volume
        max_proj = img[0].max(axis=0) if img.ndim == 4 else img.max(axis=0)
        frame_idx = 0

    n_thresh = len(thresholds)
    fig, axes = plt.subplots(1, n_thresh, figsize=(5 * n_thresh, 5))
    if n_thresh == 1:
        axes = [axes]

    for ax, t in zip(axes, thresholds):
        ax.imshow(max_proj, cmap='gray',
                  vmin=np.percentile(max_proj, 1),
                  vmax=np.percentile(max_proj, 99.5))

        filtered = pred_df[(pred_df['prob'] >= t) & (pred_df['frame_idx'] == frame_idx)]
        if len(filtered) > 0:
            ax.scatter(filtered['x'] / 100.0, filtered['y'] / 100.0,
                       s=20, marker='x', color='red', linewidths=0.8, alpha=0.8)

        ax.set_title(f't={t:.2f} (n={len(filtered)})', fontsize=11)
        ax.axis('off')

    fig.suptitle('Detections at Different Thresholds', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / 'threshold_overlays.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_multi_image_overlays(pred_df, image_paths, output_dir, threshold=0.9):
    """Overlay predictions on multiple images at a single threshold."""
    n_imgs = min(len(image_paths), 6)
    fig, axes = plt.subplots(1, n_imgs, figsize=(5 * n_imgs, 5))
    if n_imgs == 1:
        axes = [axes]

    for idx, (ax, img_path) in enumerate(zip(axes, image_paths[:n_imgs])):
        img = tifffile.imread(str(img_path))
        if img.ndim == 3:
            max_proj = img.max(axis=0)
        else:
            max_proj = img

        ax.imshow(max_proj, cmap='gray',
                  vmin=np.percentile(max_proj, 1),
                  vmax=np.percentile(max_proj, 99.5))

        filtered = pred_df[(pred_df['prob'] >= threshold) & (pred_df['frame_idx'] == idx)]
        if len(filtered) > 0:
            ax.scatter(filtered['x'] / 100.0, filtered['y'] / 100.0,
                       s=30, marker='x', color='red', linewidths=1.0)

        ax.set_title(f'Frame {idx} (n={len(filtered)})', fontsize=11)
        ax.axis('off')

    fig.suptitle(f'Multi-Image Detections (threshold={threshold})', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / f'multi_image_overlay_t{threshold:.2f}.png', dpi=150,
                bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Evaluate DECODE-FISH predictions on C. elegans')
    parser.add_argument('--pred_csv', required=True, help='Path to predictions CSV')
    parser.add_argument('--image_path', default=None,
                        help='Path to image TIF or glob pattern for overlays')
    parser.add_argument('--output_dir', default=None,
                        help='Output directory (default: pred_csv dir + /eval)')
    args = parser.parse_args()

    pred_csv = Path(args.pred_csv)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = pred_csv.parent / 'eval'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Predictions: {pred_csv}")
    print(f"Output: {output_dir}")

    # Load predictions
    pred_df = load_predictions(pred_csv)
    n_frames = pred_df['frame_idx'].nunique()
    print(f"\nTotal predictions: {len(pred_df)}")
    print(f"Frames: {n_frames}")
    if len(pred_df) > 0:
        print(f"Prob range: [{pred_df['prob'].min():.4f}, {pred_df['prob'].max():.4f}]")
        print(f"\nPer-frame counts (all probs):")
        print(pred_df.groupby('frame_idx').size().to_string())

    # Threshold sweep
    thresholds = np.arange(0.05, 1.001, 0.025)
    count_df = count_at_thresholds(pred_df, thresholds)
    count_df.to_csv(output_dir / 'count_vs_threshold.csv', index=False)

    # Summary at key thresholds
    print(f"\n=== Counts at Key Thresholds ===")
    for t in [0.3, 0.5, 0.7, 0.9, 0.95, 0.99]:
        n = len(pred_df[pred_df['prob'] >= t])
        print(f"  t={t:.2f}: {n} detections")

    # Save summary
    summary = {
        'pred_csv': str(pred_csv),
        'n_total_pred': len(pred_df),
        'n_frames': n_frames,
    }
    for t in [0.3, 0.5, 0.7, 0.9, 0.95, 0.99]:
        summary[f'count_t{t:.2f}'] = int(len(pred_df[pred_df['prob'] >= t]))
    with open(output_dir / 'eval_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Plots
    print("\nGenerating plots...")
    plot_count_vs_threshold(count_df, output_dir, n_frames)
    plot_prob_histogram(pred_df, output_dir)
    plot_spatial_density(pred_df, output_dir, threshold=0.5)
    plot_spatial_density(pred_df, output_dir, threshold=0.9)

    # Image overlays
    if args.image_path:
        import glob
        image_paths = sorted(glob.glob(args.image_path))
        if len(image_paths) == 1:
            plot_overlays(pred_df, image_paths[0], output_dir)
        elif len(image_paths) > 1:
            # Single-image threshold sweep on first image
            plot_overlays(pred_df, image_paths[0], output_dir)
            # Multi-image overlay
            plot_multi_image_overlays(pred_df, image_paths, output_dir, threshold=0.9)

    print(f"\nResults saved to {output_dir}/")
    print("  count_vs_threshold.csv")
    print("  eval_summary.json")
    print("  count_vs_threshold.png")
    print("  prob_histogram.png")
    print("  spatial_density_t*.png")
    if args.image_path:
        print("  threshold_overlays.png")


if __name__ == '__main__':
    main()
