#!/usr/bin/env python
"""Estimate noise theta from image for DECODE-FISH Gamma noise model.

DECODE-FISH theta controls the Gamma noise distribution variance: Var = mean * theta.
The effective theta must account for BOTH pixel-level noise AND spatial background
variation within the random crop window (default 48x48). We provide two methods:

  1. **Residual method** (recommended): Subtract locally-smoothed background, then
     compute theta = Var(residuals) / mean(image) per Z-slice. The smoothing window
     should roughly match the training crop size. This captures the total variance
     the noise model needs to explain.

  2. **Percentile method**: Select lowest P-th percentile as background, compute
     theta = Var(bg) / mean(bg). This gives the pixel-level noise theta, which is
     typically LOWER than what DECODE-FISH needs (it underestimates because it
     excludes spatial variation).

Supports 3D (Z,H,W) and 4D (Z,C,H,W) TIF images.

Usage:
    python estimate_noise_params.py image.tif
    python estimate_noise_params.py image.tif --crop-size 48  # match training crop
    python estimate_noise_params.py image.tif --sensitivity
    python estimate_noise_params.py image.tif --output params.yaml --hydra
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import uniform_filter


def load_image(path):
    """Load TIF image."""
    return tifffile.imread(str(path))


def extract_channel(img, channel):
    """Extract a single channel from a multi-channel image."""
    if img.ndim == 3:
        return img
    if img.ndim == 4:
        if img.shape[1] <= 10 and img.shape[1] < img.shape[0]:
            print(f"  Detected shape (Z={img.shape[0]}, C={img.shape[1]}, H={img.shape[2]}, W={img.shape[3]})")
            return img[:, channel, :, :]
        elif img.shape[0] <= 10 and img.shape[0] < img.shape[1]:
            print(f"  Detected shape (C={img.shape[0]}, Z={img.shape[1]}, H={img.shape[2]}, W={img.shape[3]})")
            return img[channel, :, :, :]
        else:
            print(f"  Ambiguous 4D shape {img.shape}, treating axis 1 as channels")
            return img[:, channel, :, :]
    raise ValueError(f"Unsupported image dimensions: {img.ndim}D {img.shape}")


def estimate_theta_residual_per_slice(volume, smooth_window=10):
    """Estimate theta per Z-slice using residual method.

    For each slice: smooth with uniform filter, compute residuals,
    then theta = Var(residuals) / mean(slice).
    """
    vol = volume.astype(np.float64)
    thetas = []
    for z in range(vol.shape[0]):
        sl = vol[z]
        smoothed = uniform_filter(sl, size=smooth_window)
        residuals = sl - smoothed
        mean_sl = sl.mean()
        if mean_sl < 1e-6:
            thetas.append(np.nan)
            continue
        theta = residuals.var() / mean_sl
        thetas.append(theta)
    return np.array(thetas)


def estimate_theta_percentile_per_slice(volume, percentile=25):
    """Estimate theta per Z-slice using percentile background selection."""
    vol = volume.astype(np.float64)
    thetas = []
    for z in range(vol.shape[0]):
        vals = vol[z].ravel()
        thresh = np.percentile(vals, percentile)
        bg = vals[vals <= thresh]
        if len(bg) < 10 or bg.mean() < 1e-6:
            thetas.append(np.nan)
            continue
        thetas.append(bg.var() / bg.mean())
    return np.array(thetas)


def summarize_per_slice(thetas):
    """Compute summary stats from per-slice theta array."""
    valid = thetas[~np.isnan(thetas)]
    if len(valid) == 0:
        return {'median': np.nan, 'mean': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan}
    return {
        'median': np.median(valid),
        'mean': np.mean(valid),
        'std': np.std(valid),
        'min': np.min(valid),
        'max': np.max(valid),
    }


def main():
    parser = argparse.ArgumentParser(
        description='Estimate noise theta for DECODE-FISH',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('image', help='Path to TIF image')
    parser.add_argument('--crop-size', type=int, default=48,
                        help='Training crop size (default: 48, used for smoothing window)')
    parser.add_argument('--channel', type=int, default=0,
                        help='Channel index for multi-channel images (default: 0)')
    parser.add_argument('--sensitivity', action='store_true',
                        help='Sweep smoothing windows 5-80 and percentiles 10-40')
    parser.add_argument('--output', type=str, default=None,
                        help='Write results to YAML file')
    parser.add_argument('--hydra', action='store_true',
                        help='Print Hydra CLI override string')
    args = parser.parse_args()

    path = Path(args.image)
    if not path.exists():
        print(f"Error: {path} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Loading: {path.name}")
    img = load_image(path)
    print(f"  Raw shape: {img.shape}, dtype: {img.dtype}")

    vol = extract_channel(img, args.channel)
    print(f"  Volume: {vol.shape} (channel {args.channel})")
    print(f"  Intensity: [{vol.min()}, {vol.max()}], mean={vol.mean():.1f}")

    # --- Method 1: Residual (recommended) ---
    # Use smoothing window = crop_size / 4 (captures variation at crop scale)
    smooth_win = max(3, args.crop_size // 4)
    thetas_res = estimate_theta_residual_per_slice(vol, smooth_window=smooth_win)
    stats_res = summarize_per_slice(thetas_res)

    print(f"\n--- Residual method (smooth_window={smooth_win}, from crop_size={args.crop_size}) ---")
    print(f"  Per-slice median: {stats_res['median']:.1f}")
    print(f"  Per-slice mean:   {stats_res['mean']:.1f} +/- {stats_res['std']:.1f}")
    print(f"  Per-slice range:  [{stats_res['min']:.1f}, {stats_res['max']:.1f}]")

    theta_recommended = stats_res['median']

    # --- Method 2: Percentile (for comparison) ---
    thetas_pct = estimate_theta_percentile_per_slice(vol, percentile=25)
    stats_pct = summarize_per_slice(thetas_pct)

    print(f"\n--- Percentile method (25th percentile, pixel-level noise only) ---")
    print(f"  Per-slice median: {stats_pct['median']:.1f}")
    print(f"  Per-slice range:  [{stats_pct['min']:.1f}, {stats_pct['max']:.1f}]")

    # --- Method 3: All-pixel per-slice var/mean ---
    vol64 = vol.astype(np.float64)
    thetas_all = []
    for z in range(vol64.shape[0]):
        sl = vol64[z].ravel()
        thetas_all.append(sl.var() / sl.mean() if sl.mean() > 0 else np.nan)
    thetas_all = np.array(thetas_all)
    stats_all = summarize_per_slice(thetas_all)

    print(f"\n--- All-pixel method (total variation including spatial structure) ---")
    print(f"  Per-slice median: {stats_all['median']:.1f}")
    print(f"  Per-slice range:  [{stats_all['min']:.1f}, {stats_all['max']:.1f}]")

    # --- Recommendation ---
    print(f"\n{'='*50}")
    print(f"  RECOMMENDED theta = {theta_recommended:.1f}")
    print(f"  (residual method, captures noise + local background variation)")
    print(f"{'='*50}")

    if args.hydra:
        print(f"\n  Hydra override: genm.noise.theta={theta_recommended:.1f}")

    # --- Sensitivity sweep ---
    if args.sensitivity:
        print(f"\n--- Sensitivity: smoothing window ---")
        print(f"  {'Window':>7s}  {'Median':>8s}  {'Mean':>8s}  {'Std':>8s}  {'Min':>8s}  {'Max':>8s}")
        sweep_medians = []
        for win in [3, 5, 8, 12, 16, 24, 32, 48]:
            th = estimate_theta_residual_per_slice(vol, smooth_window=win)
            s = summarize_per_slice(th)
            sweep_medians.append(s['median'])
            print(f"  {win:7d}  {s['median']:8.1f}  {s['mean']:8.1f}  {s['std']:8.1f}  {s['min']:8.1f}  {s['max']:8.1f}")

        spread = max(sweep_medians) - min(sweep_medians)
        mean_sw = np.mean(sweep_medians)
        cv = spread / mean_sw if mean_sw > 0 else float('inf')
        if cv < 0.3:
            print(f"  Stability: GOOD (CV={cv:.1%})")
        elif cv < 0.6:
            print(f"  Stability: OK (CV={cv:.1%})")
        else:
            print(f"  Stability: VARIABLE (CV={cv:.1%}) - theta sensitive to window size")

        print(f"\n--- Sensitivity: percentile ---")
        print(f"  {'Pctl':>7s}  {'Median':>8s}  {'Mean':>8s}  {'Std':>8s}")
        for p in [10, 15, 20, 25, 30, 35, 40]:
            th = estimate_theta_percentile_per_slice(vol, percentile=p)
            s = summarize_per_slice(th)
            print(f"  {p:7d}  {s['median']:8.1f}  {s['mean']:8.1f}  {s['std']:8.1f}")

    # --- Output YAML ---
    if args.output:
        out_path = Path(args.output)
        with open(out_path, 'w') as f:
            f.write(f"# Noise parameters estimated from {path.name}\n")
            f.write(f"# Method: residual (smooth_window={smooth_win})\n")
            f.write(f"theta: {theta_recommended:.1f}\n")
            f.write(f"theta_residual_median: {stats_res['median']:.2f}\n")
            f.write(f"theta_residual_std: {stats_res['std']:.2f}\n")
            f.write(f"theta_percentile_median: {stats_pct['median']:.2f}\n")
            f.write(f"theta_allpixel_median: {stats_all['median']:.2f}\n")
        print(f"\nSaved to {out_path}")

    print(f"\n  >>> genm.noise.theta={theta_recommended:.1f}")
    return theta_recommended


if __name__ == '__main__':
    main()
