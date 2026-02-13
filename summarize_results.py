#!/usr/bin/env python
"""Summarize results across all training runs.

Reads eval_summary.json from each model directory and creates comparison tables
and a combined figure.

Usage:
    python summarize_results.py
    python summarize_results.py --models_dir models/ --output_dir results/
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_summaries(models_dir):
    """Find and load all eval_summary.json files."""
    summaries = {}
    for json_path in sorted(models_dir.rglob('eval_summary.json')):
        # Use parent of eval/ as run name
        run_dir = json_path.parent.parent
        run_name = str(run_dir.relative_to(models_dir))
        with open(json_path) as f:
            summaries[run_name] = json.load(f)
    return summaries


def make_convergence_table(summaries):
    """msp300 convergence: iterations vs metrics."""
    rows = []
    for name, s in summaries.items():
        # Match msp300_NNNN pattern
        if not name.startswith('msp300_'):
            continue
        try:
            iters = int(name.split('_')[1])
        except (IndexError, ValueError):
            continue
        rows.append({
            'iterations': iters,
            'best_f1': s.get('best_f1'),
            'best_precision': s.get('best_precision'),
            'best_recall': s.get('best_recall'),
            'best_rmse_vol': s.get('best_rmse_vol'),
            'best_threshold': s.get('best_threshold'),
            'n_pred_at_best': s.get('best_n_pred'),
        })
    if not rows:
        return None
    df = pd.DataFrame(rows).sort_values('iterations').reset_index(drop=True)
    return df


def make_sweep_table(summaries):
    """msp300 density sweep: density prior vs metrics."""
    rows = []
    for name, s in summaries.items():
        if not name.startswith('sweep/dens_'):
            continue
        dens_str = name.split('dens_')[1]
        # Convert back to float (e.g., "0001" -> 0.0001)
        try:
            dens_val = float('0.' + dens_str)
        except ValueError:
            continue
        rows.append({
            'density_high': dens_val,
            'best_f1': s.get('best_f1'),
            'best_precision': s.get('best_precision'),
            'best_recall': s.get('best_recall'),
            'best_rmse_vol': s.get('best_rmse_vol'),
            'best_threshold': s.get('best_threshold'),
        })
    if not rows:
        return None
    df = pd.DataFrame(rows).sort_values('density_high').reset_index(drop=True)
    return df


def make_celegans_table(summaries):
    """C. elegans: iterations vs detection counts at various thresholds."""
    rows = []
    for name, s in summaries.items():
        if not name.startswith('celegans_'):
            continue
        try:
            iters = int(name.split('_')[1])
        except (IndexError, ValueError):
            continue
        row = {'iterations': iters, 'n_total': s.get('n_total_pred')}
        for key in ['count_t0.30', 'count_t0.50', 'count_t0.70', 'count_t0.90', 'count_t0.95', 'count_t0.99']:
            row[key] = s.get(key)
        rows.append(row)
    if not rows:
        return None
    df = pd.DataFrame(rows).sort_values('iterations').reset_index(drop=True)
    return df


def plot_combined(convergence_df, sweep_df, celegans_df, output_dir):
    """Create combined summary figure."""
    n_panels = sum(x is not None for x in [convergence_df, sweep_df, celegans_df])
    if n_panels == 0:
        print("No data to plot.")
        return

    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]
    ax_idx = 0

    if convergence_df is not None:
        ax = axes[ax_idx]
        ax.plot(convergence_df['iterations'], convergence_df['best_f1'],
                'bo-', linewidth=2, markersize=8)
        ax.set_xlabel('Training Iterations')
        ax.set_ylabel('Best F1')
        ax.set_title('msp300 Convergence')
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        # Add RMSE on secondary axis
        ax2 = ax.twinx()
        valid = convergence_df['best_rmse_vol'].notna()
        ax2.plot(convergence_df.loc[valid, 'iterations'],
                 convergence_df.loc[valid, 'best_rmse_vol'],
                 'r^--', linewidth=1.5, markersize=7, alpha=0.7)
        ax2.set_ylabel('RMSE_vol (nm)', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax_idx += 1

    if sweep_df is not None:
        ax = axes[ax_idx]
        ax.bar(range(len(sweep_df)), sweep_df['best_f1'], color='steelblue', alpha=0.8)
        ax.set_xticks(range(len(sweep_df)))
        ax.set_xticklabels([f'{v:.4f}' for v in sweep_df['density_high']], rotation=45)
        ax.set_xlabel('prob_generator.high')
        ax.set_ylabel('Best F1')
        ax.set_title('msp300 Density Sweep (5K iters)')
        ax.grid(True, alpha=0.3, axis='y')
        ax_idx += 1

    if celegans_df is not None:
        ax = axes[ax_idx]
        for col in ['count_t0.50', 'count_t0.70', 'count_t0.90', 'count_t0.95']:
            if col in celegans_df.columns and celegans_df[col].notna().any():
                label = col.replace('count_t', 't=')
                ax.plot(celegans_df['iterations'], celegans_df[col],
                        'o-', linewidth=1.5, markersize=7, label=label)
        ax.set_xlabel('Training Iterations')
        ax.set_ylabel('Detection Count')
        ax.set_title('C. elegans Counts')
        ax.set_yscale('log')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax_idx += 1

    fig.tight_layout()
    fig.savefig(output_dir / 'summary_combined.png', dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Summarize DECODE-FISH training results')
    parser.add_argument('--models_dir', default=None,
                        help='Models directory (default: models/)')
    parser.add_argument('--output_dir', default=None,
                        help='Output directory (default: models/summary/)')
    args = parser.parse_args()

    base_dir = Path(__file__).parent
    models_dir = Path(args.models_dir) if args.models_dir else base_dir / 'models'
    output_dir = Path(args.output_dir) if args.output_dir else models_dir / 'summary'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Models dir: {models_dir}")
    print(f"Output dir: {output_dir}")

    # Load all summaries
    summaries = load_summaries(models_dir)
    print(f"\nFound {len(summaries)} eval results:")
    for name in sorted(summaries.keys()):
        print(f"  {name}")

    if not summaries:
        print("\nNo results found. Have the training jobs completed?")
        return

    # Build tables
    convergence_df = make_convergence_table(summaries)
    sweep_df = make_sweep_table(summaries)
    celegans_df = make_celegans_table(summaries)

    # Print and save tables
    if convergence_df is not None:
        print("\n=== msp300 Convergence ===")
        print(convergence_df.to_string(index=False))
        convergence_df.to_csv(output_dir / 'convergence.csv', index=False)

    if sweep_df is not None:
        print("\n=== msp300 Density Sweep ===")
        print(sweep_df.to_string(index=False))
        sweep_df.to_csv(output_dir / 'density_sweep.csv', index=False)

    if celegans_df is not None:
        print("\n=== C. elegans Detection Counts ===")
        print(celegans_df.to_string(index=False))
        celegans_df.to_csv(output_dir / 'celegans_counts.csv', index=False)

    # Combined figure
    plot_combined(convergence_df, sweep_df, celegans_df, output_dir)
    print(f"\nSummary saved to {output_dir}/")


if __name__ == '__main__':
    main()
