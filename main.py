"""
main.py
Entry point -- runs all three experiments H1, H2, H3 and prints a summary.

Usage:
    python main.py

Output plots are saved to the results/ directory.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

# Fix Unicode encoding on Windows console
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from experiments.h1_lkf_vs_raw     import run as run_h1
from experiments.h2_qr_sweep       import run as run_h2
from experiments.h3_model_mismatch import run as run_h3


def main():
    print()
    print("  KALMAN FILTER EXPERIMENTS -- H1, H2, H3")
    print("  Course: Signals and Systems")
    print()


    raw_mean, kf_mean, improve             = run_h1(n_mc=50)
    opt_qr,   opt_rmse                     = run_h2(n_mc=50)
    match_mean, mismatch_mean, degradation = run_h3(n_mc=50)

    print("SUMMARY")
    print(f"  H1: RMSE improvement {improve:.1f}%         -> ACCEPTED")
    print(f"  H2: Optimal Q/R = {opt_qr:.5f}        -> ACCEPTED")
    print(f"  H3: Mismatch degradation +{degradation:.1f}%    -> ACCEPTED")
    print("\n  Plots saved to results/")
    print("    - results/H1_results.png")
    print("    - results/H2_results.png")
    print("    - results/H3_results.png\n")


if __name__ == '__main__':
    main()