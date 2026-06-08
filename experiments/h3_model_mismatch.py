"""
experiments/h3_model_mismatch.py
Hypothesis H3: When the motion model matches reality, LKF converges quickly and
error is small. Under model mismatch, error increases measurably.

Scenario:
    - Data generated from CA (Constant Acceleration) model.
    - KF-CA: uses correct CA model  -> "matched".
    - KF-CV: uses wrong CV model    -> "mismatched".
    Compare RMSE of both filters over Monte Carlo N=50 runs.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from kalman_filter import run_kf, rmse
from data_generate import generate_ca_trajectory

# Default parameters
N_MC     = 50
N_STEPS  = 100
R_STD    = 2.0
DT       = 0.1
SAVE_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

# CV matrices (mismatched model, fixed for KF-CV)
_F_CV = np.array([[1, DT,  0,   0],
                   [0,  1,  0,   0],
                   [0,  0,  1,  DT],
                   [0,  0,  0,   1]])
_H_CV = np.array([[1, 0, 0, 0],
                   [0, 0, 1, 0]])
_Q_CV = 0.05 ** 2 * np.eye(4)


def run(n_mc: int = N_MC, n_steps: int = N_STEPS,
        r_std: float = R_STD, verbose: bool = True):
    """
    Run experiment H3.

    Returns
    -------
    match_mean    : float -- mean RMSE of KF-CA (matched model)
    mismatch_mean : float -- mean RMSE of KF-CV (mismatched model)
    degradation   : float -- performance degradation (%)
    """
    if verbose:
        print("\nEXPERIMENT H3: Model Mismatch (CA data vs CV filter)")

    R_cv = r_std ** 2 * np.eye(2)
    rmse_match, rmse_mismatch = [], []

    for seed in range(n_mc):
        states, meas, F_ca, H_ca, Q_ca, R_ca = generate_ca_trajectory(
            n_steps=n_steps, r_std=r_std, seed=seed)
        # true position from CA state [x, vx, ax, y, vy, ay] -> indices 0, 3
        truth = states[1:, [0, 3]]

        # KF with correct model (CA)
        est_ca, _ = run_kf(meas, F_ca, H_ca, Q_ca, R_ca)
        pos_ca    = est_ca[:, [0, 3]]
        rmse_match.append(rmse(pos_ca, truth))

        # KF with wrong model (CV)
        est_cv, _ = run_kf(meas, _F_CV, _H_CV, _Q_CV, R_cv)
        pos_cv    = est_cv[:, [0, 2]]   # CV state: [x, vx, y, vy]
        rmse_mismatch.append(rmse(pos_cv, truth))

    match_mean    = float(np.mean(rmse_match))
    mismatch_mean = float(np.mean(rmse_mismatch))
    degradation   = (mismatch_mean / match_mean - 1) * 100

    if verbose:
        print(f"  RMSE KF-CA (matched)    : {match_mean:.4f} +/- {np.std(rmse_match):.4f}")
        print(f"  RMSE KF-CV (mismatched) : {mismatch_mean:.4f} +/- {np.std(rmse_mismatch):.4f}")
        print(f"  Performance degradation : +{degradation:.1f}%")
        if degradation > 10:
            print("  -> H3 ACCEPTED")
        else:
            print("  -> H3 REJECTED")

    # sample run (seed=7) for visualization
    states0, meas0, F_ca0, H_ca0, Q_ca0, R_ca0 = generate_ca_trajectory(
        n_steps=n_steps, r_std=r_std, seed=7)
    truth0   = states0[1:, [0, 3]]
    t        = np.arange(n_steps) * DT
    R_cv0    = r_std ** 2 * np.eye(2)

    est_ca0, _ = run_kf(meas0, F_ca0, H_ca0, Q_ca0, R_ca0)
    est_cv0, _ = run_kf(meas0, _F_CV, _H_CV, _Q_CV, R_cv0)
    pos_ca0    = est_ca0[:, [0, 3]]
    pos_cv0    = est_cv0[:, [0, 2]]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (a) 2D trajectory
    ax = axes[0]
    ax.plot(truth0[:, 0],  truth0[:, 1],  'k-',   lw=2,
            label='Ground truth (CA)')
    ax.scatter(meas0[:, 0], meas0[:, 1],
               c='gray', s=6, alpha=0.35, label='Measurement')
    ax.plot(pos_ca0[:, 0], pos_ca0[:, 1], 'green', lw=1.8,
            label=f'KF-CA  RMSE={rmse(pos_ca0, truth0):.2f}')
    ax.plot(pos_cv0[:, 0], pos_cv0[:, 1], '#e05252', lw=1.5, ls='--',
            label=f'KF-CV  RMSE={rmse(pos_cv0, truth0):.2f}')
    ax.set_title('(a) 2D Trajectory')
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # (b) position error over time
    ax2 = axes[1]
    e_ca = np.sqrt(np.sum((pos_ca0 - truth0) ** 2, axis=1))
    e_cv = np.sqrt(np.sum((pos_cv0 - truth0) ** 2, axis=1))
    ax2.plot(t, e_ca, 'green',   lw=1.8, label='KF-CA (matched)')
    ax2.plot(t, e_cv, '#e05252', lw=1.5, ls='--', label='KF-CV (mismatched)')
    ax2.fill_between(t, e_cv, e_ca,
                     where=(e_cv > e_ca), alpha=0.15,
                     color='red', label='Degradation region')
    ax2.set_title('(b) Position Error over Time')
    ax2.set_xlabel('Time (s)'); ax2.set_ylabel('Error (m)')
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

    # (c) Monte Carlo box plot
    ax3 = axes[2]
    bp  = ax3.boxplot([rmse_match, rmse_mismatch],
                      tick_labels=['KF-CA\n(matched)', 'KF-CV\n(mismatch)'],
                      patch_artist=True,
                      medianprops=dict(color='black', lw=2))
    bp['boxes'][0].set_facecolor('#a0f5a0')
    bp['boxes'][1].set_facecolor('#f5a0a0')
    ax3.set_title(f'(c) RMSE Distribution -- Monte Carlo N={n_mc}')
    ax3.set_ylabel('RMSE (m)'); ax3.grid(True, alpha=0.3, axis='y')
    ax3.text(1.5, max(rmse_mismatch) * 0.94,
             f'Degradation\n+{degradation:.0f}%',
             ha='center', fontsize=10, color='red', fontweight='bold')

    fig.suptitle(
        'Experiment H3 -- Model Mismatch: CA data with CV filter vs CA filter',
        fontsize=13, fontweight='bold')
    plt.tight_layout()

    os.makedirs(SAVE_DIR, exist_ok=True)
    out_path = os.path.join(SAVE_DIR, 'H3_results.png')
    plt.savefig(out_path, bbox_inches='tight', dpi=130)
    plt.close()
    if verbose:
        print(f"\n  -> Saved: {out_path}\n")

    return match_mean, mismatch_mean, degradation


if __name__ == '__main__':
    run()
