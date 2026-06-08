"""
experiments/h1_lkf_vs_raw.py
Hypothesis H1: LKF produces significantly smaller estimation error than raw
measurements under Gaussian noise.

Method: Monte Carlo N=50 runs with CV model, compare RMSE(raw) vs RMSE(KF).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from kalman_filter import run_kf, rmse
from data_generate import generate_cv_trajectory

# Default parameters
N_MC     = 50     # number of Monte Carlo runs
N_STEPS  = 100    # number of time steps
R_STD    = 2.0    # measurement noise std (m)
SAVE_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')


def run(n_mc: int = N_MC, n_steps: int = N_STEPS,
        r_std: float = R_STD, verbose: bool = True):
    """
    Run experiment H1.

    Returns
    -------
    raw_mean : float -- mean RMSE of raw measurements
    kf_mean  : float -- mean RMSE of KF estimates
    improve  : float -- improvement percentage
    """
    if verbose:
        print("\nEXPERIMENT H1: LKF vs Raw Measurement")

    rmse_raw_list, rmse_kf_list = [], []

    for seed in range(n_mc):
        states, meas, F, H, Q, R = generate_cv_trajectory(
            n_steps=n_steps, r_std=r_std, seed=seed)
        truth    = states[1:, [0, 2]]   # true position [x, y]
        meas_pos = meas                  # raw measurement is already [x, y]
        est, _   = run_kf(meas, F, H, Q, R)
        est_pos  = est[:, [0, 2]]

        rmse_raw_list.append(rmse(meas_pos, truth))
        rmse_kf_list.append(rmse(est_pos,   truth))

    raw_mean = np.mean(rmse_raw_list)
    kf_mean  = np.mean(rmse_kf_list)
    kf_std   = np.std(rmse_kf_list)
    improve  = (1 - kf_mean / raw_mean) * 100

    if verbose:
        print(f"  RMSE raw       : {raw_mean:.4f} +/- {np.std(rmse_raw_list):.4f}")
        print(f"  RMSE KF        : {kf_mean:.4f} +/- {kf_std:.4f}")
        print(f"  Improvement    : {improve:.1f}%")
        if improve > 20:
            print("  -> H1 ACCEPTED")
        else:
            print("  -> H1 REJECTED")

    # sample run (seed=7) for visualization
    states0, meas0, F, H, Q, R = generate_cv_trajectory(
        n_steps=n_steps, r_std=r_std, seed=7)
    truth0   = states0[1:, [0, 2]]
    est0, gains0 = run_kf(meas0, F, H, Q, R)
    est0_pos = est0[:, [0, 2]]
    t        = np.arange(n_steps) * 0.1

    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)

    # (a) 2D trajectory
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(truth0[:, 0], truth0[:, 1], 'g-', lw=2, label='Ground truth')
    ax0.scatter(meas0[:, 0], meas0[:, 1],
                c='#e05252', s=8, alpha=0.5, label='Measurement (raw)')
    ax0.plot(est0_pos[:, 0], est0_pos[:, 1],
             'b-', lw=1.5, label='KF estimate')
    ax0.set_title('(a) 2D Trajectory')
    ax0.set_xlabel('x (m)'); ax0.set_ylabel('y (m)')
    ax0.legend(fontsize=9); ax0.set_aspect('equal')
    ax0.grid(True, alpha=0.3)

    # (b) position error over time
    ax1 = fig.add_subplot(gs[0, 1])
    e_raw = np.sqrt(np.sum((meas0    - truth0) ** 2, axis=1))
    e_kf  = np.sqrt(np.sum((est0_pos - truth0) ** 2, axis=1))
    ax1.plot(t, e_raw, color='#e05252', alpha=0.7,
             label=f'Raw     RMSE={e_raw.mean():.2f} m')
    ax1.plot(t, e_kf, color='steelblue', lw=1.8,
             label=f'KF      RMSE={e_kf.mean():.2f} m')
    ax1.set_title('(b) Position Error over Time')
    ax1.set_xlabel('Time (s)'); ax1.set_ylabel('Error (m)')
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

    # (c) Kalman gain convergence
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(t, gains0[:, 0, 0], label='K[x,x]', color='steelblue')
    ax2.plot(t, gains0[:, 1, 1], label='K[y,y]', color='darkorange')
    ax2.set_title('(c) Kalman Gain Convergence')
    ax2.set_xlabel('Time (s)'); ax2.set_ylabel('Kalman Gain')
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)

    # (d) RMSE distribution — Monte Carlo box plot
    ax3 = fig.add_subplot(gs[1, 1])
    bp  = ax3.boxplot([rmse_raw_list, rmse_kf_list],
                      tick_labels=['Raw', 'KF estimate'],
                      patch_artist=True,
                      medianprops=dict(color='black', lw=2))
    bp['boxes'][0].set_facecolor('#f5a0a0')
    bp['boxes'][1].set_facecolor('#a0c4f5')
    ax3.set_title(f'(d) RMSE Distribution -- Monte Carlo N={n_mc}')
    ax3.set_ylabel('RMSE (m)'); ax3.grid(True, alpha=0.3, axis='y')
    ax3.text(1.5, max(rmse_raw_list) * 0.92,
             f'Improvement\n{improve:.1f}%',
             ha='center', fontsize=10, color='green', fontweight='bold')

    fig.suptitle(
        'Experiment H1 -- LKF vs Raw Measurement (Gaussian Noise, CV model)',
        fontsize=13, fontweight='bold', y=1.01)

    os.makedirs(SAVE_DIR, exist_ok=True)
    out_path = os.path.join(SAVE_DIR, 'H1_results.png')
    plt.savefig(out_path, bbox_inches='tight', dpi=130)
    plt.close()
    if verbose:
        print(f"  -> Saved: {out_path}\n")

    return raw_mean, kf_mean, improve


if __name__ == '__main__':
    run()
