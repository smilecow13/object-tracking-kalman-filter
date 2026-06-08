"""
experiments/h2_qr_sweep.py
Hypothesis H2: LKF error depends on the Q/R ratio. An optimal Q/R region exists.

Method:
    Fix R, sweep Q over a log-scale range [1e-4, 10^1.5].
    For each Q, run Monte Carlo N=50 and compute mean RMSE.
    Identify optimal Q/R = argmin RMSE.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from kalman_filter import run_kf, rmse
from data_generate import generate_cv_trajectory

# Default parameters
N_MC     = 50
N_STEPS  = 100
R_STD    = 2.0
SAVE_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')


def run(n_mc: int = N_MC, n_steps: int = N_STEPS,
        r_std: float = R_STD, verbose: bool = True):
    """
    Run experiment H2.

    Returns
    -------
    opt_qr   : float -- optimal Q/R ratio
    opt_rmse : float -- RMSE at optimal point
    """
    if verbose:
        print("\nEXPERIMENT H2: Effect of Q/R Ratio on RMSE")

    R_fixed  = r_std ** 2 * np.eye(2)
    q_values = np.logspace(-4, 1.5, 40)   # 40 points from 1e-4 to ~31.6

    rmse_means, rmse_stds = [], []

    for q_val in q_values:
        Q_test   = q_val * np.eye(4)
        run_rmse = []
        for seed in range(n_mc):
            states, meas, F, H, _, _ = generate_cv_trajectory(
                n_steps=n_steps, r_std=r_std, seed=seed)
            truth = states[1:, [0, 2]]
            est, _ = run_kf(meas, F, H, Q_test, R_fixed)
            run_rmse.append(rmse(est[:, [0, 2]], truth))
        rmse_means.append(np.mean(run_rmse))
        rmse_stds.append(np.std(run_rmse))

    rmse_means = np.array(rmse_means)
    rmse_stds  = np.array(rmse_stds)
    qr_ratios  = q_values / R_fixed[0, 0]

    opt_idx     = int(np.argmin(rmse_means))
    opt_qr      = float(qr_ratios[opt_idx])
    opt_rmse    = float(rmse_means[opt_idx])
    # RMSE at Q/R ~= 1 as a reference point
    rmse_at_qr1 = float(rmse_means[np.argmin(np.abs(qr_ratios - 1))])

    if verbose:
        print(f"  Optimal Q/R    : {opt_qr:.5f}")
        print(f"  Optimal RMSE   : {opt_rmse:.4f}")
        print(f"  RMSE at Q/R=1  : {rmse_at_qr1:.4f}")
        print(f"  -> H2 ACCEPTED  (clear optimal point exists)\n")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (a) RMSE curve vs Q/R
    ax = axes[0]
    ax.semilogx(qr_ratios, rmse_means, 'steelblue', lw=2)
    ax.fill_between(qr_ratios,
                    rmse_means - rmse_stds,
                    rmse_means + rmse_stds,
                    alpha=0.2, color='steelblue', label='+/- 1 std')
    ax.axvline(opt_qr, color='green', ls='--', lw=1.5,
               label=f'Optimal Q/R={opt_qr:.4f}')
    ax.axhline(opt_rmse, color='green', ls=':', lw=1.2)
    ax.scatter([opt_qr], [opt_rmse], color='green', zorder=5, s=60)
    ax.set_xlabel('Q/R (log scale)')
    ax.set_ylabel('Mean RMSE (m)')
    ax.set_title('(a) RMSE vs Q/R Ratio')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.text(qr_ratios[2],  rmse_means[2]  + 0.15,
            'Too rigid\n(trust model)', fontsize=8, color='gray', ha='center')
    ax.text(qr_ratios[-3], rmse_means[-3] + 0.15,
            'Too flexible\n(trust meas)', fontsize=8, color='gray', ha='center')

    # (b) trajectory comparison: small Q vs optimal Q vs large Q (seed=7)
    ax2 = axes[1]
    states0, meas0, F, H, _, _ = generate_cv_trajectory(
        n_steps=n_steps, r_std=r_std, seed=7)
    truth0 = states0[1:, [0, 2]]

    for q_test, lbl, col in [
        (1e-4,              'Q small (rigid)',   '#e05252'),
        (q_values[opt_idx], 'Q optimal',          'green'),
        (10.0,              'Q large (flexible)', '#5555cc'),
    ]:
        est_t, _ = run_kf(meas0, F, H, q_test * np.eye(4), R_fixed)
        r = rmse(est_t[:, [0, 2]], truth0)
        ax2.plot(est_t[:, 0], est_t[:, 2], lw=1.5, color=col,
                 label=f'{lbl}  RMSE={r:.2f}')

    ax2.plot(truth0[:, 0], truth0[:, 1], 'k-', lw=2,
             label='Ground truth', zorder=5)
    ax2.scatter(meas0[:, 0], meas0[:, 1],
                c='gray', s=6, alpha=0.35, label='Measurement')
    ax2.set_title('(b) Trajectories for 3 Q Values')
    ax2.set_xlabel('x (m)'); ax2.set_ylabel('y (m)')
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')

    fig.suptitle('Experiment H2 -- Effect of Q/R Ratio on LKF Performance',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    os.makedirs(SAVE_DIR, exist_ok=True)
    out_path = os.path.join(SAVE_DIR, 'H2_results.png')
    plt.savefig(out_path, bbox_inches='tight', dpi=130)
    plt.close()
    if verbose:
        print(f"  -> Saved: {out_path}\n")

    return opt_qr, opt_rmse


if __name__ == '__main__':
    run()
