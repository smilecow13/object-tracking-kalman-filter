"""
data_generate.py
Generates simulation data (ground truth + noisy measurements) for two motion models:
    - CV: Constant Velocity     — state = [x, vx, y, vy]
    - CA: Constant Acceleration — state = [x, vx, ax, y, vy, ay]

Both models are discrete-time LTI systems with Gaussian noise.
Course: Signals and Systems
"""

import numpy as np


# Constant Velocity (CV) — used for H1, H2

def generate_cv_trajectory(n_steps: int = 100,
                            dt: float = 0.1,
                            q_std: float = 0.05,
                            r_std: float = 2.0,
                            seed: int = 0):
    """
    Generate a Constant Velocity (CV) trajectory.

    State vector: x_k = [x, vx, y, vy]^T  (4-dim)

    State transition matrix F:
        | 1  dt  0   0 |
        | 0   1  0   0 |
        | 0   0  1  dt |
        | 0   0  0   1 |

    Observation matrix H (position only):
        | 1  0  0  0 |
        | 0  0  1  0 |

    Parameters
    ----------
    n_steps : int   -- number of time steps
    dt      : float -- sampling period (s)
    q_std   : float -- process noise std,  w_k ~ N(0, q_std^2 * I)
    r_std   : float -- measurement noise std, v_k ~ N(0, r_std^2 * I)
    seed    : int   -- random seed for reproducible Monte Carlo runs

    Returns
    -------
    states       : ndarray, shape (n_steps+1, 4) -- true state x_k
    measurements : ndarray, shape (n_steps, 2)   -- noisy observation z_k = [x, y]
    F, H, Q, R   : model matrices
    """
    np.random.seed(seed)

    x0 = np.array([0.0, 1.5, 0.0, 0.8])

    F = np.array([[1, dt,  0,  0],
                  [0,  1,  0,  0],
                  [0,  0,  1, dt],
                  [0,  0,  0,  1]])

    H = np.array([[1, 0, 0, 0],
                  [0, 0, 1, 0]])

    Q = q_std ** 2 * np.eye(4)   # process noise covariance
    R = r_std ** 2 * np.eye(2)   # measurement noise covariance

    states = [x0.copy()]
    measurements = []
    x = x0.copy()

    W = np.random.multivariate_normal(np.zeros(4), Q, size=n_steps)
    V = np.random.multivariate_normal(np.zeros(2), R, size=n_steps)

    for i in range(n_steps):
        # dynamics: x_{k+1} = F * x_k + w_k
        x = F @ x + W[i]
        # observation: z_k = H * x_k + v_k
        z = H @ x + V[i]
        states.append(x.copy())
        measurements.append(z)


    return np.array(states), np.array(measurements), F, H, Q, R


# Constant Acceleration (CA) — used for H3 (model mismatch)

def generate_ca_trajectory(n_steps: int = 100,
                            dt: float = 0.1,
                            q_std: float = 0.05,
                            r_std: float = 2.0,
                            ax: float = 0.3,
                            ay: float = 0.2,
                            seed: int = 0):
    """
    Generate a Constant Acceleration (CA) trajectory.

    State vector: x_k = [x, vx, ax, y, vy, ay]^T  (6-dim)

    State transition matrix F:
        | 1  dt  dt^2/2  0   0       0    |
        | 0   1      dt  0   0       0    |
        | 0   0       1  0   0       0    |
        | 0   0       0  1  dt  dt^2/2    |
        | 0   0       0  0   1      dt    |
        | 0   0       0  0   0       1    |

    Observation matrix H (position only):
        | 1  0  0  0  0  0 |
        | 0  0  0  1  0  0 |

    Parameters
    ----------
    n_steps : int   -- number of time steps
    dt      : float -- sampling period (s)
    q_std   : float -- process noise std
    r_std   : float -- measurement noise std
    ax      : float -- initial acceleration along x (m/s^2)
    ay      : float -- initial acceleration along y (m/s^2)
    seed    : int   -- random seed

    Returns
    -------
    states       : ndarray, shape (n_steps+1, 6) -- true state x_k
    measurements : ndarray, shape (n_steps, 2)   -- noisy observation z_k = [x, y]
    F, H, Q, R   : model matrices
    """
    np.random.seed(seed)

    x0 = np.array([0.0, 1.5, ax, 0.0, 0.8, ay])

    F = np.array([
        [1, dt, 0.5 * dt ** 2, 0,  0,           0],
        [0,  1,            dt, 0,  0,           0],
        [0,  0,             1, 0,  0,           0],
        [0,  0,             0, 1, dt, 0.5 * dt ** 2],
        [0,  0,             0, 0,  1,          dt],
        [0,  0,             0, 0,  0,           1],
    ])

    H = np.array([[1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0]])

    Q = q_std ** 2 * np.eye(6)
    R = r_std ** 2 * np.eye(2)

    states = [x0.copy()]
    measurements = []
    x = x0.copy()

    W = np.random.multivariate_normal(np.zeros(6), Q, size=n_steps)
    V = np.random.multivariate_normal(np.zeros(2), R, size=n_steps)

    for i in range(n_steps):
        x = F @ x + W[i]
        z = H @ x + V[i]
        states.append(x.copy())
        measurements.append(z)


    return np.array(states), np.array(measurements), F, H, Q, R
