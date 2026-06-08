"""
kalman_filter.py
Core implementation of the Linear Kalman Filter (LKF) for 2D object tracking.
Course: Signals and Systems
"""

import numpy as np


class LinearKalmanFilter:
    """
    Linear Kalman Filter (LKF) for a discrete-time LTI system.

    Model:
        x_{k+1} = F * x_k + w_k,    w_k ~ N(0, Q)
        z_k     = H * x_k + v_k,    v_k ~ N(0, R)

    Parameters
    ----------
    F       : ndarray, shape (n, n) -- state transition matrix
    H       : ndarray, shape (m, n) -- observation matrix
    Q       : ndarray, shape (n, n) -- process noise covariance
    R       : ndarray, shape (m, m) -- measurement noise covariance
    n_state : int                   -- state dimension
    """

    def __init__(self, F: np.ndarray, H: np.ndarray,
                 Q: np.ndarray, R: np.ndarray, n_state: int):
        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        # init state at origin, large initial covariance (high uncertainty)
        self.x = np.zeros(n_state)
        self.P = np.eye(n_state) * 500.0

    def predict(self) -> np.ndarray:
        """
        Predict step (time update):
            x_{k|k-1} = F * x_{k-1|k-1}
            P_{k|k-1} = F * P_{k-1|k-1} * F^T + Q
        """
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    def update(self, z: np.ndarray):
        """
        Update step (measurement update):
            S_k = H * P_{k|k-1} * H^T + R
            K_k = P_{k|k-1} * H^T * S_k^{-1}   (Kalman gain)
            x_{k|k} = x_{k|k-1} + K_k * (z_k - H * x_{k|k-1})
            P_{k|k} = (I - K_k * H) * P_{k|k-1}

        Parameters
        ----------
        z : ndarray, shape (m,) -- measurement at step k

        Returns
        -------
        x_est : ndarray -- estimated state after update
        K     : ndarray -- Kalman gain at step k
        """
        S = self.H @ self.P @ self.H.T + self.R          # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)         # Kalman gain
        innov = z - self.H @ self.x                       # innovation (residual)
        self.x = self.x + K @ innov
        # Joseph form keeps P symmetric (numerically more stable)
        I_KH = np.eye(len(self.x)) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        return self.x.copy(), K.copy()


def run_kf(measurements: np.ndarray,
           F: np.ndarray, H: np.ndarray,
           Q: np.ndarray, R: np.ndarray):
    """
    Run LKF over a full measurement sequence.

    Parameters
    ----------
    measurements : ndarray, shape (T, m) -- sequence z_1, ..., z_T
    F, H, Q, R   : model matrices (see LinearKalmanFilter)

    Returns
    -------
    estimates : ndarray, shape (T, n)    -- state estimates x_{k|k}
    gains     : ndarray, shape (T, n, m) -- Kalman gain K_k at each step
    """
    kf = LinearKalmanFilter(F, H, Q, R, n_state=F.shape[0])
    estimates, gains = [], []
    for z in measurements:
        kf.predict()
        x_est, K = kf.update(z)
        estimates.append(x_est)
        gains.append(K)
    return np.array(estimates), np.array(gains)


def rmse(pred_pos: np.ndarray, true_pos: np.ndarray) -> float:
    """
    Root Mean Square Error (RMSE) on 2D position.

    RMSE = sqrt( (1/T) * sum_k ||pred_k - true_k||^2 )

    Parameters
    ----------
    pred_pos : ndarray, shape (T, 2) -- estimated positions [x, y]
    true_pos : ndarray, shape (T, 2) -- true positions [x, y]

    Returns
    -------
    float -- RMSE in meters
    """
    return float(np.sqrt(np.mean(np.sum((pred_pos - true_pos) ** 2, axis=1))))