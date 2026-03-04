#!/usr/bin/env python3
"""
UKF Offline Test: Run the 1D UKF against LOG028.TXT flight data.
Plots altitude/velocity/acceleration estimates with 95% confidence ellipses,
ground truth baro altitude, and backup_seal.TXT comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import sys
import os

# ---------------------------------------------------------------------------
# Parse flight logs
# ---------------------------------------------------------------------------

def quat_rotate_accel(ax, ay, az, qi, qj, qk, qw):
    """Rotate body-frame acceleration to world frame using quaternion.
    Returns vertical (up) component of acceleration in world frame.
    Quaternion convention: (i, j, k, real/w) from BNO080."""
    # Rotation matrix from quaternion (body -> world)
    # Row 2 (z-world / up direction) of the rotation matrix:
    r20 = 2.0 * (qi * qk + qj * qw)
    r21 = 2.0 * (qj * qk - qi * qw)
    r22 = 1.0 - 2.0 * (qi * qi + qj * qj)

    # Vertical component of body-frame acceleration in world frame
    a_vertical = r20 * ax + r21 * ay + r22 * az
    return a_vertical


def parse_log028(path):
    """Parse LOG028.TXT -> dict of arrays. Delimiter: ', ' in data rows."""
    rows = []
    with open(path, 'r') as f:
        header_line = f.readline()  # skip header
        for line in f:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 20:
                continue
            try:
                ax = float(parts[1])
                ay = float(parts[2])
                az = float(parts[3])
                ax_hg = float(parts[4])
                ay_hg = float(parts[5])
                az_hg = float(parts[6])
                qi = float(parts[13])
                qj = float(parts[14])
                qk = float(parts[15])
                qw = float(parts[16])

                # Project acceleration to vertical using quaternion
                qmag = np.sqrt(qi**2 + qj**2 + qk**2 + qw**2)
                if qmag > 0.1:  # Valid quaternion
                    qi /= qmag; qj /= qmag; qk /= qmag; qw /= qmag
                    a_vert = quat_rotate_accel(ax, ay, az, qi, qj, qk, qw)
                    a_vert_hg = quat_rotate_accel(ax_hg, ay_hg, az_hg, qi, qj, qk, qw)
                else:
                    # No valid quat — fall back to Y-axis
                    a_vert = ay
                    a_vert_hg = ay_hg

                row = {
                    'time_ms': float(parts[0]),
                    'accel_y': ay,                   # raw Y-axis (for reference)
                    'accel_y_hg': ay_hg,
                    'accel_vert': a_vert,            # world-frame vertical accel (g's)
                    'accel_vert_hg': a_vert_hg,
                    'pressure': float(parts[7]),
                    'temperature': float(parts[8]),
                    'altitude': float(parts[9]),
                    'state': parts[17].strip(),
                    'airbrake_pct': float(parts[18]),
                }
                rows.append(row)
            except (ValueError, IndexError):
                continue
    return rows


def parse_backup_seal(path):
    """Parse backup_seal.TXT -> dict of arrays.
    Actual data format (15 fields):
    Time, Xg, Yg, Zg, Pressure, Temperature, Altitude, BNO_X, BNO_Y, BNO_Z,
    BNO_I, BNO_J, BNO_K, BNO_Real, State
    """
    rows = []
    with open(path, 'r') as f:
        header_line = f.readline()
        for line in f:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 15:
                continue
            try:
                row = {
                    'time_ms': float(parts[0]),
                    'accel_y': float(parts[2]),      # thrust axis (g's)
                    'pressure': float(parts[4]),
                    'temperature': float(parts[5]),
                    'altitude': float(parts[6]),     # already computed by backup FC
                    'state': parts[14].strip(),
                }
                rows.append(row)
            except (ValueError, IndexError):
                continue
    return rows


def compute_altitude(pressure, p_ref, t_ref_c, t_current_c=None):
    """Barometric altitude (meters AGL) from pressure.
    Mirrors STM32 altitudeDelta(): (Rd * Tbar / g0) * ln(p_ref / p)
    """
    Rd = 287.05
    g0 = 9.80665
    T_ref = t_ref_c + 273.15
    T = (t_current_c + 273.15) if t_current_c is not None else T_ref
    Tbar = 0.5 * (T_ref + T)
    if pressure <= 0 or p_ref <= 0:
        return 0.0
    return (Rd * Tbar / g0) * np.log(p_ref / pressure)


# ---------------------------------------------------------------------------
# UKF Implementation (mirrors the C++ version)
# ---------------------------------------------------------------------------

class UKF1D:
    """1D Unscented Kalman Filter: state = [altitude, velocity, acceleration]"""

    N = 3
    NUM_SIGMA = 2 * N + 1

    def __init__(self, alpha=1e-3, beta=2.0, kappa=0.0):
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.lam = alpha**2 * (self.N + kappa) - self.N

        # Weights
        self.Wm = np.zeros(self.NUM_SIGMA)
        self.Wc = np.zeros(self.NUM_SIGMA)
        n_plus_lam = self.N + self.lam
        self.Wm[0] = self.lam / n_plus_lam
        self.Wc[0] = self.lam / n_plus_lam + (1 - alpha**2 + beta)
        for i in range(1, self.NUM_SIGMA):
            self.Wm[i] = 1.0 / (2.0 * n_plus_lam)
            self.Wc[i] = 1.0 / (2.0 * n_plus_lam)

        self.x = np.zeros(self.N)
        self.P = np.eye(self.N)
        self.Q = np.diag([0.1**2, 0.5**2, 1.0**2])
        self.initialized = False

    def init(self, alt0, vel0, accel0):
        self.x = np.array([alt0, vel0, accel0])
        self.P = np.diag([10.0, 1.0, 5.0])
        self.initialized = True

    def _sigma_points(self):
        S = np.linalg.cholesky((self.N + self.lam) * self.P)
        sigmas = np.zeros((self.NUM_SIGMA, self.N))
        sigmas[0] = self.x
        for i in range(self.N):
            sigmas[i + 1] = self.x + S[:, i]
            sigmas[i + 1 + self.N] = self.x - S[:, i]
        return sigmas

    def predict(self, dt):
        if not self.initialized:
            return
        sigmas = self._sigma_points()

        # Propagate: constant acceleration model
        prop = np.zeros_like(sigmas)
        for i in range(self.NUM_SIGMA):
            alt, vel, acc = sigmas[i]
            prop[i, 0] = alt + vel * dt + 0.5 * acc * dt**2
            prop[i, 1] = vel + acc * dt
            prop[i, 2] = acc

        # Predicted mean & covariance
        x_pred = np.sum(self.Wm[:, None] * prop, axis=0)
        P_pred = self.Q.copy()
        for i in range(self.NUM_SIGMA):
            d = prop[i] - x_pred
            P_pred += self.Wc[i] * np.outer(d, d)

        self.x = x_pred
        self.P = P_pred
        self._sigmas_prop = prop  # save for update

    def _update_scalar(self, meas, H_index, R):
        """Generic scalar measurement update. H_index: which state is measured."""
        if not self.initialized:
            return
        sigmas = self._sigma_points()

        # Measurement predictions
        z_pred = sigmas[:, H_index]
        z_mean = np.sum(self.Wm * z_pred)

        # Innovation covariance
        S = R
        for i in range(self.NUM_SIGMA):
            dz = z_pred[i] - z_mean
            S += self.Wc[i] * dz**2

        # Cross covariance
        Pxz = np.zeros(self.N)
        for i in range(self.NUM_SIGMA):
            dz = z_pred[i] - z_mean
            Pxz += self.Wc[i] * (sigmas[i] - self.x) * dz

        # Kalman gain
        K = Pxz / S
        innovation = meas - z_mean
        self.x += K * innovation
        self.P -= np.outer(K, K) * S

    def update_accel(self, meas, R):
        self._update_scalar(meas, 2, R)

    def update_baro(self, meas, R):
        self._update_scalar(meas, 0, R)


# ---------------------------------------------------------------------------
# Sensor weighting (mirrors C++ SensorWeighting)
# ---------------------------------------------------------------------------

def get_sensor_weights(state_str, velocity, speed_of_sound=340.0):
    mach = abs(velocity) / speed_of_sound

    if state_str in ('IDLE', 'BOOT', 'AIRBRAKE_TEST', 'LANDED'):
        return 0.5, 100.0, 1.0  # R_low, R_high, R_baro
    elif state_str == 'IGNITION':
        return 1000.0, 1.0, 5.0
    elif state_str == 'ASCENT':
        if 0.8 < mach < 1.2:
            return 1.0, 2.0, 500.0  # transonic
        elif abs(velocity) < 20.0:
            return 0.5, 50.0, 0.5   # near apogee
        else:
            return 1.0, 3.0, 2.0    # subsonic coast
    elif state_str in ('APOGEE', 'DESCENT'):
        return 1.0, 100.0, 0.5
    else:
        return 5.0, 5.0, 5.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    analysis_dir = os.path.dirname(os.path.abspath(__file__))
    log028_path = os.path.join(analysis_dir, 'LOG028.TXT')
    backup_path = os.path.join(analysis_dir, 'backup_seal.TXT')

    print("Parsing LOG028.TXT...")
    data = parse_log028(log028_path)
    print(f"  {len(data)} rows parsed")

    print("Parsing backup_seal.TXT...")
    backup = parse_backup_seal(backup_path)
    print(f"  {len(backup)} rows parsed")

    # Find flight window: 5s before IGNITION to 60s after APOGEE
    flight_start_idx = next(i for i, r in enumerate(data) if r['state'] == 'IGNITION')
    t_ignition = data[flight_start_idx]['time_ms']
    start_idx = max(0, flight_start_idx - 70)  # ~5s before ignition

    # Find first DESCENT entry and go 60s past it
    descent_idx = next((i for i in range(len(data)) if data[i]['state'] == 'DESCENT'), len(data) - 1)
    t_descent = data[descent_idx]['time_ms']
    flight_end_idx = descent_idx
    for i in range(descent_idx, len(data)):
        if data[i]['time_ms'] - t_descent > 60000:  # 60s after descent
            break
        flight_end_idx = i

    flight_data = data[start_idx:flight_end_idx + 1]
    print(f"  Flight window: {len(flight_data)} rows, "
          f"t={flight_data[0]['time_ms']:.0f} to {flight_data[-1]['time_ms']:.0f} ms")

    # Reference pressure (average of first 50 IDLE readings)
    idle_rows = [r for r in data if r['state'] == 'IDLE'][:50]
    p_ref = np.mean([r['pressure'] for r in idle_rows])
    t_ref = np.mean([r['temperature'] for r in idle_rows])
    print(f"  Reference: p_ref={p_ref:.2f} hPa, t_ref={t_ref:.2f} C")

    # ---------- Run UKF ----------
    ukf = UKF1D()
    G = 9.80665

    times = []
    alt_est = []
    vel_est = []
    acc_est = []
    alt_baro = []
    P_history = []  # store diagonal of P for confidence bands

    airbrake_pcts = []
    states = []

    for i, row in enumerate(flight_data):
        t_s = row['time_ms'] / 1000.0

        if i == 0:
            ukf.init(row['altitude'], 0.0, 0.0)
            dt = 0.05
        else:
            dt = (row['time_ms'] - flight_data[i - 1]['time_ms']) / 1000.0
            if dt <= 0 or dt > 1.0:
                dt = 0.05

        # Predict
        ukf.predict(dt)

        # Sensor weights
        R_low, R_high, R_baro = get_sensor_weights(
            row['state'], ukf.x[1])

        # Accelerometer updates (convert g -> m/s^2, subtract gravity)
        # Using world-frame vertical acceleration (quaternion-rotated)
        accel_vert_ms2 = (row['accel_vert'] - 1.0) * G  # subtract 1g gravity
        ukf.update_accel(accel_vert_ms2, R_low)

        # High-g update
        accel_vert_hg_ms2 = (row['accel_vert_hg'] - 1.0) * G
        ukf.update_accel(accel_vert_hg_ms2, R_high)

        # Baro update
        ukf.update_baro(row['altitude'], R_baro)

        times.append(t_s)
        alt_est.append(ukf.x[0])
        vel_est.append(ukf.x[1])
        acc_est.append(ukf.x[2])
        alt_baro.append(row['altitude'])
        P_history.append(np.diag(ukf.P).copy())
        airbrake_pcts.append(row['airbrake_pct'])
        states.append(row['state'])

    times = np.array(times)
    alt_est = np.array(alt_est)
    vel_est = np.array(vel_est)
    acc_est = np.array(acc_est)
    alt_baro = np.array(alt_baro)
    P_history = np.array(P_history)
    airbrake_pcts = np.array(airbrake_pcts)

    # 95% confidence (2 sigma)
    alt_std = np.sqrt(P_history[:, 0])
    vel_std = np.sqrt(P_history[:, 1])
    acc_std = np.sqrt(P_history[:, 2])

    # ---------- Backup seal data (align by time) ----------

    backup_times = []
    backup_alts = []
    # Use pre-computed altitude from backup flight computer
    for r in backup:
        backup_times.append(r['time_ms'] / 1000.0)
        backup_alts.append(r['altitude'])

    backup_times = np.array(backup_times)
    backup_alts = np.array(backup_alts)

    # Time offset: align backup to primary by matching IGNITION timestamps
    primary_ign_t = t_ignition / 1000.0
    backup_ign_idx = next((i for i, r in enumerate(backup) if r['state'] == 'IGNITION'), None)
    if backup_ign_idx is not None:
        backup_ign_t = backup[backup_ign_idx]['time_ms'] / 1000.0
        time_offset = primary_ign_t - backup_ign_t
        backup_times = backup_times + time_offset
        print(f"  Backup time offset: {time_offset:.3f} s")

    # ---------- Plot ----------
    # Relative time (seconds from ignition)
    t0 = t_ignition / 1000.0
    t_rel = times - t0
    backup_t_rel = backup_times - t0

    # Filter backup data to flight window
    t_min = t_rel[0] - 5
    t_max = t_rel[-1] + 5
    mask = (backup_t_rel >= t_min) & (backup_t_rel <= t_max)
    backup_t_rel = backup_t_rel[mask]
    backup_alts = backup_alts[mask]

    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)

    # --- Altitude ---
    ax = axes[0]
    ax.fill_between(t_rel, alt_est - 2 * alt_std, alt_est + 2 * alt_std,
                     alpha=0.2, color='C0', label='95% CI')
    ax.plot(t_rel, alt_est, 'C0-', linewidth=1.5, label='UKF Estimate')
    ax.plot(t_rel, alt_baro, 'k--', linewidth=0.8, alpha=0.5, label='Baro (raw)')
    if len(backup_t_rel) > 0:
        ax.plot(backup_t_rel, backup_alts, 'C3-', linewidth=0.8, alpha=0.6,
                label='Backup Seal (baro)')
    ax.set_ylabel('Altitude (m AGL)')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title('UKF State Estimation — LOG028 Flight Data')
    ax.grid(True, alpha=0.3)

    # Add confidence ellipses every 1 second on altitude subplot
    ellipse_interval = 1.0  # seconds
    last_ellipse_t = -999
    for i in range(len(t_rel)):
        if t_rel[i] - last_ellipse_t >= ellipse_interval:
            # Draw altitude-velocity ellipse as a marker with error bar
            ax.plot(t_rel[i], alt_est[i], 'o', color='C0', markersize=3, zorder=5)
            ax.errorbar(t_rel[i], alt_est[i], yerr=2 * alt_std[i],
                       fmt='none', ecolor='C0', alpha=0.4, capsize=2)
            last_ellipse_t = t_rel[i]

    # --- Velocity ---
    ax = axes[1]
    ax.fill_between(t_rel, vel_est - 2 * vel_std, vel_est + 2 * vel_std,
                     alpha=0.2, color='C1')
    ax.plot(t_rel, vel_est, 'C1-', linewidth=1.5, label='UKF Velocity')
    ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
    ax.set_ylabel('Velocity (m/s)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Mark Mach 1 line (approx 340 m/s)
    ax.axhline(y=340, color='red', linewidth=0.5, linestyle=':', alpha=0.5)
    ax.text(t_rel[0] + 0.5, 345, 'Mach 1', fontsize=7, color='red', alpha=0.5)

    # --- Acceleration ---
    ax = axes[2]
    ax.fill_between(t_rel, acc_est - 2 * acc_std, acc_est + 2 * acc_std,
                     alpha=0.2, color='C2')
    ax.plot(t_rel, acc_est, 'C2-', linewidth=1.5, label='UKF Acceleration')
    ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
    ax.set_ylabel('Acceleration (m/s²)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Airbrake deployment ---
    ax = axes[3]
    ax.plot(t_rel, airbrake_pcts, 'C4-', linewidth=1.5, label='Airbrake %')
    ax.set_ylabel('Airbrake (%)')
    ax.set_xlabel('Time from Ignition (s)')
    ax.set_ylim(0, 70)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Shade flight phases
    for ax_i in axes:
        # Find phase boundaries in relative time
        for phase, color_val in [('IGNITION', 'orange'), ('ASCENT', 'red'),
                                  ('APOGEE', 'purple'), ('DESCENT', 'blue')]:
            phase_times = [t_rel[j] for j in range(len(states)) if states[j] == phase]
            if phase_times:
                ax_i.axvspan(min(phase_times), max(phase_times), alpha=0.05, color=color_val)
                if ax_i == axes[0]:
                    ax_i.text(np.mean(phase_times), ax_i.get_ylim()[1] * 0.95,
                             phase, fontsize=7, ha='center', va='top',
                             color=color_val, alpha=0.7)

    plt.tight_layout()
    out_path = os.path.join(analysis_dir, 'ukf_test_results.png')
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")

    # --- Also make alt-vel phase portrait with confidence ellipses ---
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 8))
    ax2.plot(alt_est, vel_est, 'C0-', linewidth=1, alpha=0.6, label='UKF trajectory')

    # Draw confidence ellipses every 2 seconds
    last_ellipse_t = -999
    for i in range(len(t_rel)):
        if t_rel[i] - last_ellipse_t >= 2.0:
            # 2D confidence ellipse in alt-vel plane
            # Extract 2x2 sub-covariance
            P_sub = P_history[i][:2].reshape(-1)  # just diagonals
            w_alt = 2 * np.sqrt(P_history[i][0])  # 2-sigma
            w_vel = 2 * np.sqrt(P_history[i][1])

            ellipse = Ellipse(xy=(alt_est[i], vel_est[i]),
                             width=2 * w_alt, height=2 * w_vel,
                             angle=0, facecolor='C0', alpha=0.15,
                             edgecolor='C0', linewidth=0.5)
            ax2.add_patch(ellipse)
            ax2.plot(alt_est[i], vel_est[i], 'ko', markersize=2)
            ax2.annotate(f'{t_rel[i]:.0f}s', (alt_est[i], vel_est[i]),
                        fontsize=6, alpha=0.6,
                        xytext=(5, 5), textcoords='offset points')
            last_ellipse_t = t_rel[i]

    ax2.plot(alt_baro, vel_est, 'k--', linewidth=0.5, alpha=0.3, label='Baro alt vs UKF vel')
    ax2.set_xlabel('Altitude (m AGL)')
    ax2.set_ylabel('Velocity (m/s)')
    ax2.set_title('Phase Portrait: Altitude vs Velocity (95% Confidence Ellipses)')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    out_path2 = os.path.join(analysis_dir, 'ukf_phase_portrait.png')
    plt.savefig(out_path2, dpi=150)
    print(f"Saved phase portrait to {out_path2}")

    # Print summary stats
    print(f"\n--- UKF Summary ---")
    print(f"Max estimated altitude: {np.max(alt_est):.1f} m")
    print(f"Max baro altitude:      {np.max(alt_baro):.1f} m")
    print(f"Max estimated velocity: {np.max(vel_est):.1f} m/s")
    print(f"Max estimated Mach:     {np.max(vel_est) / 340:.2f}")
    if len(backup_alts) > 0:
        print(f"Max backup seal alt:    {np.max(backup_alts):.1f} m")

    # plt.show()  # Uncomment to display interactively


if __name__ == '__main__':
    main()
