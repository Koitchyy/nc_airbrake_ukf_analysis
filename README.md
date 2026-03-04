# Flight Analysis

Python tools for post-flight data analysis of rocket avionics. Part of the SSI/IREC airbrake control system.

## Overview

This repository provides offline analysis scripts for flight logs recorded by the STM32 flight controller. It includes state estimation validation (UKF), acceleration/airbrake visualization, and 3D orientation visualization.

## Requirements

- Python ≥ 3.8
- NumPy, Matplotlib, SciPy

## Installation

```bash
# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .
```

Or install dependencies directly:

```bash
pip install numpy matplotlib "scipy>=1.10.1"
```

## Scripts

| Script | Description |
|--------|-------------|
| `ukf_test.py` | Runs the 1D Unscented Kalman Filter against flight data. Validates altitude, velocity, and acceleration estimates with 95% confidence intervals. Compares primary baro with backup flight computer data. |
| `launch_log.py` | Parses flight logs and plots acceleration components (Xg, Yg, Zg, Yg_high) and airbrake deployment vs time. |
| `vis_rocket.py` | 3D visualization of rocket body frame orientation using quaternion data. |

## Data Files

Place the following files in the `analysis/` directory:

| File | Source | Description |
|------|--------|-------------|
| `LOG028.TXT` | STM32 SD card | Primary flight log (CSV: time, accel, pressure, altitude, quaternion, state, airbrake %) |
| `backup_seal.TXT` | Backup flight computer | Secondary baro/accel log for cross-validation (optional for `ukf_test.py`) |

## Usage

### UKF State Estimation Test

Validates the 1D UKF (mirrors `airbrakes-teensy-controls` C++ implementation) against recorded flight data:

```bash
python ukf_test.py
```

**Outputs:**
- `ukf_test_results.png` — Altitude, velocity, acceleration estimates with 95% CI; baro comparison; airbrake deployment; flight phase shading
- `ukf_phase_portrait.png` — Altitude vs velocity phase portrait with confidence ellipses

### Launch Log Analysis

```bash
python launch_log.py
```

Generates `acceleration.png` with acceleration components and airbrake deployment vs time.

### Rocket Orientation Visualization

```bash
python vis_rocket.py
```

Displays a 3D plot of the rocket body frame relative to the world frame (edit quaternion values in the script for different orientations).

## Flight States

The analysis scripts recognize the following flight states from the log:

`BOOT` → `IDLE` → `AIRBRAKE_TEST` → `IGNITION` → `ASCENT` → `APOGEE` → `DESCENT` → `LANDED`

## UKF Implementation

The Python `UKF1D` in `ukf_test.py` mirrors the C++ implementation in `airbrakes-teensy-controls/lib/StateEstimation/UKF1D.cpp`:

- **State vector:** `[altitude, velocity, acceleration]` (m, m/s, m/s²)
- **Measurements:** Barometric altitude, vertical acceleration (quaternion-rotated from body frame)
- **Sensor weighting:** State-dependent noise (e.g., higher accel trust during IGNITION, higher baro trust near apogee)

## License

Internal use for SSI/IREC competition.
