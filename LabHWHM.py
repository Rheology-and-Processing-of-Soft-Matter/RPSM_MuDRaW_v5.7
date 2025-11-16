#Created by Roland Kádár on 2024-06-10.
#Copyright (c) 2024 Chalmers University of Technology. All rights reserved.
#Based on an earlier code made for Matlab by Roland Kádár in 2021.

import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.special import legendre
import seaborn as sns
import matplotlib
import array
import os
import csv
import cv2
from scipy.optimize import curve_fit
from scipy.special import legendre
from scipy import integrate
from scipy.signal import savgol_filter
#matplotlib.rcParams['figure.figsize'] = (5, 10)
from itertools import count
import random
from moepy import lowess, eda
from fitter import Fitter, get_common_distributions, get_distributions
from scipy.signal import find_peaks



#The code assumes that steady state has been extracted and that the input data has been stiched into eqaul size


def LabHWHM(stitched_path: str, n_intervals: int):
    """
    Load a stitched space–time image, split into CIE-Lab channels, 
    average each channel column-wise within equal-width intervals, and return
    (L_avg, a_avg, b_avg) arrays of shape (rows, n_intervals).

    Parameters
    ----------
    stitched_path : str
        Path to the stitched image (BGR or RGB file readable by OpenCV).
    n_intervals : int
        Number of equal-width intervals along the x-axis.

    Returns
    -------
    File_avg_L : np.ndarray
        Averaged L channel, shape (height, n_intervals).
    File_avg_a : np.ndarray
        Averaged a channel, shape (height, n_intervals).
    File_avg_b : np.ndarray
        Averaged b channel, shape (height, n_intervals).
    """
    # Read and ensure image is present
    img = cv2.imread(stitched_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image at: {stitched_path}")

    # Convert BGR -> Lab (OpenCV default read is BGR)
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(img_lab)

    # Cast to float for safe averaging
    L = L.astype(np.float64)
    a = a.astype(np.float64)
    b = b.astype(np.float64)

    height, width = L.shape
    if n_intervals <= 0:
        raise ValueError("n_intervals must be > 0")
    if n_intervals > width:
        raise ValueError("n_intervals cannot exceed image width in pixels")

    interval_size = width // n_intervals  # integer width per interval

    # Preallocate outputs
    File_avg_L = np.zeros((height, n_intervals), dtype=np.float64)
    File_avg_a = np.zeros((height, n_intervals), dtype=np.float64)
    File_avg_b = np.zeros((height, n_intervals), dtype=np.float64)

    print("File loaded and converted to Lab. Starting interval averaging...")

    for i in range(n_intervals):
        start_ = i * interval_size
        # Let the last interval absorb any remainder pixels
        stop_ = (i + 1) * interval_size if i < n_intervals - 1 else width

        # Correct slicing along x (columns)
        Temp_L = L[:, start_:stop_]
        Temp_a = a[:, start_:stop_]
        Temp_b = b[:, start_:stop_]

        # Column-wise mean to get a vertical profile for the interval
        File_avg_L[:, i] = np.mean(Temp_L, axis=1)
        File_avg_a[:, i] = np.mean(Temp_a, axis=1)
        File_avg_b[:, i] = np.mean(Temp_b, axis=1)

    print("Averaging complete, returning data for analysis.")
    return File_avg_L, File_avg_a, File_avg_b

def ExPLIDat(trim_, x_axis=None):
    """
    Fit a Gaussian to each column of the provided 2D array (m x n),
    returning the Half-Width at Half-Maximum (HWHM) and a Hermans-like orientation
    parameter (HUH ∈ [0,1]) derived from HWHM under a wrapped-Gaussian assumption.

    Parameters
    ----------
    trim_ : np.ndarray
        2D array with shape (m, n) to be fitted column-wise.
    x_axis : array-like or None, optional
        Optional 1D array of length m giving the x-positions corresponding to the rows of trim_.
        If None, a generic index axis 0..m-1 is used.

    Returns
    -------
    HWHM : np.ndarray
        Array of length n with the Gaussian HWHM for each column, in the same units as x_axis or index units.
    HUH : np.ndarray
        Array of length n with a Hermans-like orientation index (0–1), computed
        from HWHM assuming a wrapped-Gaussian azimuthal distribution over one
        full revolution. If the units of x_axis are unknown, the axis is normalized to 0..1 and treated as one revolution.
    details : dict
        Dictionary with additional outputs:
          - 'params': list of (A_fit, x0_fit, sigma_fit) per interval
          - 'fits': list of fitted curves yhat (np.ndarray of shape (m,)) per interval
          - 'r2': list of R² per interval
          - 'x': the x-axis used for fitting (np.ndarray of shape (m,))
          - 'FWHM': list/array of FWHM (= 2*HWHM) per interval (same units as x_axis)
          - 'HWHM': list/array of HWHM per interval (same units as x_axis)
          - 'AUC':  list/array of Gaussian area under the curve per interval (A * sqrt(2*pi) * sigma)
    """
    if trim_ is None:
        raise ValueError("trim_ cannot be None")
    if trim_.ndim != 2:
        raise ValueError("trim_ must be a 2D array of shape (m, n)")

    m, n = trim_.shape
    # Use provided x_axis or fall back to a generic linear index
    if x_axis is not None:
        x = np.asarray(x_axis, dtype=float).reshape(-1)
    else:
        x = np.arange(m, dtype=float)
    if x.size != m:
        # Rebuild a simple linear axis if mismatch
        x = np.linspace(0, m - 1, m)

    # Detect x-units over one full revolution (heuristics)
    xr = float(np.nanmax(x) - np.nanmin(x))
    if 5.5 <= xr <= 7.5:
        _x_units = "radians"
    elif 300.0 <= xr <= 400.0:
        _x_units = "degrees"
    elif 0.9 <= xr <= 1.1:
        _x_units = "unit"  # normalized 0..1 -> 0..2π
    else:
        _x_units = "unit"  # treat as normalized span for wrapped-Gaussian mapping

    # --- helpers for robust initialization and windowing ---
    def _dx(xarr: np.ndarray) -> float:
        xd = np.diff(xarr)
        return float(np.nanmedian(xd)) if xd.size else 1.0

    def _peak_window(xarr: np.ndarray, yarr: np.ndarray, j0: int, frac: float = 0.2, pad_pts: int = 2):
        """Return a slice (lo:hi) around the main peak where y >= frac*peak above baseline estimate."""
        yb = yarr - np.nanpercentile(yarr, 10.0)  # coarse baseline estimate
        yb = np.where(np.isfinite(yb), yb, 0.0)
        peak = float(np.nanmax(yb)) if np.isfinite(np.nanmax(yb)) else 0.0
        thr = frac * peak
        m = yb >= thr
        if not np.any(m):
            return 0, yarr.size
        # take the largest contiguous block containing j0
        lo = j0
        while lo > 0 and m[lo-1]:
            lo -= 1
        hi = j0
        n_ = yarr.size
        while hi < n_-1 and m[hi+1]:
            hi += 1
        lo = max(0, lo - pad_pts)
        hi = min(n_, hi + 1 + pad_pts)
        return int(lo), int(hi)

    # Gaussian model with constant baseline
    def Model(x, A, x0, sigma, C):
        # Gaussian with constant baseline
        return C + A * np.exp(-((x - x0) ** 2) / (2.0 * (sigma ** 2)))

    # Allocate outputs
    HWHM = np.full(n, np.nan, dtype=np.float64)
    HUH = np.full(n, np.nan, dtype=np.float64)
    FWHM = np.full(n, np.nan, dtype=np.float64)
    AUC = np.full(n, np.nan, dtype=np.float64)
    params = []
    fits = []
    r2_list = []

    sqrt_2ln2 = np.sqrt(2.0 * np.log(2.0))
    dx_est = _dx(x)

    for j in range(n):
        y = np.asarray(trim_[:, j], dtype=np.float64)

        # Initial guesses (with baseline) and adaptive window around peak
        j_max = int(np.nanargmax(y)) if y.size else 0
        C0 = float(np.nanpercentile(y, 10.0))
        A0_raw = float(np.nanmax(y) - C0)
        A0 = max(A0_raw, 1e-9)
        x0 = float(x[j_max]) if np.isfinite(x[j_max]) else float(np.nanmedian(x))
        span = float(np.nanmax(x) - np.nanmin(x)) if np.isfinite(np.nanmax(x) - np.nanmin(x)) else (dx_est * max(3, y.size))
        # estimate FWHM from threshold crossings at ~50% of peak above baseline
        y_above = y - C0
        half = 0.5 * (np.nanmax(y_above))
        left = j_max
        while left > 0 and y_above[left] > half:
            left -= 1
        right = j_max
        n_ = y.size
        while right < n_-1 and y_above[right] > half:
            right += 1
        fwhm_guess_pts = max(3, right - left)
        fwhm_guess = fwhm_guess_pts * dx_est
        sigma0 = max(fwhm_guess / (2.0 * sqrt_2ln2), dx_est * 0.5)
        # fit window
        lo, hi = _peak_window(x, y, j_max, frac=0.2, pad_pts=2)
        x_fit_local = x[lo:hi]
        y_fit_local = y[lo:hi]
        p0 = (A0, x0, sigma0, C0)

        # Ensure valid local window and bounds
        if not np.isfinite(x_fit_local).any() or not np.isfinite(y_fit_local).any() or x_fit_local.size < 3:
            # fallback: use full arrays
            x_fit_local = x
            y_fit_local = y
            # reset bounds using full arrays
            bounds = (
                (0.0, float(np.nanmin(x_fit_local)),  dx_est * 0.25,  float(np.nanmin(y_fit_local))),
                (np.inf, float(np.nanmax(x_fit_local)), span,           float(np.nanmax(y_fit_local)))
            )

        def _project_p0(p0_, lb_, ub_):
            p = list(p0_)
            for i in range(len(p)):
                lo = lb_[i]; hi = ub_[i]
                if not np.isfinite(p[i]):
                    p[i] = (lo + hi) * 0.5
                if p[i] <= lo:
                    p[i] = lo + 1e-9*(abs(hi-lo)+1.0)
                if p[i] >= hi:
                    p[i] = hi - 1e-9*(abs(hi-lo)+1.0)
            return tuple(p)

        lb = (0.0, float(np.nanmin(x_fit_local)),  dx_est * 0.25,  float(np.nanmin(y_fit_local)))
        ub = (np.inf, float(np.nanmax(x_fit_local)), span,           float(np.nanmax(y_fit_local)))
        bounds = (lb, ub)
        p0 = _project_p0(p0, lb, ub)

        try:
            popt, _ = curve_fit(
                Model, x_fit_local, y_fit_local,
                p0=p0, bounds=bounds, method="trf", maxfev=50000
            )
            A_fit, x0_fit, sigma_fit, C_fit = popt

            # HWHM = sqrt(2 ln 2) * sigma ; FWHM = 2 * HWHM
            hwhm = sqrt_2ln2 * abs(sigma_fit)
            HWHM[j] = hwhm
            # Hermans-like orientation parameter from HWHM
            def _hwhm_to_hermans(hwhm_val: float) -> float:
                """
                Convert HWHM (in detected x-units) to a Hermans-like orientation S in [0,1]
                using S = exp(-(HWHM_rad**2)/ln 2), where HWHM_rad is in radians.
                """
                if not np.isfinite(hwhm_val) or hwhm_val <= 0:
                    return np.nan
                if _x_units == "radians":
                    hwhm_rad = float(hwhm_val)
                elif _x_units == "degrees":
                    hwhm_rad = float(hwhm_val) * (np.pi / 180.0)
                elif _x_units == "unit":
                    # Normalize HWHM by the x-axis span and interpret 1.0 as a full revolution
                    span_local = float(np.nanmax(x) - np.nanmin(x))
                    if not np.isfinite(span_local) or span_local <= 0:
                        return np.nan
                    hwhm_norm = float(hwhm_val) / span_local
                    hwhm_rad = hwhm_norm * (2.0 * np.pi)
                else:
                    return np.nan
                S = np.exp(- (hwhm_rad ** 2) / np.log(2.0))
                # Clamp numerically to [0,1]
                if S < 0.0: S = 0.0
                if S > 1.0: S = 1.0
                return float(S)
            HUH[j] = _hwhm_to_hermans(hwhm)

            fwhm = 2.0 * hwhm
            FWHM[j] = fwhm
            AUC[j] = float(A_fit) * np.sqrt(2.0 * np.pi) * abs(sigma_fit)

            # Store params, fitted curve, and R²    
            yhat = Model(x, A_fit, x0_fit, sigma_fit, C_fit)
            ss_res = float(np.nansum((y - yhat) ** 2))
            ss_tot = float(np.nansum((y - np.nanmean(y)) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
            params.append((float(A_fit), float(x0_fit), float(sigma_fit), float(C_fit)))
            fits.append(yhat)
            r2_list.append(r2)
        except Exception as e:
            # Keep NaN for failed fits but continue; append placeholders
            print(f"Column {j}: Gaussian fit failed ({e}).")
            params.append((np.nan, np.nan, np.nan, np.nan))
            fits.append(np.full_like(x, np.nan, dtype=float))
            r2_list.append(np.nan)

    details = {
        "params": params,
        "fits": fits,
        "r2": r2_list,
        "x": x,
        "HWHM": HWHM,
        "FWHM": FWHM,
        "AUC": AUC,
    }
    return HWHM, HUH, details