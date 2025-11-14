from __future__ import annotations
import numpy as np
from numba import njit, prange
from typing import Tuple, Callable, Dict, Optional

from centrifugesim import constants

def read_lxcat_cross_section(
    path: str,
    *,
    prefer_columns: Tuple[int, int] = (0, 1),
) -> Tuple[np.ndarray, np.ndarray, Dict[str, str]]:
    """
    Read an LXCat ASCII cross-section file and return (E_J, sigma_m2, meta).

    Parameters
    ----------
    path : str
        Path to the LXCat file (plain text).
    prefer_columns : (int, int), optional
        Zero-based indices of columns to read as (energy, cross_section)
        if the file contains more than two numeric columns.

    Returns
    -------
    E_J : (N,) ndarray
        Electron energies in Joules (strict SI).
    sigma_m2 : (N,) ndarray
        Cross section in m^2 (strict SI).
    meta : dict
        Parsed metadata: may include 'energy_unit' and 'sigma_unit'.

    Notes
    -----
    - The parser skips blank lines and lines starting with typical comment
      markers: '#', '!', '%', ';', '//', or text in headers.
    - Units are guessed from header text if present (eV/J for energy,
      m^2/cm^2/Å^2 for cross section). If not found, defaults to E in eV,
      sigma in m^2 (which is the prevalent LXCat convention today).
    """
    # Read raw lines
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    # Extract header lines and data lines
    comment_prefixes = ("#", "!", "%", ";", "//", "E (", "e (", "Energy", "ENERGY")
    header = []
    data_lines = []
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if s.startswith(comment_prefixes) or not any(c.isdigit() for c in s):
            header.append(s)
        else:
            data_lines.append(s)

    # Guess units from header
    header_text = " ".join(header).lower()
    energy_unit = "eV"
    if " j)" in header_text or "joule" in header_text or " (j" in header_text:
        energy_unit = "J"
    sigma_unit = "m^2"
    if "cm^2" in header_text or " cm2" in header_text:
        sigma_unit = "cm^2"
    elif "å^2" in header_text or "angstrom" in header_text or "a^2" in header_text:
        sigma_unit = "A^2"

    # Load numeric table robustly
    # Try to split by whitespace for each line, collect floatable tokens
    numeric_rows = []
    for s in data_lines:
        parts = s.replace(",", " ").split()
        # keep only items that can be converted to float
        vals = []
        for p in parts:
            try:
                vals.append(float(p))
            except ValueError:
                pass
        if len(vals) >= 2:
            numeric_rows.append(vals)
    if len(numeric_rows) == 0:
        raise ValueError(f"No numeric data detected in '{path}'.")

    # Build array and select columns
    # Pad/truncate rows to the maximum consistent column count
    maxcols = max(len(r) for r in numeric_rows)
    trimmed = [r + [np.nan]*(maxcols - len(r)) for r in numeric_rows]
    arr = np.array(trimmed, dtype=float)
    e_col, s_col = prefer_columns
    if e_col >= arr.shape[1] or s_col >= arr.shape[1]:
        # fallback: take first two columns
        e_col, s_col = 0, 1
    E_raw = arr[:, e_col]
    sigma_raw = arr[:, s_col]

    # Clean non-finite
    m = np.isfinite(E_raw) & np.isfinite(sigma_raw)
    if not np.any(m):
        raise ValueError("All rows are non-finite after parsing.")
    E_raw = E_raw[m]
    sigma_raw = sigma_raw[m]

    # Sort by increasing energy and deduplicate
    order = np.argsort(E_raw)
    E_raw = E_raw[order]
    sigma_raw = sigma_raw[order]
    # Remove duplicate energies by averaging
    if np.any(np.diff(E_raw) == 0.0):
        uniq_E, idx_start = np.unique(E_raw, return_index=True)
        sigma_avg = np.empty_like(uniq_E)
        for i, e in enumerate(uniq_E):
            mask = (E_raw == e)
            sigma_avg[i] = np.nanmean(sigma_raw[mask])
        E_raw, sigma_raw = uniq_E, sigma_avg

    # Units -> SI
    if energy_unit.lower().startswith("j"):
        E_J = E_raw.astype(float)
    else:
        # assume eV
        E_J = E_raw.astype(float) * constants.q_e

    if sigma_unit == "m^2":
        sigma_m2 = sigma_raw.astype(float)
    elif sigma_unit == "cm^2":
        sigma_m2 = sigma_raw.astype(float) * 1.0e-4
    elif sigma_unit in ("A^2", "Å^2"):
        sigma_m2 = sigma_raw.astype(float) * 1.0e-20
    else:
        # Unknown; assume already SI
        sigma_m2 = sigma_raw.astype(float)

    # Drop negative/NaN values
    good = np.isfinite(sigma_m2) & (sigma_m2 >= 0.0) & np.isfinite(E_J) & (E_J >= 0.0)
    if not np.any(good):
        raise ValueError("No valid (E, sigma) pairs after cleaning.")
    E_J = E_J[good]
    sigma_m2 = sigma_m2[good]

    meta = {"energy_unit_detected": energy_unit, "sigma_unit_detected": sigma_unit, "source": path}
    return E_J, sigma_m2, meta


def _extend_energy_grid_powerlaw(
    E_J: np.ndarray,
    sigma_m2: np.ndarray,
    Emax_needed: float,
    n_tail: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    If the original grid does not reach high enough energy for the largest T_e,
    append a power-law tail (log-log extrapolation using last 6 positive points).
    """
    E_J = np.asarray(E_J)
    sigma_m2 = np.asarray(sigma_m2)

    if E_J[-1] >= Emax_needed:
        return E_J, sigma_m2

    # Use last points with strictly positive sigma to estimate slope
    mask_pos = sigma_m2 > 0.0
    Epos = E_J[mask_pos]
    spos = sigma_m2[mask_pos]
    if Epos.size >= 6:
        E_fit = Epos[-6:]
        s_fit = spos[-6:]
        # log-log slope
        slope, intercept = np.polyfit(np.log(E_fit), np.log(s_fit), 1)
        # Prevent positive slope at high E (unphysical for most e-impact processes)
        slope = min(slope, -0.01)
        E_tail = np.geomspace(E_J[-1]*1.05, Emax_needed, n_tail)
        sigma_tail = np.exp(intercept) * E_tail**slope
    else:
        # Fall back to exponential cutoff from last value
        E_tail = np.geomspace(E_J[-1]*1.05, Emax_needed, n_tail)
        sigma_tail = sigma_m2[-1] * np.exp(-(E_tail - E_J[-1]) / (0.25*(Emax_needed - E_J[-1])))

    E_ext = np.concatenate([E_J, E_tail])
    s_ext = np.concatenate([sigma_m2, sigma_tail])
    return E_ext, s_ext


def rate_coefficient_maxwellian(
    E_J: np.ndarray,
    sigma_m2: np.ndarray,
    Te_K: np.ndarray,
) -> np.ndarray:
    """
    Compute Maxwellian rate coefficient k(Te) = <sigma v> in SI units (m^3/s).

    Parameters
    ----------
    E_J : (N,) ndarray
        Energies in Joules (monotonic increasing).
    sigma_m2 : (N,) ndarray
        Cross section in m^2 (same length as E_J).
    Te_K : array_like
        Electron temperature(s) in Kelvin (any shape).

    Returns
    -------
    k_Te : ndarray
        Rate coefficient(s) in m^3/s with same shape as Te_K.

    Notes
    -----
    Uses the standard Maxwellian formula (E in J):
      k(T) = sqrt(8/(pi*m_e)) * (kB*T)^(-3/2) * ∫ sigma(E) * E * exp(-E/(kB*T)) dE
    The integral is computed by trapezoidal quadrature over the (possibly
    extended) energy grid.
    """
    E = np.asarray(E_J, dtype=float)
    s = np.asarray(sigma_m2, dtype=float)

    if E.ndim != 1 or s.ndim != 1 or E.size != s.size:
        raise ValueError("E_J and sigma_m2 must be 1-D arrays of equal length.")
    if np.any(np.diff(E) <= 0):
        raise ValueError("E_J must be strictly increasing.")

    Te = np.asarray(Te_K, dtype=float)
    Te_flat = Te.ravel()

    # Ensure the grid spans sufficiently beyond the tail for the largest Te
    Emax_needed = max(E[-1], 50.0 * constants.kb * float(np.max(Te_flat)))  # ~50 kT is plenty for the Boltzmann tail
    E_use, s_use = _extend_energy_grid_powerlaw(E, s, Emax_needed)

    # Build the integrand for all Te at once: shape (N_E, N_T)
    # integrand = sigma(E) * E * exp(-E/(kB*Te))
    Ecol = E_use[:, None]  # (N_E, 1)
    expo = np.exp(-Ecol / (constants.kb * Te_flat[None, :]))
    integrand = s_use[:, None] * Ecol * expo

    # Energy integral for each Te
    integ = np.trapz(integrand, x=E_use, axis=0)  # (N_T,)

    # Prefactor sqrt(8 / (pi*m_e)) * (kB*T)^(-3/2)
    pref = np.sqrt(8.0 / (np.pi * constants.m_e)) * (constants.kb * Te_flat) ** (-1.5)
    k_flat = pref * integ
    return k_flat.reshape(Te.shape)


def build_rate_table_and_interpolator(
    path: str,
    Te_min_K: float,
    Te_max_K: float,
    n_T: int = 200,
    *,
    prefer_columns: Tuple[int, int] = (0, 1),
    spacing: str = "log",
) -> Tuple[np.ndarray, np.ndarray, Callable[[np.ndarray], np.ndarray], Dict[str, str]]:
    """
    High-level helper:
    1) Read (E, sigma) from LXCat file (SI internally),
    2) Build a temperature grid (Kelvin),
    3) Compute k(Te) on that grid (m^3/s),
    4) Return (Te_grid, k_grid, interpolator, meta).

    Parameters
    ----------
    path : str
        LXCat file path.
    Te_min_K, Te_max_K : float
        Temperature bounds (Kelvin). Must satisfy 0 < Te_min_K < Te_max_K.
    n_T : int, optional
        Number of temperature points.
    prefer_columns : (int, int), optional
        Energy and cross-section column indices in the data file (0-based).
    spacing : {'log','linear'}, optional
        Temperature grid spacing.

    Returns
    -------
    Te_grid_K : (n_T,) ndarray
        Electron temperature grid in Kelvin.
    k_grid_m3s : (n_T,) ndarray
        Rate coefficient values (m^3/s) at Te_grid_K.
    k_interp : callable
        Function k_interp(Te_array_K) -> k_array_m3s that accepts any
        array shape (1D, 2D, ...) of temperatures in Kelvin and returns
        the corresponding rate coefficient(s) in m^3/s.
    meta : dict
        Metadata from the reader plus a couple of extras.

    Notes
    -----
    The interpolator uses 1-D linear interpolation over Te; inputs of any
    shape are flattened and reshaped back. Values outside [Te_min_K, Te_max_K]
    are extrapolated flat at the endpoints (NumPy behavior).
    """
    if not (np.isfinite(Te_min_K) and np.isfinite(Te_max_K) and Te_min_K > 0 and Te_max_K > Te_min_K):
        raise ValueError("Require 0 < Te_min_K < Te_max_K (finite).")

    E_J, sigma_m2, meta = read_lxcat_cross_section(path, prefer_columns=prefer_columns)

    if spacing == "log":
        Te_grid = np.geomspace(Te_min_K, Te_max_K, n_T)
    elif spacing == "linear":
        Te_grid = np.linspace(Te_min_K, Te_max_K, n_T)
    else:
        raise ValueError("spacing must be 'log' or 'linear'.")

    k_grid = rate_coefficient_maxwellian(E_J, sigma_m2, Te_grid)

    def _interp(Te_any: np.ndarray) -> np.ndarray:
        x = np.asarray(Te_any, dtype=float)
        y = np.interp(x.ravel(), Te_grid, k_grid)  # 1-D interp; flat extrapolation
        return y.reshape(x.shape)

    meta = dict(meta)
    meta.update({"Te_min_K": Te_min_K, "Te_max_K": Te_max_K, "n_T": n_T})
    return Te_grid, k_grid, _interp, meta


# ---- Optional convenience wrapper for Te in eV (still returns strict SI) ----
def kelvin_from_eV(Te_eV: np.ndarray) -> np.ndarray:
    """Convert electron temperature from eV to Kelvin (SI)."""
    return np.asarray(Te_eV, dtype=float) * constants.q_e / constants.kb


@njit(cache=True)
def _interp1_scalar(xp, fp, x):
    # xp must be strictly increasing
    n = xp.size
    if x <= xp[0]:       # clamp
        return fp[0]
    if x >= xp[n-1]:     # clamp
        return fp[n-1]
    k = np.searchsorted(xp, x)  # index of right bin edge
    x0 = xp[k-1]; x1 = xp[k]
    y0 = fp[k-1]; y1 = fp[k]
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)

@njit(parallel=True, cache=True)
def eval_rate_map_numba(mask, Te_grid, rate_grid, Te2d):
    out = np.zeros_like(Te2d).astype(np.float64)
    n0, n1 = Te2d.shape
    for i in prange(n0):
        for j in range(n1):
            if(mask[i, j]==1):
                out[i, j] = _interp1_scalar(Te_grid, rate_grid, Te2d[i, j])
    return out