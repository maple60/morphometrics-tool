"""
calculate_efd.py
----------------
Compute and normalize Elliptic Fourier Descriptors (EFDs) for 2D closed contours.

Implements the classical method of Kuhl & Giardina (1982) and a
"true normalization" procedure correcting for translation, rotation,
scaling, and reflection effects. Supports reconstruction and CSV export.

Author: Maple
License: BSD-3-Clause
"""

import numpy as np
import pandas as pd
from pathlib import Path
import copy


def close_contour(df_contour: pd.DataFrame) -> np.ndarray:
    """
    Ensure that a contour is closed by checking if the first and last points coincide.
    If not, append the first point to the end.

    Parameters
    ----------
    df_contour : pd.DataFrame
        DataFrame containing 'x' and 'y' columns representing contour coordinates.

    Returns
    -------
    np.ndarray
        Array of contour points (N,2), with the first point appended to the end if necessary.
    """  # noqa: E501

    xy = df_contour[["x", "y"]].to_numpy()
    if not np.allclose(xy[0], xy[-1]):
        # print("Closing contour by adding the first point to the end.")
        xy = np.vstack([xy, xy[0]])
    return xy


def arc_length_t(xy: np.ndarray) -> np.ndarray:
    """
    Compute the cumulative arc-length parameterization for a closed contour.

    Parameters
    ----------
    xy : np.ndarray
        Array of shape (N, 2) representing contour points.

    Returns
    -------
    t : np.ndarray
        Cumulative arc-length parameterization.
    T : float
        Total arc length of the contour.
    """  # noqa: E501

    d = np.sqrt(np.sum(np.diff(xy, axis=0) ** 2, axis=1))
    t = np.concatenate([[0.0], np.cumsum(d)])
    T = t[-1]
    return t, T


def efourier_xy(df_contour: pd.DataFrame, harmonics: int = 50) -> dict:
    """
    Compute the discrete Elliptic Fourier Descriptors (EFDs) following Kuhl & Giardina (1982).

    Parameters
    ----------
    df_contour : pd.DataFrame
        DataFrame containing 'x' and 'y' columns of a closed contour.
    harmonics : int, optional
        Number of harmonics to compute. Default is 50.

    Returns
    -------
    dict
        Dictionary containing EFD coefficients:
        {'A0', 'C0', 'an', 'bn', 'cn', 'dn', 'T'}.
    """
    xy = close_contour(df_contour)  # Ensure the contour is closed
    t, T = arc_length_t(xy)  # Arc-length parameterization

    x, y = xy[:, 0], xy[:, 1]
    dx = np.diff(x)
    dy = np.diff(y)
    dt = np.diff(t)

    # DC components
    A0 = np.sum((x[1:] + x[:-1]) * dt) / (2 * T)
    C0 = np.sum((y[1:] + y[:-1]) * dt) / (2 * T)

    n = np.arange(1, harmonics + 1)[:, None]  # Shape (harmonics, 1)
    k = 2.0 * n * np.pi / T  # Shape (harmonics, 1)
    t0 = t[:-1]
    t1 = t[1:]
    cos_diff = np.cos(k * t1) - np.cos(k * t0)
    sin_diff = np.sin(k * t1) - np.sin(k * t0)
    vx = dx / dt
    vy = dy / dt

    denom = 2.0 * (n.squeeze() ** 2) * (np.pi**2)
    pref = T / denom

    an = pref * (cos_diff @ vx)  # (H,)
    bn = pref * (sin_diff @ vx)
    cn = pref * (cos_diff @ vy)
    dn = pref * (sin_diff @ vy)

    ef = {"A0": A0, "C0": C0, "an": an, "bn": bn, "cn": cn, "dn": dn, "T": T}

    return ef


def set_DC_components_to_zero(ef: dict) -> dict:
    """
    Compute the discrete Elliptic Fourier Descriptors (EFDs) following Kuhl & Giardina (1982).

    Parameters
    ----------
    df_contour : pd.DataFrame
        DataFrame containing 'x' and 'y' columns of a closed contour.
    harmonics : int, optional
        Number of harmonics to compute. Default is 50.

    Returns
    -------
    dict
        Dictionary containing EFD coefficients:
        {'A0', 'C0', 'an', 'bn', 'cn', 'dn', 'T'}.
    """
    ef_norm = copy.deepcopy(ef)
    ef_norm["A0"] = 0.0
    ef_norm["C0"] = 0.0
    return ef_norm


def adjust_direction_1st_order(ef: dict) -> dict:
    """
    Ensure that the first-order ellipse of the EFD is counterclockwise.

    Parameters
    ----------
    ef : dict
        Dictionary containing EFD coefficients.

    Returns
    -------
    dict
        Dictionary with adjusted orientation (if necessary).
    """
    a1, b1 = ef["an"][0], ef["bn"][0]
    c1, d1 = ef["cn"][0], ef["dn"][0]
    cross_production = a1 * d1 - b1 * c1
    if cross_production < 0:
        print("The first-order ellipse is clockwise. Adjusting to counterclockwise.")
        ef["bn"] = -ef["bn"]
        ef["dn"] = -ef["dn"]
        cross_production = -cross_production  # Update to positive
    else:
        print("The first-order ellipse is already counterclockwise.")
    return ef


def true_efd_normalization(ef: dict, skip_rotation: bool = True) -> dict:
    """
    Perform true normalization of EFDs correcting for translation, rotation,
    scaling, and reflection, following the method of Wu et al. (2024).

    Parameters
    ----------
    ef : dict
        Dictionary containing EFD coefficients.
    skip_rotation : bool, optional
        If True, skip rotation normalization. Default is True.

    Returns
    -------
    dict
        Normalized EFD coefficients.
    """
    # --- Set DC components to zero ---
    ef = set_DC_components_to_zero(ef)

    # --- Adjust the direction of the first-order ellipse ---
    ef = adjust_direction_1st_order(ef)

    # --- Extract first harmonic coefficients ---
    a1, b1 = ef["an"][0], ef["bn"][0]
    c1, d1 = ef["cn"][0], ef["dn"][0]

    a, b = ef["an"], ef["bn"]
    c, d = ef["cn"], ef["dn"]

    EPS = 1e-10
    n = len(a)

    # --- Calculate the rotation angle theta1 ---
    if skip_rotation:
        theta1 = 0.0
    else:
        # --- Calculate the scaling factor ---
        tan_theta2 = 2 * (a1 * b1 + c1 * d1) / (a1**2 + c1**2 - b1**2 - d1**2)
        theta1 = 0.5 * np.arctan(tan_theta2)
        # Ensure theta1 is in [0, pi/2]
        if theta1 < 0:
            theta1 += np.pi / 2
        sin_2theta = np.sin(2 * theta1)
        cos_2theta = np.cos(2 * theta1)
        cos_theta_square = (1 + cos_2theta) / 2
        sin_theta_square = (1 - cos_2theta) / 2

        # length of axes
        axis_theta1 = (
            (a1**2 + c1**2) * cos_theta_square
            + (a1 * b1 + c1 * d1) * sin_2theta
            + (b1**2 + d1**2) * sin_theta_square
        ) ** 0.5
        axis_theta2 = (
            (a1**2 + c1**2) * sin_theta_square
            - (a1 * b1 + c1 * d1) * sin_2theta
            + (b1**2 + d1**2) * cos_theta_square
        ) ** 0.5

        if axis_theta1 < axis_theta2:
            theta1 += np.pi / 2

    costh1 = np.cos(theta1)
    sinth1 = np.sin(theta1)
    a_star_1 = a1 * costh1 + b1 * sinth1
    c_star_1 = c1 * costh1 + d1 * sinth1
    psi1 = np.arctan(np.abs(c_star_1 / a_star_1))

    if c_star_1 > 0 > a_star_1:
        psi1 = np.pi - psi1
    if c_star_1 < 0 and a_star_1 < 0:
        psi1 = np.pi + psi1
    if c_star_1 < 0 < a_star_1:
        psi1 = 2 * np.pi - psi1

    # length of the major axis
    E = np.sqrt(a_star_1**2 + c_star_1**2)

    # Scaling
    a /= E
    b /= E
    c /= E
    d /= E

    if not skip_rotation:
        cospsi1 = np.cos(psi1)
        sinpsi1 = np.sin(psi1)
        normalized_all = np.zeros((n, 4))

        for i in range(n):
            normalized = np.dot(
                [[cospsi1, sinpsi1], [-sinpsi1, cospsi1]],
                [[a[i], b[i]], [c[i], d[i]]],
            )
            normalized_all[i] = normalized.reshape(1, 4)

        a = normalized_all[:, 0]
        b = normalized_all[:, 1]
        c = normalized_all[:, 2]
        d = normalized_all[:, 3]

        normalized_all_1 = np.zeros((n, 4))
        for i in range(n):
            normalized_1 = np.dot(
                [[a[i], b[i]], [c[i], d[i]]],
                [
                    [np.cos(theta1 * (i + 1)), -np.sin(theta1 * (i + 1))],
                    [np.sin(theta1 * (i + 1)), np.cos(theta1 * (i + 1))],
                ],
            )
            normalized_all_1[i, :] = normalized_1.reshape(1, 4)
        a = normalized_all_1[:, 0]
        b = normalized_all_1[:, 1]
        c = normalized_all_1[:, 2]
        d = normalized_all_1[:, 3]

    # Reflection symmetry adjustment (x-axis)
    if n > 1:
        if c[1] < -EPS:
            b[1:] *= -1
            c[1:] *= -1

    ef["an"] = a
    ef["bn"] = b
    ef["cn"] = c
    ef["dn"] = d

    return ef


def ef_to_dataframe(ef: dict) -> pd.DataFrame:
    """
    Convert an EFD dictionary into a pandas DataFrame.

    Parameters
    ----------
    ef : dict
        Dictionary containing EFD coefficients.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['n', 'A0', 'C0', 'an', 'bn', 'cn', 'dn'].
    """
    n_harm = len(ef["an"])
    df = pd.DataFrame(
        {
            "n": np.arange(1, n_harm + 1),
            "A0": ef["A0"],
            "C0": ef["C0"],
            "an": ef["an"],
            "bn": ef["bn"],
            "cn": ef["cn"],
            "dn": ef["dn"],
        }
    )
    return df


def reconstruct_efd(ef: dict, num: int = 400) -> np.ndarray:
    """
    Reconstruct a contour from its elliptic Fourier descriptors.

    Parameters
    ----------
    ef : dict
        Dictionary containing EFD coefficients.
    num : int, optional
        Number of reconstructed points. Default is 400.

    Returns
    -------
    np.ndarray
        Reconstructed contour coordinates of shape (num, 2).
    """
    A0, C0 = ef["A0"], ef["C0"]
    a, b, c, d = ef["an"], ef["bn"], ef["cn"], ef["dn"]
    T = ef["T"]

    H = len(a)  # Number of harmonics
    t = np.linspace(0, T, num=num, endpoint=False)  # Parameter t
    x = np.full_like(t, A0, dtype=float)
    y = np.full_like(t, C0, dtype=float)
    for n in range(1, H + 1):
        x += a[n - 1] * np.cos(2 * np.pi * n * t / T) + b[n - 1] * np.sin(
            2 * np.pi * n * t / T
        )
        y += c[n - 1] * np.cos(2 * np.pi * n * t / T) + d[n - 1] * np.sin(
            2 * np.pi * n * t / T
        )
    return np.column_stack([x, y])


def calculate_efd_and_save(payload) -> None:
    """
    Calculate EFDs from a contour and save the coefficients to CSV files.

    Parameters
    ----------
    payload : dict
        Dictionary containing:
        - 'df_contour': pandas DataFrame with contour coordinates ('x', 'y')
        - 'metadata': optional metadata dictionary
    """
    if payload is None:
        return

    df_contour = payload["df_contour"]
    metadata = payload["metadata"] or {}

    ef = efourier_xy(df_contour, harmonics=35)
    ef_normalized = true_efd_normalization(ef)

    # --- Save results ---
    df_ef = ef_to_dataframe(ef)
    df_ef_normalized = ef_to_dataframe(ef_normalized)
    src = metadata.get("source")
    id = src.get("image_id") or "unknown"
    leaf_id = src.get("roi_index") or "unknown"
    try:
        # Int or digit string -> zero-pad 2 digits
        leaf_id = f"{int(leaf_id):02d}"
    except (TypeError, ValueError):
        # fallback to plain string
        leaf_id = str(leaf_id)

    output_dir_ef = Path("output/coefficients_efd")
    output_dir_ef_normalized = Path("output/coefficients_efd_normalized")
    output_dir_ef.mkdir(parents=True, exist_ok=True)
    output_dir_ef_normalized.mkdir(parents=True, exist_ok=True)
    ef_file = output_dir_ef / f"{id}_{leaf_id}.csv"
    ef_normalized_file = output_dir_ef_normalized / f"{id}_{leaf_id}.csv"
    df_ef.to_csv(ef_file, index=False)
    df_ef_normalized.to_csv(ef_normalized_file, index=False)
    print(f"EFD coefficients saved to {ef_file}")
