"""Numerical artifact diagnostics; these are not geographic realism scores."""
import numpy as np


def grid_artifacts(elevation, period=8):
    a = np.asarray(elevation, dtype=np.float64)
    h, w = a.shape
    y0, y1 = h//8, h-h//8
    x0, x1 = w//8, w-w//8
    gx = np.sqrt(np.mean(np.diff(a[y0:y1], axis=1)**2, axis=1))
    gy = np.sqrt(np.mean(np.diff(a[:, x0:x1], axis=0)**2, axis=0))
    row_phase = [float(gx[(np.arange(y0, y1) % period) == k].mean()) for k in range(period)]
    col_phase = [float(gy[(np.arange(x0, x1) % period) == k].mean()) for k in range(period)]
    return {"period_pixels": period, "row_phase_east_west_gradient_rms": row_phase,
            "column_phase_north_south_gradient_rms": col_phase,
            "row_phase_ratio": max(row_phase)/max(min(row_phase), 1e-12),
            "column_phase_ratio": max(col_phase)/max(min(col_phase), 1e-12),
            "range_metres": [float(a.min()), float(a.max())]}
