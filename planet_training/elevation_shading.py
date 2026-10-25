"""Fixed elevation colours and NW 5x hillshade, identical to scripts/diagnose_terrain_structure.

Duplicated here so new review scripts do not import that module, which locates
large raster files at import time.
"""
import matplotlib
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize

CMAP = LinearSegmentedColormap.from_list('fixed_elevation',
    [(0, '#163657'), (.42, '#377db0'), (.4999, '#9bcdd1'), (.5, '#50774c'),
     (.525, '#a1b67b'), (.58333, '#c9bc85'), (.66667, '#bd9770'), (.75, '#ad9587'),
     (.83333, '#dbd4cb'), (1, '#ffffff')], N=65536)
# N = 65536 colour levels (0.18 m each): matplotlib's default 256 levels are 47 m apart over +-6000 m, which drew
# contour-like bands on gentle plains that read as terraces or staircases in review pictures (found 23 Oct 2026)
NORM = Normalize(-6000, 6000)


def image(z, spacing, exaggeration=5):
    dy, dx = np.gradient(z, spacing)
    normal = np.stack((-dx*exaggeration, dy*exaggeration, np.ones_like(z)), axis=-1)
    normal /= np.linalg.norm(normal, axis=-1)[..., None]
    az, alt = np.radians([315, 45])
    light = np.array([np.sin(az)*np.cos(alt), np.cos(az)*np.cos(alt), np.sin(alt)])
    intensity = np.clip(normal@light, 0, 1)
    return np.clip(CMAP(NORM(z))[..., :3]*(.48+.65*intensity[..., None]), 0, 1)


def legend(fig, axs):
    sm = matplotlib.cm.ScalarMappable(norm=NORM, cmap=CMAP)
    cb = fig.colorbar(sm, ax=np.asarray(axs).ravel().tolist(), orientation='horizontal', fraction=.018, pad=.015, aspect=65)
    cb.set_ticks([-6000, -3000, 0, 1000, 2000, 4000, 6000])
    cb.set_label('Elevation (m); colour limits ±6000 m, values outside are colour-clipped only', fontsize=9)


def show(ax, z, title, spacing):
    half = z.shape[-1]*spacing/2000
    ax.imshow(image(z, spacing), extent=(-half, half, -half, half), interpolation='nearest')
    ax.set_title(title, fontsize=9); ax.set_xlabel('km'); ax.set_ylabel('km'); ax.tick_params(labelsize=7)
