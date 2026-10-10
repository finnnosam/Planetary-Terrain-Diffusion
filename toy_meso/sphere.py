"""Patch geometry on the sphere and area-averaged sampling of a global raster."""
import numpy as np

EARTH_RADIUS_M = 6371000.0


class GlobalRaster:
    """North-up, pixel-centred 2:1 global lon/lat raster with bilinear lookup.

    Longitude wraps; latitude clamps at the outermost pixel centres.
    """
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32)
        self.h, self.w = self.values.shape
        if self.w != 2 * self.h:
            raise ValueError("Expected a 2:1 global raster")

    @classmethod
    def from_geotiff(cls, path):
        import rasterio
        with rasterio.open(path) as ds:
            a = ds.read(1).astype(np.float32)
            if ds.nodata is not None:
                bad = a == ds.nodata
                if bad.any():
                    raise ValueError(f"{path} contains nodata cells; fill them first")
        return cls(a)

    @classmethod
    def from_png(cls, path, black_metres=-2000.0, white_metres=4000.0, ocean_metres=None):
        """2:1 north-up grayscale PNG sketch: brightness maps linearly black -> white metres.

        Same mapping as planet_diffusion --draft (8- or 16-bit; colour is converted to gray).
        ocean_metres, if given, overrides pure-black pixels (a black = ocean mask sketch).
        """
        from PIL import Image
        if not white_metres > black_metres:
            raise ValueError("White elevation must be greater than black elevation")
        with Image.open(path) as im:
            if im.mode.startswith("I"):
                gray = np.asarray(im, dtype=np.float32) / 65535
            else:
                gray = np.asarray(im.convert("L"), dtype=np.float32) / 255
        metres = black_metres + (white_metres - black_metres) * gray
        if ocean_metres is not None:
            metres = np.where(gray == 0, ocean_metres, metres)
        return cls(metres)

    @classmethod
    def from_file(cls, path, black_metres=-2000.0, white_metres=4000.0, ocean_metres=None):
        if str(path).lower().endswith(".png"):
            return cls.from_png(path, black_metres, white_metres, ocean_metres)
        return cls.from_geotiff(path)

    def sample(self, lat, lon):
        """Bilinear value at latitude/longitude in degrees (any shape)."""
        x = (np.asarray(lon) + 180.0) / 360.0 * self.w - 0.5
        y = (90.0 - np.asarray(lat)) / 180.0 * self.h - 0.5
        y = np.clip(y, 0, self.h - 1)
        x0 = np.floor(x).astype(np.int64)
        y0 = np.minimum(np.floor(y).astype(np.int64), self.h - 2)
        fx, fy = x - x0, y - y0
        x0 %= self.w
        x1 = (x0 + 1) % self.w
        v = self.values
        top = v[y0, x0] * (1 - fx) + v[y0, x1] * fx
        bottom = v[y0 + 1, x0] * (1 - fx) + v[y0 + 1, x1] * fx
        return (top * (1 - fy) + bottom * fy).astype(np.float32)


def random_frames(rng, count):
    """Area-uniform random centres with a random rotation about each centre.

    Returns (centre, east-ish axis, north-ish axis), each (count, 3).
    """
    c = rng.normal(size=(count, 3))
    c /= np.linalg.norm(c, axis=1, keepdims=True)
    helper = np.where(np.abs(c[:, 2:3]) < 0.9, [[0, 0, 1.0]], [[1.0, 0, 0]])
    e1 = np.cross(helper, c)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(c, e1)
    angle = rng.uniform(0, 2 * np.pi, size=(count, 1))
    a = np.cos(angle) * e1 + np.sin(angle) * e2
    b = np.cos(angle) * e2 - np.sin(angle) * e1
    return c, a, b


def to_latlon(p):
    lat = np.degrees(np.arcsin(np.clip(p[..., 2], -1, 1)))
    lon = np.degrees(np.arctan2(p[..., 1], p[..., 0]))
    return lat, lon


def patch_points(frame, size, spacing_m, supersample=1, radius_m=EARTH_RADIUS_M):
    """Unit vectors for a size x size gnomonic grid (optionally supersampled).

    Output shape (size*supersample, size*supersample, 3). Cell (i, j) covers
    the supersample x supersample block of points starting at
    (i*supersample, j*supersample), so a block mean is an area average.
    """
    c, a, b = frame
    n = size * supersample
    step = spacing_m / supersample / radius_m
    t = (np.arange(n) - (n - 1) / 2) * step
    u, v = np.meshgrid(t, -t)  # row 0 is "north" of the frame
    p = c + u[..., None] * a + v[..., None] * b
    return p / np.linalg.norm(p, axis=-1, keepdims=True)


def block_mean(a, k):
    """Mean over non-overlapping k x k blocks of the last two axes."""
    *lead, h, w = a.shape
    return a.reshape(*lead, h // k, k, w // k, k).mean(axis=(-3, -1))


def area_average_patch(raster, frame, size, spacing_m, supersample=3):
    p = patch_points(frame, size, spacing_m, supersample)
    lat, lon = to_latlon(p)
    return block_mean(raster.sample(lat, lon), supersample), lat, lon


def encode(metres):
    """Upstream's signed-square-root elevation encoding."""
    return np.sign(metres) * np.sqrt(np.abs(metres))


def decode(z):
    return np.sign(z) * z * z


def upsample_cells(cells, factor):
    """Bilinear, cell-centre aligned upsampling of (..., h, w) by an integer factor.

    Edge cells are extrapolated by clamping, so callers should crop a margin.
    """
    *lead, h, w = cells.shape
    def coords(n):
        x = (np.arange(n * factor) + 0.5) / factor - 0.5
        x = np.clip(x, 0, n - 1)
        i0 = np.minimum(np.floor(x).astype(int), n - 2) if n > 1 else np.zeros_like(x, int)
        return i0, x - i0
    y0, fy = coords(h)
    x0, fx = coords(w)
    c = cells
    top = c[..., y0, :][..., :, x0] * (1 - fx) + c[..., y0, :][..., :, x0 + 1] * fx
    bot = c[..., y0 + 1, :][..., :, x0] * (1 - fx) + c[..., y0 + 1, :][..., :, x0 + 1] * fx
    return top * (1 - fy[:, None]) + bot * fy[:, None]
