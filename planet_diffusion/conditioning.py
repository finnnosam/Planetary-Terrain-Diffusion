"""Upstream's five TIFF conditioning channels adapted to a closed sphere."""
import hashlib
from pathlib import Path
import numpy as np
from rasterio.io import MemoryFile
from . import cube
from .spherical_raster import GlobalRaster


# Same names, order and physical units as upstream inference/tiff_export.py.
CHANNEL_FILES = ("heightmap.tif", "temperature.tif", "temperature_std.tif",
                 "precipitation.tif", "precipitation_cv.tif")
DEFAULT_SNR = (.2, .2, 1., .2, 1.)


def parse_snr(value):
    try:
        values = np.asarray(value.split(",") if isinstance(value, str) else value, dtype=np.float32)
    except (ValueError, TypeError) as exc:
        raise ValueError("Conditioning SNR requires five numbers between 0.01 and 4") from exc
    if values.shape != (5,) or not np.isfinite(values).all() or np.any((values < .01) | (values > 4)):
        raise ValueError("Conditioning SNR requires five numbers between 0.01 and 4")
    return values


class TiffConditioning:
    def __init__(self, folder, snr=DEFAULT_SNR):
        self.cond_snr = parse_snr(snr)
        self.sources = {}
        self.rasters = {}
        files = {}
        for channel, name in enumerate(CHANNEL_FILES):
            path = Path(folder)/name
            if not path.exists():
                continue
            data = path.read_bytes()
            # Read the exact bytes being fingerprinted; external masks are not inputs.
            with MemoryFile(data) as mem, mem.open() as ds:
                t = ds.transform
                if (ds.count != 1 or ds.height < 2 or ds.width != 2*ds.height
                        or ds.crs is None or not ds.crs.is_geographic
                        or not np.isclose(ds.crs.units_factor[1], np.pi/180)
                        or t.b != 0 or t.d != 0 or t.a <= 0 or t.e >= 0
                        or not np.allclose(tuple(ds.bounds), (-180,-90,180,90), rtol=0, atol=1e-6)):
                    raise ValueError(f"{name} must be a single-band, north-up 2:1 global geographic TIFF in degrees with bounds (-180, -90, 180, 90)")
                pixels = ds.read(1, masked=True).astype(np.float64)
                values = pixels.filled(np.nan)*ds.scales[0]+ds.offsets[0]
                valid = np.isfinite(values)
                if channel >= 2 and np.any(values[valid] < 0):
                    raise ValueError(f"{name} must contain nonnegative values (or nodata)")
                values = np.where(valid, values, 0.)
                if channel == 2:
                    values *= 100.  # Upstream temperature variability is degC x100.
                if np.any(np.abs(values) > np.finfo(np.float32).max/4):
                    raise ValueError(f"{name} contains values too large for conditioning")
                h, w = values.shape
                raster = GlobalRaster()
                raster.nodes = np.empty((2,h+1,w), np.float32)
                for i, a in enumerate((values, valid.astype(np.float64))):
                    raster.nodes[i,1:-1] = (a[:-1]+a[1:]+np.roll(a[:-1],1,-1)+np.roll(a[1:],1,-1))*.25
                    raster.nodes[i,0] = a[0].mean(dtype=np.float64)
                    raster.nodes[i,-1] = a[-1].mean(dtype=np.float64)
                raster._samples = {}
                self.rasters[channel] = raster
                files[name] = {"sha256":hashlib.sha256(data).hexdigest(), "width":w, "height":h}
                self.sources[name] = data
        if not files:
            raise ValueError("No conditioning TIFFs found; expected at least one of: "+", ".join(CHANNEL_FILES))
        self.metadata = {"files":files, "snr":self.cond_snr.tolist(),
                         "sampling":"8x8 area-weighted cube-node footprint; periodic longitude; shared poles",
                         "missing":"nodata and missing channels blend with seeded procedural guide"}

    def conditioning(self, seed, n, **options):
        # WorldPipeline imports merge physical raw fields, skip synthetic
        # climate finalization, and encode elevation only after blending.
        from .procedural import encode
        guide = cube.conditioning(seed,n,raw=True,**options)
        for channel, raster in self.rasters.items():
            values, coverage = raster.on_cube(n)
            procedural = guide[channel]
            blended = values+(1-np.clip(coverage,0,1))*procedural
            guide[channel] = blended
        return cube.identify(encode(guide))

    def snapshot(self, folder):
        folder = Path(folder)/"conditioning"
        folder.mkdir(parents=True,exist_ok=True)
        for name, data in self.sources.items():
            path = folder/name
            if not path.exists():
                temp = path.with_suffix(".tif.tmp")
                temp.write_bytes(data)
                temp.replace(path)
