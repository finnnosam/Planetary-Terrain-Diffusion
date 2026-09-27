import hashlib
import json
from pathlib import Path
import numpy as np
from .topology import sample


def save_state(path, elevation, metadata, climate=None):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    metadata = dict(metadata)
    if climate is not None:
        from .climate import VERSION, BANDS
        _validate_climate(climate,metadata)
        metadata["climate"] = {"version":VERSION, "bands":[{"name":name,"unit":unit} for name,unit in BANDS],
                               "sha256":hashlib.sha256(climate.tobytes()).hexdigest(),
                               "source":"generated coarse climate; local land regression; cube interpolation"}
    elif "climate" in metadata:
        raise ValueError("Climate metadata requires climate features when saving")
    path.mkdir(parents=True)
    np.save(path/"elevation.npy", elevation.astype(np.float32))
    if climate is not None:
        np.save(path/"climate.npy",climate,allow_pickle=False)
    metadata["elevation_sha256"] = hashlib.sha256(elevation.astype(np.float32).tobytes()).hexdigest()
    (path/"planet.json").write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")


def load_state(path):
    path = Path(path)
    metadata = json.loads((path/"planet.json").read_text(encoding="utf-8"))
    a = np.load(path/"elevation.npy", mmap_mode="r", allow_pickle=False)
    h = metadata["native_height"]
    shape = ((metadata["output_height"], metadata["output_width"])
             if metadata.get("coverage") == "region" else (h+1, 2*h))
    if a.shape != shape or a.dtype != np.float32:
        raise ValueError("State dimensions/dtype disagree with planet.json")
    if "climate" in metadata:
        load_climate(path,metadata)
    return a, metadata


def _validate_climate(climate, metadata):
    n = metadata.get("face_coarse_intervals",0)
    if (n < 2 or climate.shape != (5,6,n+1,n+1) or climate.dtype != np.float32
            or not np.isfinite(climate).all()):
        raise ValueError("Invalid climate feature dimensions, dtype or values")


def load_climate(path, metadata):
    from .climate import VERSION
    if "climate" not in metadata:
        raise ValueError("This state has no saved climate; generate a new cube state to export climate")
    if metadata["climate"].get("version") != VERSION:
        raise ValueError("Unsupported saved climate version")
    climate = np.load(Path(path)/"climate.npy",allow_pickle=False)
    _validate_climate(climate,metadata)
    if hashlib.sha256(climate.tobytes()).hexdigest() != metadata["climate"]["sha256"]:
        raise ValueError("Saved climate checksum mismatch")
    return climate


def _describe_bands(dst, climate):
    from .climate import BANDS
    bands = BANDS if climate is not None else (("Elevation above model sea level","m"),)
    for i,(name,unit) in enumerate(bands,1):
        dst.set_band_description(i,name)
        dst.set_band_unit(i,unit)


def export_tiff(path, a, metadata, height=None, window=None, climate=None):
    """Window = (x, y, width, height) in the selected global pixel grid.

    Longitude windows may cross +180; their affine coordinates then exceed 180.
    This keeps one continuous GeoTIFF rather than mislabelling its bounds.
    """
    if climate is not None:
        _validate_climate(climate,metadata)
    if metadata.get("coverage") == "region":
        if window is not None or height is not None and height != a.shape[0]:
            raise ValueError("Regional states export at their saved bounds and resolution; generate another region for a different grid")
        return export_region_tiff(path, a, metadata, climate=climate)
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.windows import Window
    h = a.shape[0]-1
    height = h if height is None else height
    if height < 2 or height > h:
        raise ValueError(f"Export height must be between 2 and native height {h}; no invented detail")
    x, y, width, rows = window or (0, 0, 2*height, height)
    if min(width, rows) < 1 or width > 2*height or y < 0 or y+rows > height:
        raise ValueError("Invalid tile window: latitude must stay on planet, width <= global width")
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    radius = float(metadata["radius_metres"])
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be positive and finite")
    # Custom sphere, NOT EPSG:4326 (which would incorrectly assert WGS84).
    crs = rasterio.crs.CRS.from_string(f"+proj=longlat +R={radius:.12g} +no_defs")
    spacing = 180/height
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", width=width, height=rows,
                       count=5 if climate is not None else 1, dtype="float32", crs=crs,
                       transform=from_origin(-180+x*spacing, 90-y*spacing, spacing, spacing),
                       compress="deflate", predictor=3, tiled=True, BIGTIFF="IF_SAFER",
                       nodata=float("nan")) as dst:
        _describe_bands(dst,climate)
        dst.update_tags(AREA_OR_POINT="Area", **{k: str(v) for k, v in metadata.items()},
                        export_global_height=str(height), window=json.dumps([x, y, width, rows]),
                        sampling="point evaluation at pixel centres; not area-averaged")
        if climate is not None:
            dst.update_tags(units="per-band",product="climate")
        for start in range(0, rows, 128):
            count = min(128, rows-start)
            yy = (y+np.arange(start, start+count)+.5)*h/height
            xx = (x+np.arange(width)+.5)*h/height
            values = sample(a, yy[:, None], xx[None, :]).astype(np.float32)
            if climate is None:
                dst.write(values, 1, window=Window(0, start, width, count))
            else:
                from .climate import evaluate
                climate_values = evaluate(climate,-180+xx[None,:]*180/h,90-yy[:,None]*180/h,values)
                dst.write(climate_values,window=Window(0,start,width,count))


def verify_state(a, metadata):
    """Check saved-state integrity and actual limits at identified sphere points."""
    if metadata.get("coverage") == "region":
        digest = hashlib.sha256(np.asarray(a).tobytes()).hexdigest()
        matches = digest == metadata["elevation_sha256"]
        finite = bool(np.isfinite(a).all())
        return {"passed":finite and matches, "sha256_matches":matches,
                "finite":finite, "coverage":"region", "bounds":metadata["bounds"],
                "note":"Regional integrity only; this state does not cover the whole globe"}
    h, w = a.shape[0]-1, a.shape[1]
    lon = np.linspace(0, w, 129)
    y = np.linspace(0, h, 129)
    eps = 1e-6
    seam = float(np.max(np.abs(sample(a, y, 0)-sample(a, y, w))))
    crossing = float(np.max(np.abs(sample(a, y, -eps)-sample(a, y, eps))))
    pole_spread = [float(np.ptp(sample(a, p, lon))) for p in (0, h)]
    cap_spread = [float(np.ptp(sample(a, p, lon))) for p in (eps, h-eps)]
    cap_far = [float(np.ptp(sample(a, p, lon))) for p in (eps*100, h-eps*100)]
    # A periodic interpolant can hide an anomalously steep seam. Compare the
    # seam-adjacent elevation edge with ordinary edges in a broad latitude belt.
    belt = np.asarray(a[max(1, h//8):h-h//8], dtype=np.float64)
    seam_rms = float(np.sqrt(np.mean((belt[:, 0]-belt[:, -1])**2)))
    interior_rms = float(np.sqrt(np.mean(np.diff(belt, axis=-1)**2)))
    edge_ratio = seam_rms/max(interior_rms, 1e-12)
    from .quality import grid_artifacts
    from .geometry_quality import polar_anisotropy
    artifacts = grid_artifacts(a)
    digest = hashlib.sha256(np.asarray(a).tobytes()).hexdigest()
    valid = bool(np.isfinite(a).all() and seam < 1e-8 and max(pole_spread) < 1e-8
                 and digest == metadata["elevation_sha256"]
                 and all(s <= f*.011+1e-5 for s, f in zip(cap_spread, cap_far)))
    return {"passed": valid, "sha256_matches": digest == metadata["elevation_sha256"],
            "dateline_identification_error_m": seam, "dateline_two_sided_gap_m": crossing,
            "pole_longitude_spread_m": pole_spread, "near_pole_spread_m": cap_spread,
            "near_pole_spread_at_100x_distance_m": cap_far,
            "seam_edge_rms_m": seam_rms, "ordinary_longitude_edge_rms_m": interior_rms,
            "seam_to_ordinary_edge_ratio": edge_ratio,
            "seam_statistical_outlier": edge_ratio > 5,
            "latent_grid_artifacts": artifacts,
            "polar_geometry": polar_anisotropy(a),
            "grid_artifact_warning": max(artifacts["row_phase_ratio"], artifacts["column_phase_ratio"]) > 1.15,
            "note": "Continuity checks, not a learned-terrain quality score or cross-device determinism guarantee"}


def export_region_tiff(path, a, metadata, climate=None):
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.windows import Window
    if climate is not None:
        _validate_climate(climate,metadata)
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    radius = float(metadata["radius_metres"])
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be positive and finite")
    path.parent.mkdir(parents=True, exist_ok=True)
    rows, width = a.shape
    crs = rasterio.crs.CRS.from_string(f"+proj=longlat +R={radius:.12g} +no_defs")
    with rasterio.open(path,"w",driver="GTiff",width=width,height=rows,count=5 if climate is not None else 1,
                       dtype="float32",crs=crs,transform=from_bounds(*metadata["bounds"],width,rows),
                       compress="deflate",predictor=3,tiled=True,BIGTIFF="IF_SAFER",nodata=float("nan")) as dst:
        _describe_bands(dst,climate)
        dst.update_tags(AREA_OR_POINT="Area", **{k:str(v) for k,v in metadata.items()},
                        sampling="point evaluation at pixel centres; not area-averaged")
        if climate is None:
            dst.write(a,1)
        else:
            from .climate import evaluate
            dst.update_tags(units="per-band",product="climate")
            west,south,east,north = metadata["bounds"]
            lon = west+(np.arange(width)+.5)*(east-west)/width
            for start in range(0,rows,128):
                end = min(start+128,rows)
                lat = north-(np.arange(start,end)+.5)*(north-south)/rows
                dst.write(evaluate(climate,lon[None,:],lat[:,None],a[start:end]),
                          window=Window(0,start,width,end-start))
