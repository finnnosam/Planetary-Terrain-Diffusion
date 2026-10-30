"""Compare generated land/ocean layout with a PNG; no rendering or colour maps."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import rasterio
from planet_diffusion.storage import load_state, verify_state, export_tiff
from planet_diffusion.geometry_quality import cube_boundary_quality


def compare(values, draft, white_metres):
    h,w = values.shape
    with Image.open(draft) as im:
        gray = np.asarray(im.convert("L"))
        coverage = Image.fromarray((gray>0).astype(np.uint8)*255).resize((w,h),Image.Resampling.BOX)
        expected = np.asarray(coverage) >= 128
        heights = np.asarray(im.convert("L").resize((w,h),Image.Resampling.BOX),dtype=np.float32)*white_metres/255
    actual = values > 0
    weights = np.cos(np.pi*((np.arange(h)+.5)/h-.5))[:,None]
    def area(mask):
        return float((mask*weights).sum())
    ocean = values[~expected]
    land = values[expected]
    valid_correlation = bool(land.size>1 and land.std()>0 and heights[expected].std()>0)
    return {"land_intersection_over_union":area(actual & expected)/max(area(actual | expected),1e-12),
            "land_ocean_agreement":area(actual==expected)/(weights.sum()*w),
            "painted_land_remaining_land":area(actual & expected)/max(area(expected),1e-12),
            "painted_ocean_below_sea_level":area((values<0) & ~expected)/max(area(~expected),1e-12),
            "land_height_correlation":float(np.corrcoef(land,heights[expected])[0,1]) if valid_correlation else None,
            "range_metres":[float(values.min()),float(values.max())],
            "painted_ocean_range_metres":[float(ocean.min()),float(ocean.max())] if ocean.size else None,
            "painted_ocean_std_m":float(ocean.std()) if ocean.size else None,
            "finite":bool(np.isfinite(values).all()),
            "note":"Mask scores are sphere-area weighted; coast pixels use majority coverage. Correlation is within painted land."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",required=True)
    parser.add_argument("--draft",required=True)
    parser.add_argument("--white-metres",type=float,default=6250.)
    parser.add_argument("--tiff",help="Override the run's planet-native.tif filename")
    args = parser.parse_args()
    run = Path(args.run)
    a,meta = load_state(run/"state")
    if hashlib.sha256(Path(args.draft).read_bytes()).hexdigest() != meta["draft"]["sha256"]:
        raise ValueError("Comparison PNG differs from the actual generation input")
    if args.white_metres != meta["draft"]["white_metres"]:
        raise ValueError("White elevation differs from generation metadata")
    with rasterio.open(args.tiff or run/"planet-native.tif") as src:
        assert tuple(src.bounds) == (-180.,-90.,180.,90.)
        assert src.units == ("m",) and src.width == 2*src.height
        values = src.read(1)
        report = {"seed":meta["seed"],"draft_sha256_matches":True,
                  "layout":compare(values,args.draft,args.white_metres),
                  "boundaries":verify_state(a,meta)}
    cube_path = run/"checkpoints"/"cube-elevation.npy"
    if cube_path.exists():
        report["cube_boundaries"] = cube_boundary_quality(np.load(cube_path,mmap_mode="r"))
    h = values.shape[0]
    tile_path = run/"dateline-layout-check.tif"
    if not tile_path.exists():
        export_tiff(tile_path,a,meta,height=h,window=(2*h-16,h//2,32,32))
    with rasterio.open(tile_path) as tile:
        expected = np.concatenate([values[h//2:h//2+32,-16:],values[h//2:h//2+32,:16]],axis=1)
        report["dateline_tile_matches_global"] = bool(np.array_equal(tile.read(1),expected))
    (run/"draft-layout-verification.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
    if not report["boundaries"]["passed"] or not report["dateline_tile_matches_global"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
