"""Generate, resume, and numerically inspect a pretrained cubed-sphere planet."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import rasterio
from planet_diffusion.backends import TerrainBackend, MODEL_REVISION
from planet_diffusion.cube_generate import generate_cube
from planet_diffusion.storage import save_state,load_state,export_tiff,verify_state
from planet_diffusion.geometry_quality import polar_anisotropy, cube_boundary_quality


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output",required=True)
    p.add_argument("--face-coarse",type=int,default=4)
    p.add_argument("--seed",type=int,default=42)
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    if (out/"state").exists():
        a,meta = load_state(out/"state")
        if meta["seed"] != args.seed or meta["face_coarse_intervals"] != args.face_coarse:
            raise ValueError("Saved state differs from the requested cube run")
    else:
        torch.set_num_threads(4)
        backend = TerrainBackend("upstream","models/terrain-diffusion-90m",MODEL_REVISION)
        a,meta = generate_cube(backend,args.seed,args.face_coarse,20,
                              progress=lambda s:print(s,flush=True),checkpoint_dir=out/"checkpoints")
        meta["radius_metres"] = 6371000.
        save_state(out/"state",a,meta)
        a,meta = load_state(out/"state")
    if not (out/"planet-native.tif").exists():
        export_tiff(out/"planet-native.tif",a,meta)
    h = meta["native_height"]
    x,y,width,rows = 2*h-32,h//2-64,64,128
    tile_path = out/"dateline-detail.tif"
    if not tile_path.exists():
        export_tiff(tile_path,a,meta,window=(x,y,width,rows))
    with rasterio.open(out/"planet-native.tif") as full, rasterio.open(tile_path) as tile:
        values = full.read(1)
        expected = np.concatenate([values[y:y+rows,x:],values[y:y+rows,:32]],axis=1)
        np.testing.assert_array_equal(tile.read(1),expected)
        assert full.units == ("m",) and full.dtypes == ("float32",)
        assert full.scales == (1.,) and full.offsets == (0.,)
        assert full.width == 2*h and full.height == h
        assert tuple(full.bounds) == (-180.,-90.,180.,90.) and np.isfinite(values).all()
        assert full.tags()["algorithm"] == "cubed-sphere-v3"
        assert np.isclose(full.crs.to_dict()["R"],meta["radius_metres"])
        geotiff = {"size":[full.width,full.height],"units":"m","dtype":"float32",
                   "bounds":list(full.bounds),"radius_metres":meta["radius_metres"],
                   "finite":True,"dateline_tile_matches_global_bitwise":True}
    cube_elevation = np.load(out/"checkpoints"/"cube-elevation.npy",mmap_mode="r")
    report = {"boundaries":verify_state(a,meta),"polar_geometry":polar_anisotropy(a),
              "cube_boundaries":cube_boundary_quality(cube_elevation),"geotiff":geotiff}
    (out/"verification.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
    if not report["boundaries"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
