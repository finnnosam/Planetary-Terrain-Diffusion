"""Learned coarse-only before/after comparison; no decoder or latent changes.

Run from the project root: python tools/compare_coarse.py --output outputs/coarse-comparison
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("MPLCONFIGDIR",str(Path(__file__).resolve().parents[1]/"outputs/.matplotlib"))
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from planet_diffusion import cube
from planet_diffusion.backends import TerrainBackend, MODEL_REVISION
from planet_diffusion.coarse import sample_coarse, linear_weight_window
from planet_diffusion.draft import Draft
from planet_diffusion.geometry_quality import cube_boundary_quality


def old_loop(backend,seed,raw,steps,size=16,stride=8):
    channels = [0,2,3,4,5]
    guide = (raw-backend.means[channels,None,None,None])/backend.stds[channels,None,None,None]
    angles = np.arctan(backend.cond_snr)[:,None,None,None]
    n = raw.shape[-1]-1
    guide = (np.cos(angles)*guide+np.sin(angles)*cube.noise(seed,0,5,n)).astype(np.float32)
    state = cube.noise(seed,1,6,n)*backend.start_coarse(steps)
    for step in range(steps):
        def predict(a,f,y,x):
            cond = cube.read(guide,f,np.arange(y,y+size)[:,None],np.arange(x,x+size)[None,:],nearest=True)
            return backend.coarse_predict(a,cond,step)
        options = {} if size == 16 else dict(weight_window=linear_weight_window(size),shared_weights=True)
        pred = cube.consensus(state,predict,size=size,stride=stride,**options)
        state = cube.identify(backend.coarse_advance(pred,state,step))
    result = backend.coarse_finish(state)*backend.stds[:,None,None,None]+backend.means[:,None,None,None]
    result[1] = result[0]-result[1]
    return cube.identify(result)


def project(a):
    return cube.to_equirectangular(a[:1],256)[0][:-1]


def metrics(coarse,raw):
    # Compare like-for-like signed-sqrt coarse means, decoded to metres after
    # projection. This is a large-scale proxy, not final decoded terrain.
    z, target = project(coarse), project(raw)
    elevation, target = np.sign(z)*z*z, np.sign(target)*target*target
    h,w = z.shape
    weights = np.broadcast_to(np.sin(np.pi*np.arange(h)[:,None]/h),z.shape).copy()
    weights /= weights.sum()
    area = lambda mask: float(np.sum(weights*mask))
    actual, expected = elevation>0,target>0
    delta = elevation-target
    mean = float(np.sum(weights*elevation))
    boundary = cube_boundary_quality(coarse[:1])
    return {
        "land_fraction":area(actual), "guide_land_fraction":area(expected),
        "guide_land_iou":area(actual&expected)/max(area(actual|expected),1e-12),
        "guide_land_ocean_agreement":area(actual==expected),
        "guide_elevation_rmse_m":float(np.sqrt(np.sum(weights*delta**2))),
        "mean_elevation_m":mean,
        "elevation_std_m":float(np.sqrt(np.sum(weights*(elevation-mean)**2))),
        "range_m":[float(elevation.min()),float(elevation.max())],
        "shared_node_error_signed_sqrt":boundary["shared_node_error_m"],
        "mean_edge_gradient_ratio":float(np.mean([e["ratio"] for e in boundary["edges"]])),
        "max_edge_gradient_ratio":float(max(e["ratio"] for e in boundary["edges"]))},elevation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    parser.add_argument("--draft",default="human_input/Blobby_2.png")
    parser.add_argument("--faces",type=int,nargs="+",default=[8,32])
    parser.add_argument("--seeds",type=int,nargs="+",default=[42])
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    backend = TerrainBackend("upstream","models/terrain-diffusion-90m",MODEL_REVISION,"cuda")
    draft = Draft(args.draft)
    report = {"backend":backend.metadata,"steps":20,"draft":draft.metadata,
              "scope":"Pretrained coarse model only; no final terrain decode. Same guides and spherical noise for all variants.",
              "cases":[]}
    for n in args.faces:
        for seed in args.seeds:
            for mode in ("procedural","draft"):
                runner = copy.copy(backend)
                if mode == "draft":
                    runner.cond_snr = np.array([draft.refinement,.2,1,.2,1],np.float32)
                options = dict(frequency_mult=runner.frequency_mult,drop_water_pct=runner.drop_water_pct)
                raw = cube.conditioning(seed,n,**options) if mode == "procedural" else draft.conditioning(seed,n,**options)
                case = {"n":n,"seed":seed,"mode":mode,"variants":{}}
                maps = [("Guide",np.sign(project(raw))*project(raw)**2)]
                for variant in ("global16","global64","tiles64"):
                    name = f"{mode}-n{n}-seed{seed}-{variant}"
                    path = out/(name+".npz")
                    started = time.perf_counter()
                    print(name,flush=True)
                    if path.exists():
                        raise ValueError(f"Output already exists: {path}; use a fresh output directory")
                    if variant == "tiles64":
                        result = sample_coarse(runner,seed,raw,20,lambda s:print(s,flush=True))
                    else:
                        result = old_loop(runner,seed,raw,20,*( (16,8) if variant == "global16" else (64,48)))
                    elapsed = time.perf_counter()-started
                    np.savez_compressed(path,coarse=result,guide=raw)
                    values,elevation = metrics(result,raw)
                    values["seconds"] = elapsed
                    case["variants"][variant] = values
                    maps.append((variant,elevation))
                    print(json.dumps(values),flush=True)
                fig,axes = plt.subplots(4,1,figsize=(12,15),layout="constrained")
                for ax,(name,data) in zip(axes,maps):
                    im=ax.imshow(data,cmap="RdBu_r",vmin=-6000,vmax=6000,extent=(-180,180,-90,90))
                    ax.contour(data>0,levels=[.5],colors="black",linewidths=.3,extent=(-180,180,-90,90),origin="upper")
                    ax.set_title(name)
                fig.colorbar(im,ax=axes,label="Coarse elevation proxy (m)",shrink=.7)
                fig.suptitle(f"{mode}, face intervals {n}, seed {seed}; identical input and noise")
                fig.savefig(out/f"{mode}-n{n}-seed{seed}.png",dpi=120)
                plt.close(fig)
                report["cases"].append(case)
                (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    report["code_sha256"] = {name:hashlib.sha256(Path(name).read_bytes()).hexdigest()
                              for name in ("planet_diffusion/coarse.py","planet_diffusion/cube.py",__file__)}
    (out/"report.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__ == "__main__":
    main()
