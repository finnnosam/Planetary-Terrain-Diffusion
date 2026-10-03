"""Render saved continent passes and coarse checkpoints as one comparison PNG.

Usage: python tools/preview_layout.py RUN_OR_CHECKPOINT_DIR [--width 1024]
Run it after a coarse-only preview (or with --keep-checkpoints), before checkpoints are removed.
Writes layout-preview.png into the checkpoint directory. Elevation colours:
ocean blue (darker is deeper, to -5000 m), land green to white (at 4000 m).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from planet_diffusion import cube  # noqa: E402

FINAL_PANELS = (("continent-guide.npy", "Guide given to the coarse stage (last continent pass, smoothed)"),
                ("coarse-model.npy", "Coarse model output before pooling (model-scale layout)"),
                ("coarse.npy", "Coarse layout used by later stages (after pooling)"))


def panel_files(folder):
    passes = sorted(folder.glob("continent-pass-*.npy"), key=lambda p: int(p.stem.rsplit("-", 1)[1]))
    return ([(p.name, f"Continent pass on guide {p.stem.rsplit('-', 1)[1]}") for p in passes]
            + list(FINAL_PANELS))


def colour(metres):
    land = metres >= 0
    t = np.clip(metres/4000, 0, 1)[..., None]
    d = np.clip(-metres/5000, 0, 1)[..., None]
    land_rgb = (1-t)*np.array([70, 150, 70])+t*np.array([235, 230, 215])
    ocean_rgb = (1-d)*np.array([130, 180, 225])+d*np.array([15, 35, 95])
    return Image.fromarray(np.where(land[..., None], land_rgb, ocean_rgb).astype(np.uint8))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder")
    parser.add_argument("--width", type=int, default=1024)
    args = parser.parse_args(argv)
    folder = Path(args.folder)
    if (folder/"checkpoints").is_dir():
        folder = folder/"checkpoints"
    panels = []
    for name, label in panel_files(folder):
        path = folder/name
        if not path.exists():
            continue
        a = np.load(path, allow_pickle=False)
        erp = cube.to_equirectangular(a[:1], 512)[0][:-1]
        metres = np.sign(erp)*erp*erp
        land = 100*(metres > 0).mean()
        panels.append((colour(metres).resize((args.width, args.width//2), Image.BILINEAR),
                       f"{label}: land {land:.0f}%, {metres.min():.0f} to {metres.max():.0f} m"))
    if not panels:
        raise SystemExit(f"No layout checkpoints found in {folder}")
    out = Image.new("RGB", (args.width, len(panels)*(args.width//2+26)), "white")
    draw = ImageDraw.Draw(out)
    y = 0
    for image, label in panels:
        draw.text((8, y+7), label, fill="black")
        out.paste(image, (0, y+26))
        y += args.width//2+26
    target = folder/"layout-preview.png"
    out.save(target)
    print(target)


if __name__ == "__main__":
    main()
