"""Import a rough, pixel-centred global PNG as spherical coarse conditioning."""
import hashlib
import io
from pathlib import Path
import numpy as np
from PIL import Image, UnidentifiedImageError
from . import cube
from .spherical_raster import GlobalRaster


class Draft(GlobalRaster):
    def __init__(self, path, ocean_depth=0., maximum=4000., refinement=.2, *, black_metres=None, _legacy_mapping=False):
        if not np.isfinite(ocean_depth) or ocean_depth < 0:
            raise ValueError("Draft ocean depth must be finite and >=0")
        black = -(ocean_depth if ocean_depth > 0 else 2000.) if black_metres is None else float(black_metres)
        if not np.isfinite([black, maximum]).all() or maximum <= black:
            raise ValueError("Black and white elevations must be finite; white must be greater than black")
        if not np.isfinite(refinement) or not .01 <= refinement <= 4:
            raise ValueError("Draft refinement must be between 0.01 and 4")
        self.refinement = float(refinement)
        data = Path(path).read_bytes()
        self.png_bytes = data
        try:
            with Image.open(io.BytesIO(data)) as im:
                if im.format != "PNG":
                    raise ValueError("Draft must be a PNG image")
                w,h = im.size
                if w != 2*h or h < 2:
                    raise ValueError("Draft PNG must be 2:1 (for example 2048 x 1024), north at top")
                if getattr(im, "n_frames", 1) != 1:
                    raise ValueError("Animated PNG drafts are not supported")
                if im.mode.startswith("I"):
                    gray = np.asarray(im,dtype=np.float32)/65535
                    alpha = np.ones_like(gray)
                    bits = 16
                else:
                    gray = np.asarray(im.convert("L"),dtype=np.float32)/255
                    alpha = np.asarray(im.convert("RGBA").getchannel("A"),dtype=np.float32)/255
                    bits = 8
        except (UnidentifiedImageError, Image.DecompressionBombError) as exc:
            raise ValueError(f"Cannot read draft PNG: {exc}") from exc
        # Legacy depth arguments still select the black endpoint for new runs.
        prior_depth = ocean_depth if ocean_depth > 0 else 2000.
        metres = (np.where(gray == 0,-prior_depth,maximum*gray) if _legacy_mapping
                  else black+(maximum-black)*gray)
        # Premultiplied elevation prevents transparent RGB from affecting heights.
        nodes = np.empty((3,h+1,w),np.float32)
        for channel,pixels in enumerate((metres*alpha,alpha,(gray>0)*alpha)):
            nodes[channel,1:-1] = (pixels[:-1]+pixels[1:]+np.roll(pixels[:-1],1,-1)
                                   +np.roll(pixels[1:],1,-1))*.25
            nodes[channel,0] = pixels[0].mean(keepdims=True,dtype=np.float64)
            nodes[channel,-1] = pixels[-1].mean(keepdims=True,dtype=np.float64)
        self.nodes = nodes
        self.prior_depth = prior_depth
        self._samples = {}
        self.metadata = {"sha256":hashlib.sha256(data).hexdigest(),"width":w,"height":h,
                         "mapping":"black is ocean; nonblack brightness scales land elevation",
                         "ocean_depth_hint_metres":ocean_depth,"ocean_prior_metres":-prior_depth,
                         "black":"ocean mask; 0 depth hint means automatic bathymetry",
                         "white_metres":maximum,"source_bits":bits,
                         "layout":"upstream coarse conditioning; free learned detail and reconstruction",
                         "refinement":self.refinement,
                         "alpha":"transparent uses seeded procedural guide",
                         "sampling":"8x8 area-weighted cube-node footprint; periodic longitude; shared poles"}
        if not _legacy_mapping:
            self.metadata.update(mapping="linear elevation between black and white",
                                 mapping_version="linear-v1", black_metres=black)
            self.metadata.pop("black")
            self.metadata.pop("ocean_depth_hint_metres")
            self.metadata.pop("ocean_prior_metres")

    @classmethod
    def from_metadata(cls, path, info):
        """Retain the original mapping when querying older saved PNG runs."""
        if info.get("mapping_version") == "linear-v1":
            return cls(path,maximum=info["white_metres"],refinement=info["refinement"],
                       black_metres=info["black_metres"])
        return cls(path,info["ocean_depth_hint_metres"],info["white_metres"],
                   info["refinement"],_legacy_mapping=True)

    def conditioning(self, seed, n, **options):
        guide = cube.conditioning(seed,n,**options)
        imported = self.on_cube(n)
        procedural = np.sign(guide[0])*guide[0]**2
        metres = imported[0]+(1-imported[1])*procedural
        guide[0] = np.sign(metres)*np.sqrt(np.abs(metres))
        return cube.identify(guide)

    def conditioning_nodes(self, seed, face, y, x, n, **options):
        from . import procedural
        p = cube.directions(face,y,x,n)
        guide = procedural.encode(procedural.finalize(
            procedural.sample_raw(seed,p,2*n/np.pi,**options)))
        imported = self.on_cube_nodes(face,y,x,n)
        procedural_elev = np.sign(guide[0])*guide[0]**2
        metres = imported[0]+(1-imported[1])*procedural_elev
        guide[0] = np.sign(metres)*np.sqrt(np.abs(metres))
        return guide

