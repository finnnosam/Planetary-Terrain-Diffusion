"""Import a rough, pixel-centred global PNG as spherical coarse conditioning."""
import hashlib
import io
from pathlib import Path
import numpy as np
from PIL import Image, UnidentifiedImageError
from . import cube
from .spherical_raster import GlobalRaster


class Draft(GlobalRaster):
    def __init__(self, path, ocean_depth=0., maximum=4000., refinement=.2):
        if not np.isfinite([ocean_depth, maximum]).all() or ocean_depth < 0 or maximum <= 0:
            raise ValueError("Draft ocean depth must be finite and >=0; white elevation must be finite and >0")
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
        # Black specifies water coverage, not a measured, flat seafloor at zero.
        # Automatic mode supplies a negative prior; learned bathymetry refines it.
        prior_depth = ocean_depth if ocean_depth > 0 else 2000.
        metres = np.where(gray == 0,-prior_depth,maximum*gray)
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

    def conditioning(self, seed, n, **options):
        guide = cube.conditioning(seed,n,**options)
        imported = self.on_cube(n)
        procedural = np.sign(guide[0])*guide[0]**2
        metres = imported[0]+(1-imported[1])*procedural
        guide[0] = np.sign(metres)*np.sqrt(np.abs(metres))
        return cube.identify(guide)

