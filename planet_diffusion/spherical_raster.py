"""Shared spherical sampling for pixel-centred global input maps."""
import numpy as np
from . import cube
from .topology import sample


class GlobalRaster:
    def evaluate(self, directions):
        p = directions/np.linalg.norm(directions,axis=-1,keepdims=True)
        h = self.nodes.shape[-2]-1
        y = h*(.5-np.arcsin(np.clip(p[...,2],-1,1))/np.pi)
        x = h*(np.arctan2(p[...,1],p[...,0])/np.pi+1)
        return sample(self.nodes,y,x)

    def on_cube(self, n):
        if n in self._samples:
            return self._samples[n]
        accum = np.zeros((self.nodes.shape[0],6,n+1,n+1),np.float64)
        norm = np.zeros((6,n+1,n+1),np.float64)
        offsets = (np.arange(8)+.5)/8-.5
        # Integrate a coarse-node footprint rather than picking isolated pixels.
        for f in range(6):
            for dy in offsets:
                for dx in offsets:
                    y = np.arange(n+1)[:,None]+dy
                    x = np.arange(n+1)[None,:]+dx
                    p = cube.directions(f,y,x,n,normalize=False)
                    weight = cube.area_weight(y,x,n)
                    accum[:,f] += self.evaluate(p)*weight
                    norm[f] += weight
        imported = cube.identify(accum/norm)
        self._samples[n] = imported
        return imported

    def on_cube_nodes(self, face, y, x, n):
        """Evaluate only requested coarse-node footprints on one cube chart."""
        y, x = np.broadcast_arrays(y, x)
        total = np.zeros((self.nodes.shape[0],) + y.shape, np.float64)
        weight_sum = np.zeros(y.shape, np.float64)
        offsets = (np.arange(8)+.5)/8-.5
        for dy in offsets:
            for dx in offsets:
                p = cube.directions(face,y+dy,x+dx,n,normalize=False)
                weight = cube.area_weight(y+dy,x+dx,n)
                total += self.evaluate(p)*weight
                weight_sum += weight
        return (total/weight_sum).astype(np.float32)

