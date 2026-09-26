"""Equal-geodesic-distance directional diagnostics (no rendering)."""
import numpy as np
from .topology import sample


def polar_anisotropy(elevation, latitudes=(0, 45, 60, 75, 80, 85, 88)):
    h = elevation.shape[-2]-1
    delta = np.pi/h
    lon = np.linspace(-np.pi, np.pi, 512, endpoint=False)
    def evaluate(p):
        y = h*(.5-np.arcsin(np.clip(p[..., 2], -1, 1))/np.pi)
        x = h*(np.arctan2(p[..., 1], p[..., 0])/np.pi+1)
        return sample(elevation, y, x)
    result = []
    for degrees in latitudes:
        for hemisphere in ([1] if degrees == 0 else [1, -1]):
            lat = np.deg2rad(degrees*hemisphere)
            q = np.stack([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon),
                          np.full_like(lon, np.sin(lat))], -1)
            east = np.stack([-np.sin(lon), np.cos(lon), np.zeros_like(lon)], -1)
            north = np.stack([-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon),
                              np.full_like(lon, np.cos(lat))], -1)
            rms = []
            for tangent in (east, north):
                d = evaluate(q*np.cos(delta)+tangent*np.sin(delta))-evaluate(q*np.cos(delta)-tangent*np.sin(delta))
                rms.append(float(np.sqrt(np.mean(d*d))))
            result.append({"latitude_degrees": degrees*hemisphere, "east_west_rms_m": rms[0],
                           "north_south_rms_m": rms[1], "direction_ratio": rms[0]/max(rms[1], 1e-12)})
    return {"angular_offset_degrees": float(np.rad2deg(delta)), "rings": result}


def cube_boundary_quality(elevation):
    """Exact alias agreement and geodesic edge gradients versus nearby interiors."""
    from . import cube
    n = elevation.shape[-1]-1
    members, ids, owners, _ = cube.edge_groups(n)
    flat = elevation.reshape(elevation.shape[0], -1)
    error = float(np.max(np.abs(flat[:,members]-flat[:,owners[ids]])))
    delta = 1/n
    rows = []
    for f in range(6):
        for g in range(f+1,6):
            if np.dot(cube.NORMAL[f],cube.NORMAL[g]) != 0:
                continue
            along = np.cross(cube.NORMAL[f],cube.NORMAL[g])
            q = cube.NORMAL[f]+cube.NORMAL[g]+np.linspace(-.9,.9,256)[:,None]*along
            q /= np.linalg.norm(q,axis=-1,keepdims=True)
            tangent = (cube.NORMAL[f]-cube.NORMAL[g])/np.sqrt(2)
            def at(angle):
                return cube.sample(elevation,q*np.cos(angle)+tangent*np.sin(angle))
            edge = at(delta)-at(-delta)
            nearby = np.concatenate([at(5*delta)-at(3*delta),at(-3*delta)-at(-5*delta)],axis=-1)
            edge_rms = float(np.sqrt(np.mean(edge**2)))
            nearby_rms = float(np.sqrt(np.mean(nearby**2)))
            rows.append({"faces":[f,g],"edge_rms_m":edge_rms,"nearby_rms_m":nearby_rms,
                         "ratio":edge_rms/max(nearby_rms,1e-12)})
    return {"shared_node_error_m":error,"edges":rows,
            "note":"Statistical gradient comparison, not a derivative-continuity proof"}
