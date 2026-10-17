"""Whole-planet measurements on unique physical cube nodes, including polar faces."""
from functools import lru_cache
import math
import numpy as np
from planet_diffusion import cube
from .data import node_weights


@lru_cache(maxsize=8)
def geometry(n):
    side=n+1
    owner=np.arange(6*side*side)
    members,ids,owners,_=cube.edge_groups(n)
    owner[members]=owners[ids]
    unique,groups=np.unique(owner,return_inverse=True)
    weights=np.bincount(groups,weights=node_weights(n).ravel())
    weights/=weights.sum()
    directions=np.stack([cube.directions(f,np.arange(side)[:,None],np.arange(side)[None],n) for f in range(6)])
    points=directions.reshape(-1,3)[unique]
    nodes=groups.reshape(6,side,side)
    edges=np.concatenate((np.stack((nodes[:,:,:-1].ravel(),nodes[:,:,1:].ravel()),1),
                          np.stack((nodes[:,:-1].ravel(),nodes[:,1:].ravel()),1)))
    edges=np.unique(np.sort(edges,axis=1),axis=0)
    angle=np.arccos(np.clip((points[edges[:,0]]*points[edges[:,1]]).sum(1),-1,1))
    return unique,groups,weights,points,edges,angle


def weighted_quantile(values,weights,quantiles):
    order=np.argsort(values)
    cumulative=np.cumsum(weights[order])
    return np.interp(np.asarray(quantiles)*cumulative[-1],cumulative,values[order])


def component_areas(mask,edges,weights):
    """Four-neighbour land connectivity crosses shared cube nodes and corners."""
    parent=np.arange(len(mask))
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    for a,b in edges[mask[edges].all(1)]:
        ra,rb=root(a),root(b)
        if ra!=rb:parent[rb]=ra
    selected=np.flatnonzero(mask)
    labels=np.array([root(i) for i in selected],dtype=int)
    areas=np.bincount(labels,weights=weights[selected],minlength=len(mask))
    return np.sort(areas[areas>0])[::-1]


def measures(metres,radius_m=6371000.):
    z=np.asarray(metres,dtype=np.float64)
    if z.ndim!=3 or z.shape[0]!=6 or z.shape[1]!=z.shape[2] or not np.isfinite(z).all():
        raise ValueError('Expected finite six-face square terrain in metres')
    unique,groups,w,points,edges,angle=geometry(z.shape[-1]-1)
    a=z.ravel()[unique];land=a>0
    fraction=float(w[land].sum())
    areas=component_areas(land,edges,w)
    ocean=component_areas(~land,edges,w)
    both=land[edges].all(1);coast=land[edges[:,0]]!=land[edges[:,1]]
    difference=abs(a[edges[:,0]]-a[edges[:,1]])
    relief=weighted_quantile(a[land],w[land],[.5,.95,.99]).tolist() if land.any() else [None]*3
    result=dict(land_fraction=fraction,large_landmass_count=int((areas>=.01).sum()),
                largest_landmass_share_of_land=float(areas[0]/fraction) if fraction else 0.,
                top_landmass_planet_fractions=areas[:5].tolist(),
                largest_ocean_share=float(ocean[0]/(1-fraction)) if len(ocean) else 0.,
                shelf_planet_fraction=float(w[(a<0)&(a>=-200)].sum()),
                land_median_m=relief[0],land_p95_m=relief[1],land_p99_m=relief[2],
                highland_land_fraction=float(w[land&(a>=2000)].sum()/fraction) if fraction else 0.,
                coastline_crossing_length_proxy_km=float((angle[coast]*radius_m/1000).sum()),
                adjacent_land_mean_abs_change_m=float(difference[both].mean()) if both.any() else 0.,
                seam_max_error_m=float(np.max(abs(z.ravel()-a[groups]))))
    # Low-degree, rotation-invariant moments of the area-weighted land mask.
    moment=(points*(w*land)[:,None]).sum(0)
    second=np.einsum('ni,nj,n->ij',points,points,w*land)-np.eye(3)*fraction/3
    result['land_dipole_magnitude']=float(np.linalg.norm(moment))
    result['land_quadrupole_magnitude']=float(np.linalg.norm(second))
    return result


@lru_cache(maxsize=4)
def harmonic_basis(n,max_degree=12):
    """Real orthonormal spherical harmonics; no optional scipy dependency."""
    _,_,w,p,_,_=geometry(n)
    x=p[:,2];lon=np.arctan2(p[:,1],p[:,0]);s=np.sqrt(np.maximum(0,1-x*x))
    basis=[];degrees=[]
    pmm=np.ones_like(x)
    for m in range(max_degree+1):
        if m:pmm=-(2*m-1)*s*pmm
        prev=None;current=pmm
        for ell in range(m,max_degree+1):
            if ell==m:plm=current
            elif ell==m+1:plm=(2*m+1)*x*current
            else:plm=((2*ell-1)*x*current-(ell+m-1)*prev)/(ell-m)
            if ell>m:prev,current=current,plm
            norm=math.sqrt((2*ell+1)/(4*np.pi)*math.factorial(ell-m)/math.factorial(ell+m))
            if m:
                basis.extend((np.sqrt(2)*norm*plm*np.cos(m*lon),np.sqrt(2)*norm*plm*np.sin(m*lon)))
                degrees.extend((ell,ell))
            else:basis.append(norm*plm);degrees.append(ell)
    return np.stack(basis),np.asarray(degrees)


def spectrum(metres,max_degree=12):
    n=metres.shape[-1]-1
    unique,_,w,_,_,_=geometry(n)
    basis,degrees=harmonic_basis(n,max_degree)
    a=np.asarray(metres).ravel()[unique]
    a=a-np.dot(a,w)
    coefficients=basis@(a*w*4*np.pi)
    return {str(ell):float((coefficients[degrees==ell]**2).sum()) for ell in range(1,max_degree+1)}


def control_layout(metres,land_percent=None,land_relief=1.):
    """Generation-only sea-level/positive-relief edits; preserves shared nodes."""
    a=np.array(metres,dtype=np.float32,copy=True)
    if not np.isfinite(a).all() or not np.isfinite(land_relief) or land_relief<=0:
        raise ValueError('Terrain and positive relief multiplier must be finite')
    if land_percent is not None and (not np.isfinite(land_percent) or not 0<land_percent<100):
        raise ValueError('Land coverage must be between 0 and 100 percent')
    unique,_,w,_,_,_=geometry(a.shape[-1]-1)
    shift=0.
    if land_percent is not None:
        shift=float(weighted_quantile(a.ravel()[unique],w,[1-land_percent/100])[0])
        a-=shift
    a=np.where(a>0,a*land_relief,a).astype(np.float32)
    achieved=float(w[a.ravel()[unique]>0].sum()*100)
    return a,dict(requested_land_percent=land_percent,achieved_land_percent=achieved,
                  sea_level_offset_m=shift,land_relief_multiplier=land_relief,
                  method='post-sampling area-weighted sea-level shift, then positive elevation scaling')


def export_pixels(metres,height):
    """Sample raster cell centres; cube.to_equirectangular is a node-grid preview."""
    if height<2:raise ValueError('Export height must be at least two')
    width=height*2
    out=np.empty((height,width),np.float32)
    lon=-np.pi+(np.arange(width)+.5)*2*np.pi/width
    for start in range(0,height,64):
        lat=np.pi/2-(np.arange(start,min(start+64,height))+.5)*np.pi/height
        p=np.stack(np.broadcast_arrays(np.cos(lat)[:,None]*np.cos(lon)[None],
                                        np.cos(lat)[:,None]*np.sin(lon)[None],np.sin(lat)[:,None]),-1)
        out[start:start+len(lat)]=cube.sample(metres[None],p)[0]
    return out
