import numpy as np
import pytest
from planet_diffusion import cube
from planet_training.geography import geometry,measures,spectrum,control_layout,export_pixels
from planet_training.evaluate_layout import CopyScreen
from planet_training.data import layout_target,PaleoRaster
from toy_meso.sphere import decode


def field(n,axis=0):
    return np.stack([cube.directions(f,np.arange(n+1)[:,None],np.arange(n+1)[None],n)[...,axis]*3000 for f in range(6)]).astype(np.float32)


def test_unique_spherical_connectivity_crosses_faces_and_ignores_duplicate_edges():
    n=12
    unique,groups,w,p,e,a=geometry(n)
    assert len(unique)==6*n*n+2 and np.isclose(w.sum(),1)
    z=field(n)
    report=measures(z)
    assert .45<report['land_fraction']<.55
    assert report['large_landmass_count']==1
    assert report['largest_landmass_share_of_land']==pytest.approx(1)
    assert report['largest_ocean_share']==pytest.approx(1)
    assert report['seam_max_error_m']==0
    ocean=measures(np.full_like(z,-1000))
    assert ocean['land_fraction']==0 and ocean['land_p95_m'] is None and ocean['largest_ocean_share']==pytest.approx(1)
    land=measures(np.full_like(z,1000))
    assert land['land_fraction']==pytest.approx(1) and land['large_landmass_count']==1


def test_relief_and_land_controls_preserve_defaults_seams_and_area_target():
    z=field(24)+.123*field(24,1)+.067*field(24,2)+140
    neutral,meta=control_layout(z)
    np.testing.assert_array_equal(neutral,z)
    edited,meta=control_layout(z,30,1.5)
    assert abs(meta['achieved_land_percent']-30)<.15
    np.testing.assert_allclose(edited,cube.identify(edited[None])[0],atol=1e-4)
    sea_only,_=control_layout(z,30,1)
    np.testing.assert_array_equal(edited>0,sea_only>0)
    np.testing.assert_allclose(edited[sea_only<=0],sea_only[sea_only<=0])
    np.testing.assert_allclose(edited[sea_only>0],sea_only[sea_only>0]*1.5)
    for percent,relief in ((0,1),(100,1),(float('nan'),1),(30,-1),(30,float('inf'))):
        with pytest.raises(ValueError):control_layout(z,percent,relief)


def test_spherical_power_is_orientation_invariant_and_recovers_degree_one():
    x=spectrum(field(24,0),max_degree=4)
    y=spectrum(field(24,1),max_degree=4)
    for ell in x:assert x[ell]==pytest.approx(y[ell],abs=1e-4,rel=1e-4)
    assert x['1']==pytest.approx(4*np.pi*3000**2/3,rel=.003)
    assert x['2']<x['1']*1e-8 and x['3']<x['1']*.001


def test_export_uses_declared_raster_cell_centres_without_pole_endpoints():
    z=field(24,2);image=export_pixels(z,32)
    assert image.shape==(32,64)
    lat=np.radians(90-(np.arange(32)+.5)*180/32)
    lon=np.radians(-180+(np.arange(64)+.5)*360/64)
    p=np.stack(np.broadcast_arrays(np.cos(lat)[:,None]*np.cos(lon)[None],
                                  np.cos(lat)[:,None]*np.sin(lon)[None],np.sin(lat)[:,None]),-1)
    np.testing.assert_array_equal(image,cube.sample(z[None],p)[0])


def test_copy_screen_detects_rotated_training_land():
    import torch
    torch.set_num_threads(2)
    la=np.radians(np.arange(90,-91,-1));lo=np.radians(np.arange(-180,181))
    z=(np.cos(la[:,None])*np.cos(lo[None])*1800+np.sin(2*la[:,None])*800-300).astype(np.float32)
    rotation=np.array([[0,1,0],[1,0,0],[0,0,-1.]])
    sample=decode(layout_target(PaleoRaster(z),rotation,n=24,sub=2))
    screen=CopyScreen(np.stack([z]),[0],[{'age_ma':100}],points=512,rotations=0)
    report=screen.match(sample)
    assert report['approx_best_training_land_iou']>.97 and report['closest_training_age_ma']==100


def test_controlled_export_and_matched_evaluator(tmp_path):
    import json
    import torch
    import rasterio
    from planet_training.layout import LayoutDenoiser
    from planet_training.data import VERSION,manifest_id
    from planet_training.generate import main as generate
    from planet_training.evaluate_layout import main as evaluate
    torch.set_num_threads(2)
    la=np.radians(np.arange(90,-91,-1));lo=np.radians(np.arange(-180,181))
    values=(1800*np.cos(la[:,None])*np.cos(lo[None])-300).astype(np.float32)
    data=tmp_path/'data';data.mkdir()
    np.save(data/'maps.npy',np.stack((values,values*.8,values*.9)))
    meta=dict(kind='layout',version=VERSION,face_intervals=6,supersample=2,
              spacing_m=161280.,radius_m=6371000.,records=[dict(split=s,age_ma=i*100) for i,s in enumerate(('train','val','test'))])
    (data/'manifest.json').write_text(json.dumps(meta))
    config=dict(ch=8,mults=[1,2],blocks=1,dropout=0.)
    model=LayoutDenoiser(n=6,**config)
    run=tmp_path/'run';run.mkdir();checkpoint=run/'best.pt'
    torch.save(dict(stage='layout',training_version=VERSION,dataset_id=manifest_id(meta),
                    meta=meta,config=config,stats=dict(mean=-20.,std=40.),ema=model.state_dict(),step=10),checkpoint)
    out=tmp_path/'export'
    generate(['--run',str(run),'--out',str(out),'--steps','2','--height','32','--device','cpu','--land-percent','30','--land-relief','1.4'])
    report=json.loads((out/'layout.json').read_text())
    assert report['controls']['requested_land_percent']==30 and report['controls']['land_relief_multiplier']==1.4
    assert len(report['checkpoint_sha256'])==64 and report['geography']['seam_max_error_m']<1e-3
    with rasterio.open(out/'layout.tif') as ds:
        assert ds.shape==(32,64) and np.isfinite(ds.read(1)).all() and ds.units[0]=='m'
    with pytest.raises(ValueError,match='new output'):generate(['--run',str(run),'--out',str(out),'--device','cpu'])
    ev=tmp_path/'evaluation'
    evaluate(['--reference',str(checkpoint),'--candidate',str(checkpoint),'--data',str(data),
              '--out',str(ev),'--steps','2','--seeds','42','--device','cpu','--batch','2'])
    r=json.loads((ev/'evaluation.json').read_text())
    assert r['models'][0]['val_loss']==r['models'][1]['val_loss']
    assert r['models'][0]['planets']==r['models'][1]['planets']
    assert len(r['references']['train']['planets'])==1 and 'test' not in r['references']
    assert (ev/'comparison.png').is_file() and (ev/'distributions.png').is_file()
