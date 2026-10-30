import numpy as np
from planet_diffusion import cube
from planet_diffusion.geometry_quality import polar_anisotropy, cube_boundary_quality


def test_edges_corners_and_gaussian_variance():
    n = 8
    members,ids,owners,counts = cube.edge_groups(n)
    assert (counts == 3).sum() == 8
    assert (counts == 2).sum() == 12*(n-1)
    assert 6*(n+1)**2-len(members)+len(owners) == 6*n*n+2
    a = cube.noise(42,0,2048,n)
    flat = a.reshape(2048,-1)
    np.testing.assert_array_equal(flat[:,members],flat[:,owners[ids]])
    np.testing.assert_allclose(a.std(axis=0),1,atol=.065)
    halo = cube.read(a,4,np.arange(-3,12)[:,None],np.arange(-3,12)[None,:],nearest=True)
    np.testing.assert_allclose(halo.std(axis=0),1,atol=.065)
    np.testing.assert_array_equal(a,cube.noise(42,0,2048,n))


def test_smooth_field_crosses_all_cube_edges():
    n = 32
    a = np.stack([cube.directions(f,np.arange(n+1)[:,None],np.arange(n+1)[None,:],n)[...,2]
                  for f in range(6)],axis=0)[None].astype(np.float32)
    a = cube.identify(a)
    for f in range(6):
        y,x = np.linspace(-3,n+3,67)[:,None],np.linspace(-3,n+3,67)[None,:]
        expected = cube.directions(f,y,x,n)[...,2]
        np.testing.assert_allclose(cube.read(a,f,y,x)[0],expected,atol=.001)


def test_polar_chart_has_same_metric_as_equatorial_chart():
    n,eps = 32,1e-3
    def metric(f,y,x):
        dx = (cube.directions(f,y,x+eps,n)-cube.directions(f,y,x-eps,n))/(2*eps)
        dy = (cube.directions(f,y+eps,x,n)-cube.directions(f,y-eps,x,n))/(2*eps)
        return np.linalg.svd(np.stack([dx,dy],axis=-1),compute_uv=False)
    for f in range(6):
        s = metric(f,np.array(16.),np.array(16.))
        np.testing.assert_allclose(s,metric(0,np.array(16.),np.array(16.)),atol=1e-12)
        assert s[0]/s[1] < 1.000001
        y,x = np.meshgrid(np.linspace(0,n,9),np.linspace(0,n,9))
        s = metric(f,y,x)
        assert (s[...,0]/s[...,1]).max() < np.sqrt(3)+1e-6


def test_prediction_consensus_and_reconstruction_preserve_constant_fields():
    a = np.ones((1,6,9,9),np.float32)
    b = cube.consensus(a,lambda p,f,y,x:p*2,size=8,stride=4)
    np.testing.assert_array_equal(b,a*2)
    low = np.full((1,6,3,3),20,np.float32)
    residual = np.full((1,6,17,17),2,np.float32)
    np.testing.assert_allclose(cube.reconstruct(residual,low),24,atol=1e-5)


def test_export_has_unique_poles_and_no_polar_metric_singularity():
    n = 64
    fields = []
    for f in range(6):
        p = cube.directions(f,np.arange(n+1)[:,None],np.arange(n+1)[None,:],n)
        fields.append(100*(p[...,0]+2*p[...,1]+.3*p[...,2]))
    a = cube.identify(np.array(fields,dtype=np.float32)[None])
    edges = cube_boundary_quality(a)
    assert edges["shared_node_error_m"] == 0
    assert len(edges["edges"]) == 12
    assert all(.9 < edge["ratio"] < 1.1 for edge in edges["edges"])
    erp = cube.to_equirectangular(a,256)[0]
    assert np.ptp(erp[0]) == 0 and np.ptp(erp[-1]) == 0
    q = polar_anisotropy(erp,latitudes=(85,88))
    for ring in q["rings"]:
        assert .95 < ring["direction_ratio"] < 1.05


def _neighbour_angles(n, face=0):
    y, x = np.meshgrid(np.arange(n+1), np.arange(n+1), indexing="ij")
    p = cube.directions(face, y, x, n)
    dx = np.arccos(np.clip((p[:, 1:]*p[:, :-1]).sum(-1), -1, 1))
    dy = np.arccos(np.clip((p[1:]*p[:-1]).sum(-1), -1, 1))
    return np.concatenate([dx.ravel(), dy.ravel()])


def test_equiangular_spacing_is_bounded_and_nominal_on_axes():
    n = 64
    nominal = np.pi/2/n
    d = _neighbour_angles(n)
    # Gnomonic charts varied 2.12x (0.60-1.27 nominal); equi-angular is <= sqrt(2).
    assert d.max()/d.min() < np.sqrt(2)+1e-6
    assert d.max() <= nominal*(1+1e-9)
    # Central chart axes are great circles stepped at exactly the nominal angle,
    # and halo reads continue that great circle across the face edge.
    axis = cube.directions(0, np.full(2*n+1, n/2), np.arange(-n//2, 3*n//2+1), n)
    steps = np.arccos(np.clip((axis[1:]*axis[:-1]).sum(-1), -1, 1))
    np.testing.assert_allclose(steps, nominal, rtol=1e-9)


def test_chart_round_trip_and_far_halos_are_finite():
    n = 32
    p = np.random.default_rng(3).normal(size=(20000, 3))
    p /= np.linalg.norm(p, axis=-1, keepdims=True)
    face, y, x = cube.coordinates(p, n)
    np.testing.assert_allclose(cube.directions(face, y, x, n), p, atol=1e-12)
    for f in range(6):
        y, x = np.meshgrid(np.arange(n+1), np.arange(n+1), indexing="ij")
        np.testing.assert_allclose(cube.chart_indices(cube.directions(f, y, x, n), f, n), (y, x), atol=1e-9)
    far = cube.directions(0, np.arange(-60, 70)[:, None], np.arange(-60, 70)[None, :], 4)
    assert np.isfinite(far).all()
    np.testing.assert_allclose(np.linalg.norm(far, axis=-1), 1)


def test_area_weight_matches_numerical_solid_angle():
    n, h = 16, 1e-4
    def solid(y, x):
        a = cube.directions(2, y, x+h, n)-cube.directions(2, y, x-h, n)
        b = cube.directions(2, y+h, x, n)-cube.directions(2, y-h, x, n)
        return np.linalg.norm(np.cross(a, b), axis=-1)/(4*h*h)
    y, x = np.meshgrid(np.linspace(-.4, n+.4, 9), np.linspace(-.4, n+.4, 9), indexing="ij")
    ratio = solid(y, x)/cube.area_weight(y, x, n)
    np.testing.assert_allclose(ratio, ratio.flat[0], rtol=1e-6)


def test_cube_surface_edge_nodes_index_neighbour_charts():
    n = 12
    members, ids, owners, _ = cube.edge_groups(n)
    for f in range(6):
        y, x = np.meshgrid(np.arange(n+1), np.arange(n+1), indexing="ij")
        edge = (y == 0) | (y == n) | (x == 0) | (x == n)
        p = cube.directions(f, y[edge], x[edge], n, normalize=False)
        for other in range(6):
            use = np.isclose(p @ cube.NORMAL[other], 1, atol=1e-12, rtol=0)
            if not use.any():
                continue
            oy, ox = cube.chart_indices(p[use], other, n)
            np.testing.assert_allclose((oy, ox), np.rint((oy, ox)), atol=1e-9)
            here = f*(n+1)**2+y[edge][use]*(n+1)+x[edge][use]
            there = other*(n+1)**2+np.rint(oy).astype(int)*(n+1)+np.rint(ox).astype(int)
            group = dict(zip(members, ids))
            assert [group.get(a, a) for a in here] == [group.get(b, b) for b in there]
