import io
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest

from planet_diffusion import cube
from planet_diffusion.cache import PredictionCache
from planet_diffusion.gui import preview_guide_height
from planet_diffusion.sparse import Channels
from planet_diffusion.worker import serve, PREFIX, BackendPool


@pytest.mark.parametrize('device,capability,supported,expected',[
    ('cpu',8,True,'float32'),('cuda',8,True,'bfloat16'),
    ('cuda',9,True,'bfloat16'),('cuda',7,True,'float32'),
    ('cuda',8,False,'float32')])
def test_auto_precision_is_a_stable_native_hardware_choice(device,capability,supported,expected):
    from types import SimpleNamespace
    from planet_diffusion.backends import resolve_precision
    torch = SimpleNamespace(cuda=SimpleNamespace(get_device_capability=lambda:(capability,0),
                                                is_bf16_supported=lambda:supported))
    assert resolve_precision('auto',device,torch) == expected
    assert resolve_precision('auto',device,torch) == expected
    assert resolve_precision('float32',device,torch) == 'float32'


def test_launcher_preserves_auto_and_explicit_precision(tmp_path):
    from planet_diffusion.gui import build_command
    for precision in ('auto','float32','tf32','bfloat16'):
        command,_,_ = build_command(tmp_path/precision,42,device='auto',precision=precision)
        assert command[command.index('--precision')+1] == precision


def test_preview_height_matches_export_sampling_and_date_line():
    assert preview_guide_height((-2,-2,2,2),512,512) == 92
    assert preview_guide_height((178,-2,-178,2),512,512) == 92
    assert preview_guide_height((-180,-90,180,90),128,64) == 4


def test_cli_preview_chooses_grid_and_rejects_negative_budgets(tmp_path):
    from planet_diffusion.cli import main
    argv = ['generate','--backend','diagnostic','--regional-only','--preview','--seed','42',
        '--coarse-steps','2','--bounds','-2','-2','2','2','--width','2','--height','2',
        '--state',str(tmp_path/'state'),'--output',str(tmp_path/'preview.tif')]
    assert main(argv) == 0
    info = json.loads((tmp_path/'state/planet.json').read_text())
    assert info['coarse_height'] == 4
    with pytest.raises(SystemExit) as exc:
        main(argv+['--patch-cache-mib','-1'])
    assert exc.value.code == 2


def test_query_rejects_precision_mismatch_before_decoding():
    from planet_diffusion.query import query_loaded_region
    from planet_diffusion.backends import DiagnosticBackend
    with pytest.raises(ValueError,match='precision'):
        query_loaded_region(DiagnosticBackend(),(None,{'precision':'bfloat16'},{},None,None),
                            (-2,-2,2,2),2,2)


def test_channel_geometry_matches_scalar_views_at_seams():
    rng = np.random.default_rng(3)
    values = rng.normal(size=(3,6,49,49)).astype(np.float32)
    channels = Channels(48,3,lambda c,f,y,x:values[c,f,y,x],tile=8)
    y,x = np.arange(-3,52)[:,None]+.3,np.arange(-3,52)[None,:]+.7
    for nearest in (False,True):
        for face in range(6):
            np.testing.assert_array_equal(channels.read(face,y,x,nearest),
                np.stack([field.read(face,y,x,nearest) for field in channels.fields]))


def test_evicted_disk_prediction_skips_only_unchanged_validation(tmp_path):
    cache = PredictionCache(tmp_path,{},lambda *args:None)
    cache.save(np.ones((1,4,4),np.float32),(1,4,4),0,0,0)
    first = cache.load((1,4,4),0,0,0)
    assert not isinstance(first,np.memmap)
    cache.load((1,4,4),0,0,0)
    assert cache.validation_scans == 1
    np.save(tmp_path/'0_0_0.npy',np.full((1,4,4),np.nan,np.float32))
    with pytest.raises(ValueError,match='Invalid decoder prediction'):
        cache.load((1,4,4),0,0,0)


def test_worker_survives_failed_jobs_and_keeps_framed_logs():
    requests = io.StringIO('\n'.join(json.dumps({'argv':args}) for args in
        [['generate'],['query'],['gui'],['verify']])+'\n')
    output = io.StringIO()
    pools = []
    def run(args,backend_factory):
        pools.append(backend_factory)
        print('running',args[0])
        if args[0] == 'query':
            raise SystemExit(2)
        return 0
    serve(requests,output,run)
    events = [json.loads(line[len(PREFIX):]) for line in output.getvalue().splitlines()]
    assert [e['code'] for e in events if e['event']=='done'] == [0,2,1,0]
    assert len(pools) == 3 and all(pool is pools[0] for pool in pools)
    assert any('running' in e.get('text','') for e in events)


def test_backend_pool_shares_models_but_not_mutable_job_configuration(monkeypatch):
    from planet_diffusion import backends
    class FakeBackend:
        device = 'cpu'
        def __init__(self,*args,**kwargs):
            self.metadata = dict(model=args[0])
            self.models = {}
            self.cond_snr = np.ones(5,np.float32)
    monkeypatch.setattr(backends,'TerrainBackend',FakeBackend)
    pool = BackendPool()
    first = pool('one')
    first.models['latent'] = object()
    first.cond_snr[:] = 9
    second = pool('one')
    assert second.models is first.models and np.all(second.cond_snr == 1)
    assert pool('two').models == {}


def test_model_hash_reuses_unchanged_files_and_detects_updates(tmp_path):
    from planet_diffusion.backends import _model_revision, model_files
    model = tmp_path/'weights.bin'
    model.write_bytes(b'abc')
    _model_revision.cache_clear()
    first = _model_revision(str(tmp_path),model_files(tmp_path))
    assert _model_revision(str(tmp_path),model_files(tmp_path)) == first
    assert _model_revision.cache_info().hits == 1
    model.write_bytes(b'different weights')
    assert _model_revision(str(tmp_path),model_files(tmp_path)) != first


def test_prepared_dependencies_match_actual_reads_at_cube_edges(tmp_path):
    from planet_diffusion.sparse import Stage
    calls = [[],[]]
    stages = []
    for index in range(2):
        def predict(face,y,x,index=index):
            calls[index].append((face,y,x))
            return np.full((3,8,8),face+y*.1+x*.01,np.float32)
        stages.append(Stage(tmp_path/str(index),{},'test',16,3,8,4,predict,
                            tile=4,include_endpoint=True,batch_size=3))
    requests = [(0,np.arange(-2,6)[:,None],np.arange(12,19)[None,:]),
                (4,np.arange(12,17)[:,None],np.arange(0,5)[None,:])]
    stages[1].prepare_reads(requests)
    for face,y,x in requests:
        np.testing.assert_array_equal(stages[0].read(face,y,x,nearest=True),
                                      stages[1].read(face,y,x,nearest=True))
    assert set(calls[0]) == set(calls[1])


def test_hidden_launcher_reuses_worker_and_restarts_after_cancel(tmp_path):
    import time
    import tkinter as tk
    from planet_diffusion.gui import Launcher, configure_tk, python_executable
    from planet_diffusion.storage import save_state
    configure_tk()
    root = tk.Tk()
    root.withdraw()
    launcher = Launcher(root)
    state = tmp_path/'state'
    save_state(state,np.zeros((3,4),np.float32),{'native_height':2})
    command = [python_executable(),'-u','-m','planet_diffusion','verify','--state',str(state)]
    def wait_done():
        end = time.monotonic()+15
        while launcher.process is not None and time.monotonic() < end:
            root.update()
            time.sleep(.01)
        assert launcher.process is None
    try:
        launcher._launch(command,tmp_path,'query','first.log')
        wait_done()
        pid = launcher.worker.pid
        launcher._launch(command,tmp_path,'query','second.log')
        wait_done()
        assert launcher.worker.pid == pid
        launcher._launch(command,tmp_path,'query','cancelled.log')
        launcher.cancel()
        wait_done()
        launcher._launch(command,tmp_path,'query','restarted.log')
        wait_done()
        assert launcher.worker.pid != pid
    finally:
        launcher.close()


def test_real_worker_two_jobs_and_failure_recovery(tmp_path):
    run = tmp_path/'run'
    argv = ['generate','--backend','diagnostic','--regional-only','--seed','42',
        '--coarse-height','4','--coarse-steps','2','--bounds','-2','-2','2','2',
        '--width','2','--height','2','--state',str(run),'--output',str(tmp_path/'first.tif')]
    requests = [{'argv':argv},{'argv':['query','--state',str(run),'--bounds','-2','-2','2','2',
        '--width','2','--height','2','--output',str(tmp_path/'second.tif')]},
        {'argv':['invalid']},{'argv':['verify','--state',str(run)]}]
    result = subprocess.run([sys.executable,'-u','-m','planet_diffusion.worker'],
        input=''.join(json.dumps(r)+'\n' for r in requests),text=True,capture_output=True,
        cwd=Path(__file__).resolve().parents[1],timeout=60)
    assert result.returncode == 0, result.stderr
    events = [json.loads(line[len(PREFIX):]) for line in result.stdout.splitlines()]
    assert [e['code'] for e in events if e['event']=='done'] == [0,0,1,0]
    import rasterio
    with rasterio.open(tmp_path/'first.tif') as a, rasterio.open(tmp_path/'second.tif') as b:
        np.testing.assert_array_equal(a.read(),b.read())
