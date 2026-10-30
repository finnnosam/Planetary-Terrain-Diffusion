import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from planet_diffusion.gui import Launcher, build_command, load_launch_settings, ROOT


def save_launch(tmp_path, command, **extra):
    path = tmp_path/'launch.json'
    path.write_text(json.dumps(dict(command=command, **extra)),encoding='utf-8')
    return path


def test_saved_regional_command_restores_settings(tmp_path):
    command, seed, folder = build_command(tmp_path,42,coarse_height=32,device='cuda',
        export_height=128,export_width=256,bounds=(170,-10,-170,10),regional_only=True,
        latent_batch_size=3,model_choice='30 m',precision='bfloat16',export_climate=True)
    settings = load_launch_settings(save_launch(tmp_path,command,seed=seed))
    assert settings['folder'] == str(folder)
    assert settings['seed'] == '42'
    assert settings['scope'] == 'Region'
    assert settings['bounds'] == ['170.0','-10.0','-170.0','10.0']
    assert settings['coarse_height'] == '32'
    assert settings['detail_level'] == 'Custom'
    assert settings['export_width'] == '256'
    assert settings['export_height'] == '128'
    assert settings['model'] == '30 m'
    assert settings['precision'] == 'bfloat16'
    assert settings['latent_batch_size'] == '3'
    assert settings['export_climate'] and settings['regional_only']
    assert settings['warnings'] == []


def test_older_launch_defaults_and_relative_inputs(tmp_path):
    settings = load_launch_settings(save_launch(tmp_path,[
        'python','-m','planet_diffusion','generate','--draft','human_input/draft.png',
        '--draft-ocean-depth','1000','--height','64','--backend','diagnostic'],seed=123))
    assert settings['seed'] == '123'
    assert Path(settings['draft']) == ROOT/'human_input'/'draft.png'
    assert settings['black'] == '-1000.0'
    assert settings['white'] == '4000.0'
    assert settings['export_width'] == '128'
    assert settings['scope'] == 'Whole globe'
    assert settings['warnings'] == ['--backend diagnostic']


def test_conditioning_refinements(tmp_path):
    settings = load_launch_settings(save_launch(tmp_path,[
        'generate','--conditioning-dir','human_input/maps','--snr','.01,.3,.4,.5,.6',
        '--draft-black-metres','-6000','--draft-white-metres','7000']))
    assert settings['refinement'] == '.01'
    assert settings['climate_refinement'] == '.3,.4,.5,.6'
    assert settings['black'] == '-6000.0'
    assert settings['draft'] == ''


@pytest.mark.parametrize('command', [None, 'generate', [1],
    ['python','-m','planet_diffusion','query'],['generate','--bounds','0','1'],
    ['generate','--height','bad'],['generate','--snr','1,2'],['generate','--regional-only']])
def test_invalid_settings_raise_value_error(tmp_path,command):
    with pytest.raises(ValueError):
        load_launch_settings(save_launch(tmp_path,command))


def test_loading_updates_ui_without_recomputing_saved_resolution(tmp_path,monkeypatch):
    from planet_diffusion import gui
    path = save_launch(tmp_path,['generate','--bounds','-2','-2','2','2',
        '--regional-only','--coarse-height','32','--width','100','--height','80'])
    settings = load_launch_settings(path)
    class Variable:
        def __init__(self,value=''):
            self.value = value
        def get(self):
            return self.value
        def set(self,value):
            self.value = value
    launcher = Launcher.__new__(Launcher)
    for name,value in settings.items():
        if name not in ('bounds','warnings'):
            setattr(launcher,name,Variable())
    launcher.bounds = [Variable() for _ in range(4)]
    launcher.status = Variable()
    states = []
    launcher.bound_controls = [SimpleNamespace(configure=lambda **kw:states.append(kw))]
    launcher.generate = SimpleNamespace(configure=lambda **kw:None)
    launcher.append = lambda text:None
    monkeypatch.setattr(gui.filedialog,'askopenfilename',lambda **kw:str(path))
    launcher.load_settings()
    assert launcher.coarse_height.get() == '32'
    assert launcher.export_width.get() == '100'
    assert launcher.export_height.get() == '80'
    assert launcher.detail_level.get() == 'Custom'
    assert launcher.regional_only.get()
    assert states == [{'state':'normal'}]
