import json
from pathlib import Path
import pytest
from planet_training.gui import prepare_command,selected_checkpoint


def settings(tmp_path):
    checkpoint=tmp_path/'model.pt';checkpoint.write_bytes(b'placeholder')
    return dict(checkpoint=str(checkpoint),out=str(tmp_path/'new-planet'),seed='42',land_percent='',land_relief='1')


def test_continent_command_defaults_and_controls(tmp_path):
    values=settings(tmp_path);command,out=prepare_command(values)
    assert '--land-percent' not in command and command[command.index('--steps')+1]=='64'
    assert out==tmp_path/'new-planet' and command[command.index('--checkpoint')+1]=='model.pt'
    values.update(land_percent='30',land_relief='1.25')
    command,_=prepare_command(values)
    assert command[command.index('--land-percent')+1]=='30.0'
    assert command[command.index('--land-relief')+1]=='1.25'


@pytest.mark.parametrize('key,value',[('seed','x'),('seed','-1'),('seed',str(2**64)),
    ('land_percent','0'),('land_percent','100'),('land_percent','nan'),('land_percent','text'),
    ('land_relief','0'),('land_relief','inf'),('land_relief','-2')])
def test_continent_command_rejects_invalid_values(tmp_path,key,value):
    values=settings(tmp_path);values[key]=value
    with pytest.raises(ValueError):prepare_command(values)


def test_continent_command_protects_existing_output_and_requires_checkpoint(tmp_path):
    values=settings(tmp_path)
    folder=Path(values['out']);folder.mkdir();(folder/'layout.tif').write_bytes(b'keep')
    with pytest.raises(ValueError,match='preserve'):prepare_command(values)
    values['checkpoint']=str(tmp_path/'missing.pt')
    with pytest.raises(ValueError,match='checkpoint'):prepare_command(values)


def test_selected_continent_checkpoint_resolves_relative_selection(tmp_path,monkeypatch):
    monkeypatch.setattr('planet_training.gui.ROOT',tmp_path)
    folder=tmp_path/'outputs/continents';folder.mkdir(parents=True)
    ck=folder/'selected/best.pt';ck.parent.mkdir();ck.write_bytes(b'checkpoint')
    (folder/'selection.json').write_text(json.dumps(dict(checkpoint='outputs/continents/selected/best.pt')))
    assert selected_checkpoint()==ck


def test_continent_widgets_can_construct_without_showing_a_window():
    import tkinter as tk
    from planet_training.gui import Launcher,configure_tk
    configure_tk();root=tk.Tk();root.withdraw()
    try:
        app=Launcher(root);root.update_idletasks()
        assert app.values['land_percent'].get()=='' and app.values['land_relief'].get()=='1'
        assert str(app.generate_button['state'])=='normal'
        root.after_cancel(app.poll_id)
    finally:root.destroy()
