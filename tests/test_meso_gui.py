"""The meso UI must launch the fixed sampler and preserve reusable run settings."""
import json
import time
from pathlib import Path

import pytest
from PIL import Image

from toy_meso import gui


@pytest.fixture
def values(tmp_path):
    checkpoint = tmp_path / "trained/latest.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")
    dem = tmp_path / "map.png"
    Image.new("L", (16, 8), 128).save(dem)
    settings = gui.default_settings()
    settings.update(checkpoint=str(checkpoint), dem=str(dem), folder=str(tmp_path / "planet"))
    return settings


def test_region_command_uses_only_fixed_model_options(values):
    values.update(scope="Region", west="170", south="-10", east="-170", north="10",
                  width="320", height="160", name="Dateline.tif", ocean="-4000", stride="64",
                  steps="24", batch="4", device="cuda")
    command, saved, target = gui.prepare_command(values)
    assert command[1:4] == ["-u", "-m", "toy_meso.planet_tiled"]
    assert command[command.index("--run") + 1] == str(Path(values["checkpoint"]).parent)
    assert command[command.index("--checkpoint") + 1] == "latest.pt"
    start = command.index("--bounds") + 1
    assert command[start:start + 4] == ["170.0", "-10.0", "-170.0", "10.0"]
    for flag, expected in (("--width", "320"), ("--height", "160"), ("--ocean-metres", "-4000.0"),
                           ("--stride", "64"), ("--batch", "4"), ("--steps", "24")):
        assert command[command.index(flag) + 1] == expected
    assert target.name == "Dateline.tif"
    assert saved == values
    assert not target.parent.exists()  # validation itself does not create output


def test_globe_native_size_and_automatic_stride_are_omitted(values):
    values["width"] = "ignored outside a region"
    command, _, target = gui.prepare_command(values)
    assert "--width" not in command
    assert "--height" not in command
    assert "--stride" not in command
    assert "--ocean-metres" not in command
    assert "--bounds" not in command
    assert target.name == "planet.tif"


def test_random_mode_needs_no_map_and_ignores_inactive_png_controls(values):
    values.update(source="Random terrain", dem="", black="invalid unused value", white="", ocean="unused")
    command, _, _ = gui.prepare_command(values)
    assert "--dem" not in command
    assert command[command.index("--source") + 1] == "random"
    assert "--ocean-metres" not in command


def test_mixed_mode_requires_map_and_passes_percentage(values):
    values.update(source="Map + random", map_percent="65.25")
    command, saved, _ = gui.prepare_command(values)
    assert command[command.index("--map-weight") + 1] == "0.6525"
    assert saved["map_percent"] == "65.25"
    with pytest.raises(ValueError, match="Choose a map"):
        gui.prepare_command(dict(values, dem=""))
    for percent in ("-1", "101", "nan"):
        with pytest.raises(ValueError, match="Map percentage"):
            gui.prepare_command(dict(values, map_percent=percent))


@pytest.mark.parametrize("change, message", [
    ({"steps": "7"}, "at least 8"), ({"batch": "0"}, "at least 1"),
    ({"seed": "-1"}, "at least 0"), ({"seed": str(2**64)}, "less than"),
    ({"radius": "nan"}, "finite"), ({"radius": "0"}, "greater than zero"),
    ({"white": "-7000"}, "greater than black"), ({"height": "-1"}, "at least 1"),
    ({"scope": "Region", "south": "50", "north": "40"}, "below north"),
    ({"scope": "Region", "west": "-200"}, "between -180"),
    ({"stride": "0"}, "at least 1"), ({"name": "../elsewhere.tif"}, "plain"),
    ({"name": "wrong.png"}, "plain"), ({"device": "unsupported"}, "device"),
])
def test_bad_inputs_fail_before_starting_a_run(values, change, message):
    values.update(change)
    with pytest.raises(ValueError, match=message):
        gui.prepare_command(values)
    assert not Path(values["folder"]).exists()


def test_map_shape_and_existing_export_rejected(values):
    Image.new("L", (8, 8)).save(values["dem"])
    with pytest.raises(ValueError, match="twice as wide"):
        gui.prepare_command(values)
    Image.new("L", (16, 8)).save(values["dem"])
    folder = Path(values["folder"])
    folder.mkdir()
    (folder / "planet.tif").write_bytes(b"preserve this")
    with pytest.raises(ValueError, match="already exists"):
        gui.prepare_command(values)
    assert (folder / "planet.tif").read_bytes() == b"preserve this"


def cache_settings(values):
    return dict(version=gui.SAMPLER_VERSION, checkpoint_sha256=gui.file_sha(values["checkpoint"]),
                dem_sha256=gui.file_sha(values["dem"]), checkpoint=values["checkpoint"], dem=values["dem"],
                seed=0, steps=32, batch=16, radius_metres=6371000.0, black_metres=-6000.0,
                white_metres=6000.0, ocean_metres=None, tile=128, stride=96)


def test_cache_reuse_allows_other_regions_but_rejects_changed_model_settings(values):
    folder = Path(values["folder"])
    folder.mkdir()
    (folder / "settings.json").write_text(json.dumps(cache_settings(values)))
    values.update(scope="Region", name="Another region.tif", width="32", height="16")
    gui.prepare_command(values)
    for key, value, label in (("steps", "16", "steps"), ("batch", "8", "batch"),
                              ("stride", "64", "stride"), ("seed", "1", "seed")):
        with pytest.raises(ValueError, match=label):
            gui.prepare_command(dict(values, **{key: value}))
    old = cache_settings(values)
    old["version"] = "meso-tiled-v1"
    (folder / "settings.json").write_text(json.dumps(old))
    with pytest.raises(ValueError, match="different settings"):
        gui.prepare_command(values)


def test_saved_ui_settings_round_trip_without_executing_command(values, tmp_path):
    values.update(scope="Region", ocean="-3000", width="400", height="200", name="Custom.tif", device="cpu")
    command, saved, _ = gui.prepare_command(values)
    path = tmp_path / "meso-launch.json"
    gui.save_launch(path, saved, ["this is data, never execute it"])
    assert gui.load_settings(path) == saved
    assert not path.with_suffix(".tmp.json").exists()
    assert command[3] == "toy_meso.planet_tiled"


def test_existing_sampler_settings_restore_cache_contract(values):
    folder = Path(values["folder"])
    folder.mkdir()
    path = folder / "settings.json"
    path.write_text(json.dumps(cache_settings(values)))
    restored = gui.load_settings(path)
    assert restored["checkpoint"] == values["checkpoint"]
    assert restored["dem"] == values["dem"]
    assert restored["folder"] == str(folder)
    assert restored["ocean"] == ""
    assert restored["stride"] == "96"
    gui.prepare_command(restored)


@pytest.mark.parametrize('source, label', [('random', 'Random terrain'), ('mixed', 'Map + random')])
def test_random_and_mixed_cache_settings_restore_and_reject_different_mix(values, source, label):
    from toy_meso.conditioning import source_settings
    folder = Path(values['folder'])
    folder.mkdir()
    saved = cache_settings(values)
    saved.update(source_settings(source, values['dem'], .6525))
    if source == 'random':
        saved.update(dem=None, black_metres=-2000., white_metres=4000., ocean_metres=None)
    path = folder / 'settings.json'
    path.write_text(json.dumps(saved))
    restored = gui.load_settings(path)
    assert restored['source'] == label
    gui.prepare_command(restored)
    if source == 'mixed':
        assert restored['map_percent'] == '65.25'
        with pytest.raises(ValueError, match='map_weight'):
            gui.prepare_command(dict(restored, map_percent='60'))
        with pytest.raises(ValueError, match='input_source'):
            gui.prepare_command(dict(restored, source='Input map'))


def test_window_controls_and_failed_launch_remain_usable(values, monkeypatch):
    import tkinter as tk
    gui.configure_tk()
    root = tk.Tk()
    root.withdraw()
    app = gui.Launcher(root)
    errors = []
    monkeypatch.setattr(gui.messagebox, "showerror", lambda title, text: errors.append(text))
    try:
        for key, value in values.items():
            app.variables[key].set(value)
        assert all(str(widget.cget("state")) == "disabled" for widget in app.bounds_controls)
        app.variables["scope"].set("Region")
        assert all(str(widget.cget("state")) == "normal" for widget in app.bounds_controls)
        app.variables["dem"].set("a-map.tif")
        assert all(str(widget.cget("state")) == "disabled" for widget in app.png_controls)
        app.variables["dem"].set(values["dem"])
        app.variables['source'].set('Random terrain')
        assert all(str(widget.cget('state')) == 'disabled' for widget in app.map_controls + app.png_controls)
        assert app.mix_value.get() == '100% random'
        app.variables['source'].set('Map + random')
        assert str(app.mix_control.cget('state')) == 'normal'
        app.variables['map_percent'].set('65.25')
        assert app.mix_control.get() == 65.25
        assert app.variables['map_percent'].get() == '65.25'
        app.variables['source'].set('Input map')
        assert app.mix_value.get() == '100% map'
        app.variables["steps"].set("2")
        app.start()
        assert app.busy
        assert str(app.generate_button.cget("state")) == "disabled"
        deadline = time.monotonic() + 5
        while app.busy and time.monotonic() < deadline:
            root.update()
            time.sleep(.01)
        assert not app.busy
        assert errors and "at least 8" in errors[0]
        assert str(app.generate_button.cget("state")) == "normal"
        assert not Path(values["folder"]).exists()
        folder = Path(values['folder'])
        folder.mkdir()
        (folder / 'input.preview.png').write_bytes(b'preview')
        app.variables['source'].set('Random terrain')
        app.variables['review_input'].set('1')
        paused_values = {k: v.get() for k, v in app.variables.items()}
        app.busy = True
        app.events.put(('done', 42, folder / 'planet.tif', paused_values))
        app.poll()
        assert not app.busy and app.awaiting_review
        assert app.generate_button.cget('text') == 'Approve & Generate'
        assert app.preview == folder / 'input.preview.png'
        assert app.preview_button.cget('text') == 'Open input preview'
        app.new_run()
        assert not app.awaiting_review
        assert app.generate_button.cget('text') == 'Generate / Resume'
        assert app.variables['random_version'].get() == gui.RANDOM_VERSION
    finally:
        app.close()
