"""Desktop controls for the fixed, elevation-only tiled meso generator."""
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import re
import secrets
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

ROOT = Path(__file__).resolve().parents[1]
LAUNCH_VERSION = "meso-ui-v1"
SAMPLER_VERSION = "meso-tiled-v2-cell-centres"


def default_settings():
    return dict(checkpoint=str(ROOT / "outputs/meso/run2-lowlr/latest.pt"),
                dem=str(ROOT / "human_input/B2W.png"),
                folder=str(ROOT / "outputs/meso/planet-ui"), seed="0",
                black="-6000", white="6000", ocean="", scope="Whole globe",
                west="5", south="44", east="15", north="49", width="", height="",
                name="", steps="32", batch="16", stride="", radius="6371000",
                device="auto")


def python_executable():
    path = Path(sys.executable)
    console = path.with_name("python.exe")
    return str(console if path.name.lower() == "pythonw.exe" and console.exists() else path)


def resolve_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def file_sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_number(value, label):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a number.") from None
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite.")
    return number


def integer(value, label, minimum=1):
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a whole number.") from None
    if number < minimum:
        raise ValueError(f"{label} must be at least {minimum}.")
    return number


def prepare_command(values):
    """Validate inputs and existing cache before creating a run or loading a model."""
    settings = default_settings()
    settings.update(values)
    for key in ("checkpoint", "dem", "folder"):
        if not str(settings[key]).strip():
            raise ValueError(f"Choose a {dict(checkpoint='model checkpoint', dem='map', folder='run folder')[key]}.")
        settings[key] = str(resolve_path(settings[key]))
    checkpoint, dem, folder = (Path(settings[k]) for k in ("checkpoint", "dem", "folder"))
    if not checkpoint.is_file():
        raise ValueError("The model checkpoint was not found. Choose your .pt file.")
    if not dem.is_file() or dem.suffix.lower() not in (".png", ".tif", ".tiff"):
        raise ValueError("Choose an existing global PNG or elevation TIFF.")
    if folder.exists() and not folder.is_dir():
        raise ValueError("The run folder points to a file.")
    if dem.suffix.lower() == ".png":
        from PIL import Image
        with Image.open(dem) as image:
            width, height = image.size
    else:
        import rasterio
        with rasterio.open(dem) as image:
            width, height = image.width, image.height
            if image.count != 1:
                raise ValueError("Choose a single-band elevation TIFF in metres.")
    if height < 2 or width != 2 * height:
        raise ValueError("The input must be a global map twice as wide as it is tall, with north at the top.")
    seed = integer(settings["seed"], "Seed", 0)
    if seed >= 2**64:
        raise ValueError("Seed must be less than 18446744073709551616.")
    steps = integer(settings["steps"], "Sampling steps", 8)
    batch = integer(settings["batch"], "Tile batch size")
    radius = finite_number(settings["radius"], "Planet radius")
    if radius <= 0:
        raise ValueError("Planet radius must be greater than zero.")
    black = finite_number(settings["black"], "Black elevation")
    white = finite_number(settings["white"], "White elevation")
    ocean = finite_number(settings["ocean"], "Pure-black ocean elevation") if settings["ocean"].strip() else None
    if dem.suffix.lower() == ".png" and white <= black:
        raise ValueError("White elevation must be greater than black elevation.")
    if settings["device"] not in ("auto", "cpu", "cuda"):
        raise ValueError("Choose Auto, CPU, or CUDA for the device.")
    if settings["scope"] not in ("Whole globe", "Region"):
        raise ValueError("Choose Whole globe or Region.")
    bounds = None
    if settings["scope"] == "Region":
        bounds = [finite_number(settings[k], k.title()) for k in ("west", "south", "east", "north")]
        west, south, east, north = bounds
        if not (-180 <= west <= 180 and -180 <= east <= 180):
            raise ValueError("West and east must be between -180 and 180 degrees.")
        if not -90 <= south < north <= 90:
            raise ValueError("South must be below north, within -90 to 90 degrees.")
    height = integer(settings["height"], "Output height") if settings["height"].strip() else None
    width = integer(settings["width"], "Output width") if bounds and settings["width"].strip() else None
    stride = integer(settings["stride"], "Tile stride") if settings["stride"].strip() else None
    name = settings["name"].strip() or ("region.tif" if bounds else "planet.tif")
    if (Path(name).name != name or any(c in name for c in '<>:"/\\|?*')
            or Path(name).suffix.lower() not in (".tif", ".tiff")):
        raise ValueError("Output filename must be a plain .tif or .tiff filename, for example Alps.tif.")
    target = folder / name
    if target.exists():
        raise ValueError(f"{name} already exists. Choose a different output filename, or use New run.")

    pinned = dict(version=SAMPLER_VERSION, checkpoint_sha256=file_sha(checkpoint), dem_sha256=file_sha(dem),
                  seed=seed, steps=steps, batch=batch, radius_metres=radius,
                  black_metres=black, white_metres=white, ocean_metres=ocean)
    cache_settings = folder / "settings.json"
    if cache_settings.exists():
        old = json.loads(cache_settings.read_text(encoding="utf-8"))
        if stride is not None:
            pinned["stride"] = stride
        elif "tile" in old:
            pinned["stride"] = old["tile"] * 3 // 4
        changed = [key for key, value in pinned.items() if old.get(key) != value]
        if changed:
            raise ValueError("This folder contains tiles made with different settings "
                             f"({', '.join(changed)}). Load its settings to resume, or use New run.")
    cmd = [python_executable(), "-u", "-m", "toy_meso.planet_tiled", "--run", str(checkpoint.parent),
           "--checkpoint", checkpoint.name, "--dem", str(dem), "--out", str(folder),
           "--black-metres", str(black), "--white-metres", str(white), "--seed", str(seed),
           "--steps", str(steps), "--batch", str(batch), "--radius-metres", str(radius),
           "--device", settings["device"], "--name", name]
    if ocean is not None:
        cmd += ["--ocean-metres", str(ocean)]
    if bounds:
        cmd += ["--bounds", *map(str, bounds)]
    if height is not None:
        cmd += ["--height", str(height)]
    if width is not None:
        cmd += ["--width", str(width)]
    if stride is not None:
        cmd += ["--stride", str(stride)]
    settings["seed"] = str(seed)
    return cmd, settings, target


def save_launch(path, settings, command):
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(dict(version=LAUNCH_VERSION, settings=settings, command=command), indent=2),
                         encoding="utf-8")
    temporary.replace(path)


def load_settings(path):
    """Read settings as data; never execute a saved command."""
    path = Path(path).resolve()
    saved = json.loads(path.read_text(encoding="utf-8-sig"))
    values = default_settings()
    if saved.get("version") == LAUNCH_VERSION:
        fields = saved.get("settings")
        if not isinstance(fields, dict) or not all(isinstance(v, str) for v in fields.values()):
            raise ValueError("This is not a valid meso UI settings file.")
        if fields.keys() - values.keys():
            raise ValueError("The saved file contains unsupported settings.")
        values.update(fields)
    elif saved.get("version") == SAMPLER_VERSION:
        mapping = dict(checkpoint="checkpoint", dem="dem", seed="seed", black="black_metres",
                       white="white_metres", ocean="ocean_metres", steps="steps", batch="batch",
                       stride="stride", radius="radius_metres")
        for key, source in mapping.items():
            if source in saved:
                values[key] = "" if saved[source] is None else str(saved[source])
        values["folder"] = str(path.parent)
    else:
        raise ValueError("Choose meso-launch.json or settings.json from the fixed meso generator.")
    return values


def configure_tk():
    libraries = Path(sys.prefix) / "tcl"
    for name, folder, filename in (("TCL_LIBRARY", f"tcl{tk.TclVersion}", "init.tcl"),
                                    ("TK_LIBRARY", f"tk{tk.TkVersion}", "tk.tcl")):
        path = libraries / folder
        if (path / filename).is_file():
            os.environ.setdefault(name, str(path))


class Launcher:
    def __init__(self, root):
        self.root = root
        root.title("Meso Terrain — Fixed Model")
        root.geometry("960x760")
        root.minsize(880, 700)
        self.variables = {key: tk.StringVar(value=value) for key, value in default_settings().items()}
        self.status = tk.StringVar(value="Ready. Choose a map, then generate a globe or region.")
        self.events = queue.Queue()
        self.process = None
        self.process_lock = threading.Lock()
        self.stop_requested = threading.Event()
        self.busy = False
        self.controls = []
        self.png_controls = []
        self.bounds_controls = []
        self.preview = None
        self.run_folder = None

        style = ttk.Style(root)
        style.configure("Title.TLabel", font=("Segoe UI", 18, "bold"))
        style.configure("Hint.TLabel", foreground="#555555")
        main = ttk.Frame(root, padding=18)
        main.pack(fill="both", expand=True)
        main.columnconfigure(0, weight=1)
        main.rowconfigure(8, weight=1)
        ttk.Label(main, text="Meso Terrain", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(main, text="Elevation from your map · corrected sampler · cached tiles for resuming",
                  style="Hint.TLabel").grid(row=1, column=0, sticky="w", pady=(2, 12))

        inputs = ttk.LabelFrame(main, text="Map and model", padding=10)
        inputs.grid(row=2, column=0, sticky="ew")
        inputs.columnconfigure(1, weight=1)
        self.path_row(inputs, 0, "Input map", "dem", self.browse_map)
        self.path_row(inputs, 1, "Model checkpoint", "checkpoint", self.browse_checkpoint)
        ttk.Label(inputs, text="Global 2:1 PNG or elevation TIFF in metres. The checkpoint determines terrain scale.",
                  style="Hint.TLabel").grid(row=2, column=0, columnspan=3, sticky="w", pady=(4, 0))

        levels = ttk.Frame(inputs)
        levels.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(9, 0))
        for column, (label, key) in enumerate((("Black (m)", "black"), ("White (m)", "white"),
                                              ("Pure-black ocean (m)", "ocean"))):
            ttk.Label(levels, text=label).grid(row=0, column=2 * column, padx=(0 if column == 0 else 12, 5))
            entry = self.entry(levels, key, width=12)
            entry.grid(row=0, column=2 * column + 1)
            self.png_controls.append(entry)
        ttk.Label(inputs, text="PNG only. Leave pure-black ocean blank to use the normal black-to-white mapping.",
                  style="Hint.TLabel").grid(row=4, column=0, columnspan=3, sticky="w", pady=(5, 0))

        output = ttk.LabelFrame(main, text="Run and area", padding=10)
        output.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        output.columnconfigure(1, weight=1)
        self.path_row(output, 0, "Run folder", "folder", self.browse_folder)
        row = ttk.Frame(output)
        row.grid(row=1, column=0, columnspan=3, sticky="w", pady=7)
        ttk.Label(row, text="Seed").pack(side="left", padx=(0, 8))
        self.entry(row, "seed", width=22).pack(side="left")
        self.button(row, "Random", lambda: self.variables["seed"].set(str(secrets.randbits(32)))).pack(side="left", padx=6)
        ttk.Label(row, text="Area").pack(side="left", padx=(20, 8))
        self.combo(row, "scope", ("Whole globe", "Region"), width=16).pack(side="left")
        bounds = ttk.Frame(output)
        bounds.grid(row=2, column=0, columnspan=3, sticky="w")
        for column, key in enumerate(("west", "south", "east", "north")):
            ttk.Label(bounds, text=key.title()).grid(row=0, column=column * 2, padx=(0 if column == 0 else 12, 5))
            entry = self.entry(bounds, key, width=12)
            entry.grid(row=0, column=column * 2 + 1)
            self.bounds_controls.append(entry)
        ttk.Label(output, text="Bounds are degrees. East below west crosses the date line. Regions reuse this run's tiles.",
                  style="Hint.TLabel").grid(row=3, column=0, columnspan=3, sticky="w", pady=(5, 0))

        self.advanced_button = ttk.Button(main, text="Show advanced settings", command=self.toggle_advanced)
        self.advanced_button.grid(row=4, column=0, sticky="w", pady=10)
        self.advanced = ttk.LabelFrame(main, text="Generation and export", padding=10)
        self.advanced.columnconfigure(1, weight=1)
        self.advanced.columnconfigure(3, weight=1)
        for index, (label, key) in enumerate((("Sampling steps (8 or more)", "steps"), ("Tile batch size", "batch"),
                                            ("Planet radius (m)", "radius"), ("Tile stride (blank = model default)", "stride"),
                                            ("Output height (blank = native)", "height"), ("Region width (blank = automatic)", "width"),
                                            ("Output filename (blank = automatic)", "name"))):
            r, c = divmod(index, 2)
            ttk.Label(self.advanced, text=label).grid(row=r, column=c * 2, sticky="w", padx=(0 if c == 0 else 18, 8), pady=4)
            entry = self.entry(self.advanced, key, width=20)
            entry.grid(row=r, column=c * 2 + 1, sticky="ew", pady=4)
            if key == "width":
                self.width_control = entry
        ttk.Label(self.advanced, text="Device").grid(row=3, column=2, sticky="w", padx=(18, 8))
        self.combo(self.advanced, "device", ("auto", "cpu", "cuda"), width=18).grid(row=3, column=3, sticky="ew")
        ttk.Label(self.advanced, text="Globe width is twice its height. More export pixels do not change the model's terrain scale.",
                  style="Hint.TLabel").grid(row=4, column=0, columnspan=4, sticky="w", pady=(6, 0))

        actions = ttk.Frame(main)
        actions.grid(row=6, column=0, sticky="ew", pady=(0, 8))
        self.generate_button = ttk.Button(actions, text="Generate / Resume", command=self.start)
        self.generate_button.pack(side="left")
        self.stop_button = ttk.Button(actions, text="Stop", command=self.cancel, state="disabled")
        self.stop_button.pack(side="left", padx=6)
        self.button(actions, "New run", self.new_run).pack(side="left", padx=6)
        self.button(actions, "Load settings…", self.load).pack(side="left", padx=6)
        ttk.Button(actions, text="Open folder", command=self.open_folder).pack(side="right")
        self.preview_button = ttk.Button(actions, text="Open preview", command=self.open_preview, state="disabled")
        self.preview_button.pack(side="right", padx=6)

        progress_row = ttk.Frame(main)
        progress_row.grid(row=7, column=0, sticky="ew")
        progress_row.columnconfigure(0, weight=1)
        ttk.Label(progress_row, textvariable=self.status, wraplength=890).grid(row=0, column=0, sticky="w")
        self.progress = ttk.Progressbar(progress_row, mode="determinate")
        self.progress.grid(row=1, column=0, sticky="ew", pady=(6, 8))
        logs = ttk.Frame(main)
        logs.grid(row=8, column=0, sticky="nsew")
        logs.columnconfigure(0, weight=1)
        logs.rowconfigure(0, weight=1)
        self.log = tk.Text(logs, height=6, wrap="word", state="disabled", font=("Consolas", 9))
        self.log.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(logs, command=self.log.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.log.configure(yscrollcommand=scroll.set)
        self.variables["scope"].trace_add("write", lambda *_: self.update_controls())
        self.variables["dem"].trace_add("write", lambda *_: self.update_controls())
        self.update_controls()
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.poll_id = root.after(100, self.poll)

    def entry(self, parent, key, **kwargs):
        widget = ttk.Entry(parent, textvariable=self.variables[key], **kwargs)
        self.controls.append((widget, "normal"))
        return widget

    def button(self, parent, text, command):
        widget = ttk.Button(parent, text=text, command=command)
        self.controls.append((widget, "normal"))
        return widget

    def combo(self, parent, key, choices, **kwargs):
        widget = ttk.Combobox(parent, textvariable=self.variables[key], values=choices, state="readonly", **kwargs)
        self.controls.append((widget, "readonly"))
        return widget

    def path_row(self, parent, row, label, key, command):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 10), pady=3)
        self.entry(parent, key).grid(row=row, column=1, sticky="ew", pady=3)
        self.button(parent, "Browse…", command).grid(row=row, column=2, padx=(8, 0), pady=3)

    def update_controls(self):
        for widget, state in self.controls:
            widget.configure(state="disabled" if self.busy else state)
        self.generate_button.configure(state="disabled" if self.busy else "normal")
        if self.busy:
            return
        is_png = self.variables["dem"].get().lower().endswith(".png")
        for widget in self.png_controls:
            widget.configure(state="normal" if is_png else "disabled")
        regional = self.variables["scope"].get() == "Region"
        for widget in self.bounds_controls + [self.width_control]:
            widget.configure(state="normal" if regional else "disabled")

    def toggle_advanced(self):
        if self.advanced.winfo_ismapped():
            self.advanced.grid_remove()
            self.advanced_button.configure(text="Show advanced settings")
            self.root.minsize(880, 700)
        else:
            self.advanced.grid(row=5, column=0, sticky="ew", pady=(0, 10))
            self.advanced_button.configure(text="Hide advanced settings")
            self.root.minsize(880, 860)

    def browse_map(self):
        path = filedialog.askopenfilename(title="Choose a global input map", initialdir=str(ROOT / "human_input"),
                                          filetypes=[("Maps", "*.png *.tif *.tiff"), ("All files", "*.*")])
        if path:
            self.variables["dem"].set(path)

    def browse_checkpoint(self):
        path = filedialog.askopenfilename(title="Choose a meso checkpoint", initialdir=str(ROOT / "outputs/meso"),
                                          filetypes=[("Model checkpoint", "*.pt"), ("All files", "*.*")])
        if path:
            self.variables["checkpoint"].set(path)

    def browse_folder(self):
        path = filedialog.askdirectory(title="Choose a run folder", initialdir=str(ROOT / "outputs/meso"))
        if path:
            self.variables["folder"].set(path)

    def new_run(self):
        parent = resolve_path(self.variables["folder"].get()).parent
        name = datetime.now().strftime("planet-%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)
        self.variables["folder"].set(str(parent / name))
        self.variables["name"].set("")
        self.preview = None
        self.preview_button.configure(state="disabled")
        self.status.set("Fresh run selected. Current model and map settings kept.")

    def load(self):
        path = filedialog.askopenfilename(title="Load fixed meso settings", initialdir=str(ROOT / "outputs/meso"),
                                          filetypes=[("Settings", "*.json")])
        if not path:
            return
        try:
            values = load_settings(path)
            for key, value in values.items():
                self.variables[key].set(value)
            self.update_controls()
            self.status.set("Settings loaded. Choose the area and an unused output filename, then Generate / Resume.")
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            messagebox.showerror("Cannot load settings", str(exc))

    def append(self, line):
        self.log.configure(state="normal")
        self.log.insert("end", line if line.endswith("\n") else line + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def start(self):
        if self.busy:
            return
        values = {key: value.get() for key, value in self.variables.items()}
        self.busy = True
        self.stop_requested.clear()
        self.preview = None
        self.preview_button.configure(state="disabled")
        self.update_controls()
        self.stop_button.configure(state="normal")
        self.status.set("Checking input and saved tiles…")
        self.progress.configure(mode="indeterminate", value=0)
        self.progress.start(15)
        threading.Thread(target=self.run, args=(values,), daemon=True).start()

    def run(self, values):
        try:
            command, settings, target = prepare_command(values)
            folder = target.parent
            with self.process_lock:
                if self.stop_requested.is_set():
                    self.events.put(("done", None, target))
                    return
                folder.mkdir(parents=True, exist_ok=True)
                save_launch(folder / "meso-launch.json", settings, command)
                self.process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                                text=True, encoding="utf-8", errors="replace", bufsize=1,
                                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            self.events.put(("started", folder))
            with (folder / "generation.log").open("a", encoding="utf-8") as log:
                log.write(f"\n{datetime.now().isoformat(timespec='seconds')} — {target.name}\n")
                for line in self.process.stdout:
                    log.write(line)
                    log.flush()
                    self.events.put(("line", line))
            code = self.process.wait()
            self.events.put(("done", code, target))
        except Exception as exc:
            with self.process_lock:
                if self.process is not None and self.process.poll() is None:
                    self.process.terminate()
                    self.process.wait()
            self.events.put(("error", str(exc)))

    def poll(self):
        for _ in range(100):
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            kind = event[0]
            if kind == "started":
                self.run_folder = event[1]
                self.status.set("Preparing model and input map…")
                self.append(f"Run folder: {event[1]}")
            elif kind == "line":
                self.append(event[1])
                self.show_progress(event[1].strip())
            elif kind in ("done", "error"):
                self.progress.stop()
                self.busy = False
                self.process = None
                self.stop_button.configure(state="disabled")
                self.update_controls()
                if kind == "error":
                    self.progress.configure(mode="determinate", value=0)
                    self.status.set("Stopped." if self.stop_requested.is_set() else "Could not start generation.")
                    self.append(event[1])
                    if not self.stop_requested.is_set():
                        messagebox.showerror("Generation settings", event[1])
                elif event[1] == 0:
                    self.progress.configure(mode="determinate", maximum=100, value=100)
                    self.preview = event[2].with_suffix(".preview.png")
                    self.preview_button.configure(state="normal" if self.preview.exists() else "disabled")
                    self.status.set(f"Complete: {event[2].name}. Open preview to see the result.")
                elif self.stop_requested.is_set():
                    self.progress.configure(mode="determinate", value=0)
                    self.status.set("Stopped. Keep the same settings and folder, then Generate / Resume.")
                else:
                    self.progress.configure(mode="determinate", value=0)
                    self.status.set("Generation failed. See the message in the log below.")
        self.poll_id = self.root.after(100, self.poll)

    def show_progress(self, line):
        count = re.match(r"tiles (\d+)/(\d+)", line)
        if count:
            self.progress.stop()
            self.progress.configure(mode="determinate", maximum=int(count[2]), value=int(count[1]))
            self.status.set(line.replace("tiles ", "Generating tiles ", 1))
        elif line.startswith("tiles needed"):
            self.status.set(line)
        elif line.startswith("blending") or line.startswith("writing"):
            self.progress.configure(mode="indeterminate", value=0)
            self.progress.start(15)
            self.status.set("Combining finished tiles…" if line.startswith("blending") else "Exporting elevation and preview…")

    def cancel(self):
        if not self.busy:
            return
        self.stop_requested.set()
        with self.process_lock:
            if self.process is not None and self.process.poll() is None:
                try:
                    self.process.terminate()
                except ProcessLookupError:
                    pass
        self.stop_button.configure(state="disabled")
        self.status.set("Stopping. Finished tiles will be kept.")

    def open_path(self, path):
        if os.name == "nt":
            os.startfile(path)
        else:
            subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(path)])

    def open_folder(self):
        path = resolve_path(self.variables["folder"].get())
        if path.is_dir():
            self.open_path(path)
        else:
            messagebox.showinfo("Run folder", "The run folder will be created when generation starts.")

    def open_preview(self):
        if self.preview and self.preview.is_file():
            self.open_path(self.preview)

    def close(self):
        if self.busy:
            if not messagebox.askyesno("Stop generation?", "Stop this run and close? Finished tiles will be kept."):
                return
            self.cancel()
        self.root.after_cancel(self.poll_id)
        self.root.destroy()


def main():
    configure_tk()
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        text = f"Cannot open Meso Terrain: {exc}\nInstall Python with Tcl/Tk support."
        if os.name == "nt":
            import ctypes
            ctypes.windll.user32.MessageBoxW(None, text, "Meso Terrain", 16)
        else:
            print(text, file=sys.stderr)
        return 1
    Launcher(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
