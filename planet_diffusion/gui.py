"""Small desktop launcher. Generation runs in a separate, cancellable process."""
from datetime import datetime
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from .model_presets import MODEL_30M, MODEL_90M, default_revision, local_checkpoint, sparse_guide_height
from .seeds import resolve_seed

ROOT = Path(__file__).resolve().parents[1]


def configure_tk():
    # Some bundled interpreters cannot resolve their external Tcl script path.
    # A local copy in the virtual environment takes precedence when installed.
    libraries = Path(sys.prefix)/"tcl"
    for name,folder,file in (("TCL_LIBRARY",f"tcl{tk.TclVersion}","init.tcl"),
                             ("TK_LIBRARY",f"tk{tk.TkVersion}","tk.tcl")):
        path = libraries/folder
        if (path/file).is_file():
            os.environ.setdefault(name,str(path))


def python_executable():
    path = Path(sys.executable)
    # pythonw is suitable for the window; a console interpreter supplies worker logs.
    console = path.with_name("python.exe")
    return str(console if path.name.lower() == "pythonw.exe" and console.exists() else path)


def build_command(folder, seed, draft="", coarse_height=8, ocean_depth=0., white_metres=6250., device="cpu", refinement=.2, radius_metres=6371000., export_height=None, bounds=None, export_width=None, latent_batch_size=1, export_climate=False, conditioning_dir="", conditioning_snr=".2,.2,1,.2,1", regional_only=False, model_choice="90 m"):
    folder = Path(folder).expanduser().resolve()
    if (folder/"state").exists() or (folder/"planet-native.tif").exists():
        raise ValueError("This run already has output. Choose New run to preserve it.")
    if export_climate and (folder/"planet-climate.tif").exists():
        raise ValueError("This run already has climate output. Choose New run to preserve it.")
    if coarse_height < 4 or coarse_height % 4:
        raise ValueError("Resolution must use a coarse height divisible by four")
    if not math.isfinite(radius_metres) or radius_metres <= 0:
        raise ValueError("Radius metres must be positive and finite")
    if regional_only and bounds is None:
        raise ValueError("Regional-only guides require regional bounds")
    if bounds is not None:
        from .region import validate_region
        validate_region(bounds,export_width,export_height,coarse_height*256)
    elif export_height is not None and not 2 <= export_height <= coarse_height*256:
        raise ValueError("Export height must be between 2 and coarse height × 256")
    elif export_width is not None and export_width != 2*(export_height or coarse_height*256):
        raise ValueError("Whole-globe output must have width = 2 × height")
    if device not in ("cpu","cuda"):
        raise ValueError("Device must be cpu or cuda")
    if isinstance(latent_batch_size,bool) or not isinstance(latent_batch_size,int) or latent_batch_size < 1:
        raise ValueError("Latent batch size must be a positive integer")
    resolved = resolve_seed(seed,folder/"checkpoints")
    cmd = [python_executable(),"-u","-m","planet_diffusion","generate",
           "--seed",str(resolved),"--state",str(folder/"state"),
           "--output",str(folder/"planet-native.tif"),"--checkpoint-dir",str(folder/"checkpoints"),
           "--coarse-height",str(coarse_height),"--radius-metres",str(radius_metres),
           "--device",device,"--upstream",str(ROOT/"upstream"),
           "--latent-batch-size",str(latent_batch_size)]
    if export_climate:
        cmd += ["--climate-output",str(folder/"planet-climate.tif")]
    if export_height is not None:
        cmd += ["--height",str(export_height)]
    if bounds is not None:
        cmd += ["--bounds",*[str(v) for v in bounds],"--width",str(export_width)]
    if regional_only:
        cmd += ["--regional-only"]
    if model_choice not in ("90 m", "30 m"):
        raise ValueError("Choose a 30 m or 90 m model")
    model_id = MODEL_30M if model_choice == "30 m" else MODEL_90M
    model = local_checkpoint(ROOT/"models",model_id) or model_id
    cmd += ["--model",str(model),"--revision",default_revision(model_id)]
    if conditioning_dir.strip():
        if draft.strip():
            raise ValueError("Choose either a PNG draft or a conditioning TIFF folder")
        from .conditioning import TiffConditioning
        source = Path(conditioning_dir).expanduser().resolve()
        TiffConditioning(source,conditioning_snr)
        cmd += ["--conditioning-dir",str(source),"--snr",conditioning_snr]
    if draft.strip():
        from .draft import Draft
        draft_path = Path(draft).expanduser().resolve()
        Draft(draft_path,ocean_depth,white_metres,refinement)  # Fail before launching large models.
        cmd += ["--draft",str(draft_path),"--draft-ocean-depth",str(ocean_depth),
                "--draft-white-metres",str(white_metres),"--draft-refinement",str(refinement)]
    return cmd,resolved,folder


def build_query_command(folder, bounds, width, height, export_climate=False, stamp=None):
    """Build a request for another region of a completed desktop or CLI run."""
    from .query import load_query
    from .region import validate_region

    folder = Path(folder).expanduser().resolve()
    if (folder/"state"/"planet.json").is_file():
        state, output_folder = folder/"state", folder
    elif (folder/"planet.json").is_file():
        state, output_folder = folder, folder.parent
    else:
        raise ValueError("Choose a completed run folder or its state folder")
    directory, _, metadata, _, _ = load_query(state, with_climate=export_climate)
    validate_region(bounds,width,height,metadata["native_height"])
    stamp = stamp or datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    output = output_folder/(f"region-{stamp}.tif")
    climate = output_folder/(f"region-{stamp}-climate.tif")
    if output.exists() or export_climate and climate.exists():
        raise ValueError("Query output already exists; choose a new output name")
    command = [python_executable(),"-u","-m","planet_diffusion","query",
               "--state",str(state),"--checkpoint-dir",str(directory),
               "--bounds",*[str(value) for value in bounds],
               "--width",str(width),"--height",str(height),
               "--output",str(output),"--upstream",str(ROOT/"upstream")]
    if export_climate:
        command += ["--climate-output",str(climate)]
    return command, output_folder, output


class Launcher:
    def __init__(self, root):
        self.root = root
        self.process = None
        self.events = queue.Queue()
        self.stopping = False
        self.controls = []
        root.title("Planet Terrain Diffusion")
        root.geometry("940x970")
        root.minsize(850,800)
        root.protocol("WM_DELETE_WINDOW",self.close)
        main = ttk.Frame(root,padding=16)
        main.pack(fill="both",expand=True)
        main.columnconfigure(1,weight=1)
        self.draft = tk.StringVar()
        self.conditioning_dir = tk.StringVar()
        self.climate_refinement = tk.StringVar(value=".2,1,.2,1")
        self.seed = tk.StringVar(value="random")
        self.folder = tk.StringVar()
        self.coarse_height = tk.StringVar(value="16")
        self.export_width = tk.StringVar(value="8192")
        self.export_height = tk.StringVar(value="4096")
        self.scope = tk.StringVar(value="Whole globe")
        self.bounds = [tk.StringVar(value=v) for v in ("-30","-30","30","30")]
        self.radius_metres = tk.StringVar(value="6371000")
        self.refinement = tk.StringVar(value="0.2")
        self.ocean = tk.StringVar(value="0")
        self.white = tk.StringVar(value="6250")
        self.device = tk.StringVar(value="cpu")
        self.model = tk.StringVar(value="90 m")
        self.latent_batch_size = tk.StringVar(value="1")
        self.export_climate = tk.BooleanVar(value=False)
        self.regional_only = tk.BooleanVar(value=True)
        self.status = tk.StringVar(value="Choose a PNG draft or TIFF folder, or leave both blank for a procedural planet.")
        ttk.Label(main,text="Generate or query spherical elevation",font=("Segoe UI",15)).grid(row=0,column=0,columnspan=3,sticky="w",pady=(0,12))
        self.entry(main,1,"Draft PNG (optional)",self.draft,"Browse…",self.browse_draft)
        ttk.Label(main,text="2:1 global map · north at top · black = ocean · lighter = higher land").grid(row=2,column=0,columnspan=3,sticky="w",pady=(0,12))
        self.entry(main,3,"Seed",self.seed,"Random",self.random_seed)
        self.entry(main,4,"Run folder",self.folder,"Browse…",self.browse_folder)
        climate = ttk.Checkbutton(main,text="Export climate maps",variable=self.export_climate)
        climate.grid(row=5,column=1,sticky="w",padx=8); self.controls.append(climate)
        button = ttk.Button(main,text="New run",command=self.new_run)
        button.grid(row=5,column=2,sticky="e",pady=(0,8)); self.controls.append(button)
        self.entry(main,6,"Logical guide height (multiple of 4)",self.coarse_height)
        self.entry(main,7,"Radius (metres)",self.radius_metres)
        ttk.Label(main,text="Output resolution (pixels)").grid(row=8,column=0,sticky="w")
        resolution = ttk.Frame(main)
        resolution.grid(row=8,column=1,sticky="ew",padx=8,pady=4)
        for label,variable in (("Width",self.export_width),("Height",self.export_height)):
            ttk.Label(resolution,text=label).pack(side="left",padx=(0,6))
            widget = ttk.Entry(resolution,textvariable=variable,width=12)
            widget.pack(side="left",padx=(0,12)); self.controls.append(widget)
        self.entry(main,9,"Ocean depth hint (0 = auto)",self.ocean)
        self.entry(main,10,"White land elevation (m)",self.white)
        self.entry(main,11,"Elevation refinement (0.01–4)",self.refinement)
        ttk.Label(main,text="For PNG or TIFF elevation · smaller values follow the input more closely; larger values allow more change.").grid(row=12,column=0,columnspan=3,sticky="w",pady=(4,8))
        ttk.Label(main,text="Compute device").grid(row=13,column=0,sticky="w")
        device = ttk.Combobox(main,textvariable=self.device,values=["cpu","cuda"],state="readonly")
        device.grid(row=13,column=1,sticky="ew",padx=8); self.controls.append(device)
        ttk.Label(main,text="Terrain model").grid(row=14,column=0,sticky="w")
        model = ttk.Combobox(main,textvariable=self.model,values=["90 m","30 m"],state="readonly")
        model.grid(row=14,column=1,sticky="ew",padx=8); self.controls.append(model)
        model.bind("<<ComboboxSelected>>",self.change_model)
        self.entry(main,15,"Latent batch size (larger uses more memory)",self.latent_batch_size)
        ttk.Label(main,text="Generation area / saved query").grid(row=16,column=0,sticky="w")
        scope = ttk.Combobox(main,textvariable=self.scope,values=["Whole globe","Region","Saved run query"],state="readonly")
        scope.grid(row=16,column=1,sticky="ew",padx=8); self.controls.append(scope)
        scope.bind("<<ComboboxSelected>>",self.change_scope)
        ttk.Label(main,text="Regional bounds (degrees)").grid(row=17,column=0,sticky="w")
        bounds = ttk.Frame(main)
        bounds.grid(row=17,column=1,columnspan=2,sticky="ew",padx=8,pady=6)
        self.bound_controls = []
        for label,variable in zip(("West","South","East","North"),self.bounds):
            ttk.Label(bounds,text=label).pack(side="left",padx=(0,4))
            widget = ttk.Entry(bounds,textvariable=variable,width=7,state="disabled")
            widget.pack(side="left",padx=(0,8)); self.controls.append(widget); self.bound_controls.append(widget)
        ttk.Label(main,text="East < west crosses the date line. Latitude: −90 to 90. Drafts always cover the globe.").grid(row=18,column=0,columnspan=3,sticky="w",pady=(0,6))
        self.entry(main,19,"Conditioning TIFF folder (instead of PNG)",self.conditioning_dir,"Browse...",self.browse_conditioning)
        self.entry(main,20,"TIFF climate refinement: temp, T std, precip, P CV",self.climate_refinement)
        actions = ttk.Frame(main)
        actions.grid(row=21,column=0,columnspan=3,sticky="ew",pady=8)
        self.generate = ttk.Button(actions,text="Generate / Resume",command=self.start)
        self.generate.pack(side="left"); self.controls.append(self.generate)
        sparse = ttk.Checkbutton(actions,text="Only generate requested region",variable=self.regional_only,
                                command=self.change_regional_mode)
        sparse.pack(side="left",padx=8); self.controls.append(sparse)
        self.stop = ttk.Button(actions,text="Stop",command=self.cancel,state="disabled")
        self.stop.pack(side="left",padx=8)
        ttk.Button(actions,text="Open run folder",command=self.open_folder).pack(side="right")
        ttk.Label(main,textvariable=self.status,wraplength=850).grid(row=22,column=0,columnspan=3,sticky="w",pady=6)
        logframe = ttk.Frame(main)
        logframe.grid(row=23,column=0,columnspan=3,sticky="nsew")
        main.rowconfigure(23,weight=1)
        self.log = tk.Text(logframe,height=12,wrap="word",state="disabled")
        scroll = ttk.Scrollbar(logframe,command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right",fill="y"); self.log.pack(fill="both",expand=True)
        self.new_run()
        root.after(100,self.poll)

    def entry(self,parent,row,label,variable,button=None,command=None):
        ttk.Label(parent,text=label).grid(row=row,column=0,sticky="w",pady=4)
        entry = ttk.Entry(parent,textvariable=variable)
        entry.grid(row=row,column=1,sticky="ew",padx=8,pady=4)
        self.controls.append(entry)
        if button:
            widget = ttk.Button(parent,text=button,command=command)
            widget.grid(row=row,column=2,sticky="e"); self.controls.append(widget)

    def browse_draft(self):
        path = filedialog.askopenfilename(title="Choose a global PNG draft",filetypes=[("PNG image","*.png")])
        if path: self.draft.set(path)

    def browse_conditioning(self):
        path = filedialog.askdirectory(title="Choose global conditioning TIFF folder")
        if path: self.conditioning_dir.set(path)

    def browse_folder(self):
        path = filedialog.askdirectory(title="Choose a run folder, including a completed run for queries")
        if path: self.folder.set(path)

    def new_run(self):
        self.folder.set(str(ROOT/"outputs"/("planet-"+datetime.now().strftime("%Y%m%d-%H%M%S-%f"))))

    def random_seed(self):
        self.seed.set(str(resolve_seed()))

    def append(self,text):
        self.log.configure(state="normal")
        self.log.insert("end",text); self.log.see("end")
        self.log.configure(state="disabled")

    def busy(self,value):
        for widget in self.controls:
            widget.configure(state="disabled" if value else ("readonly" if isinstance(widget,ttk.Combobox) else "normal"))
        self.stop.configure(state="normal" if value else "disabled")
        if not value:
            for widget in self.bound_controls:
                widget.configure(state="normal" if self.scope.get() != "Whole globe" else "disabled")

    def change_scope(self,event=None):
        regional = self.scope.get() != "Whole globe"
        for widget in self.bound_controls:
            widget.configure(state="normal" if regional else "disabled")
        self.generate.configure(text="Query saved run" if self.scope.get() == "Saved run query" else "Generate / Resume")
        if self.scope.get() == "Saved run query":
            self.status.set("Choose a completed run folder, enter bounds and output resolution, then query its saved guides.")
        elif self.scope.get() == "Region" and self.regional_only.get():
            if self.coarse_height.get() == "16":
                self.coarse_height.set(str(self.preferred_sparse_height()))
            if [v.get() for v in self.bounds] == ["-30","-30","30","30"]:
                for variable,value in zip(self.bounds,("-2","-2","2","2")):
                    variable.set(value)
            self.status.set("Sparse regional guides use a fixed global cell grid; only nearby model tiles are computed.")
        elif self.coarse_height.get() in ("1024","2560"):
            self.coarse_height.set("16")
        coarse = int(self.coarse_height.get()) if self.coarse_height.get().isdigit() else 16
        self.export_width.set("512" if regional else str(coarse*512))
        self.export_height.set("512" if regional else str(coarse*256))

    def change_regional_mode(self):
        if self.scope.get() != "Region":
            return
        if self.regional_only.get() and self.coarse_height.get() == "16":
            self.coarse_height.set(str(self.preferred_sparse_height()))
        elif not self.regional_only.get() and self.coarse_height.get() in ("1024","2560"):
            self.coarse_height.set("16")

    def preferred_sparse_height(self):
        return sparse_guide_height(MODEL_30M if self.model.get() == "30 m" else MODEL_90M)

    def change_model(self,event=None):
        if (self.scope.get() == "Region" and self.regional_only.get()
                and self.coarse_height.get() in ("1024","2560")):
            self.coarse_height.set(str(self.preferred_sparse_height()))

    def start(self):
        try:
            if not self.folder.get().strip():
                raise ValueError("Choose a run folder")
            if self.scope.get() == "Saved run query":
                command,folder,output = build_query_command(
                    self.folder.get(),[float(v.get()) for v in self.bounds],
                    int(self.export_width.get()),int(self.export_height.get()),
                    export_climate=self.export_climate.get())
                self.append(f"\nQuerying saved run: {folder}\nOutput: {output}\n")
                self._launch(command,folder,"query",f"query-{output.stem}.log")
                return
            command,seed,folder = build_command(self.folder.get(),self.seed.get(),self.draft.get(),
                int(self.coarse_height.get()),float(self.ocean.get()),float(self.white.get()),self.device.get(),
                float(self.refinement.get()),float(self.radius_metres.get()),
                int(self.export_height.get()),
                bounds=[float(v.get()) for v in self.bounds] if self.scope.get() == "Region" else None,
                export_width=int(self.export_width.get()),latent_batch_size=int(self.latent_batch_size.get()),
                export_climate=self.export_climate.get(),
                conditioning_dir=self.conditioning_dir.get(),
                conditioning_snr=self.refinement.get()+","+self.climate_refinement.get(),
                regional_only=self.regional_only.get() and self.scope.get() == "Region",
                model_choice=self.model.get())
            folder.mkdir(parents=True,exist_ok=True)
            self.seed.set(str(seed))
            self.append(f"\nSeed: {seed}\nRun folder: {folder}\n")
            (folder/"launch.json").write_text(json.dumps({"seed":seed,"command":command},indent=2)+"\n")
            self._launch(command,folder,"generation","generation.log",seed=seed)
        except (OSError,ValueError) as exc:
            messagebox.showerror("Cannot start task",str(exc)); return

    def _launch(self,command,folder,kind,log_name,seed=None):
        self.process = subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                text=True,encoding="utf-8",errors="replace",creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0),
                env={**os.environ,"PYTHONIOENCODING":"utf-8"})
        self.stopping = False
        self.current_action = kind
        started = time.perf_counter()
        self.busy(True)
        self.status.set(f"Generating with seed {seed}. Progress is saved for resuming." if kind == "generation"
                        else "Querying the saved run. Decoder predictions are cached for later requests.")
        threading.Thread(target=self.read_worker,args=(self.process,folder/log_name,started),daemon=True).start()

    def read_worker(self,process,log_path,started):
        try:
            with log_path.open("a",encoding="utf-8") as log:
                for line in process.stdout:
                    log.write(line); log.flush(); self.events.put(("log",line))
            code = process.wait()
            elapsed = time.perf_counter() - started
            outcome = "Complete" if code == 0 else ("Stopped" if self.stopping else "Failed")
            line = f"{outcome} in {elapsed:.1f} seconds.\n"
            with log_path.open("a",encoding="utf-8") as log:
                log.write(line)
            self.events.put(("log",line))
            self.events.put(("done",code))
        except OSError as exc:
            self.events.put(("log",f"Log error: {exc}\n"))
            if process.poll() is None: process.terminate()
            self.events.put(("done",process.wait()))
        finally:
            process.stdout.close()

    def poll(self):
        try:
            while True:
                kind,value = self.events.get_nowait()
                if kind == "log": self.append(value)
                else:
                    self.process = None; self.busy(False)
                    if self.current_action == "query":
                        self.status.set("Query stopped. Completed decoder predictions were kept." if self.stopping else
                            ("Query complete: regional GeoTIFFs are ready in the run folder." if value == 0 else
                             "Query failed. See the log above; completed predictions were kept."))
                    else:
                        self.status.set("Stopped. Use the same settings and folder to resume." if self.stopping else
                            ("Complete: exported GeoTIFFs are ready in the run folder." if value == 0 else
                             "Generation failed. See the log above; completed checkpoints were kept."))
        except queue.Empty:
            pass
        self.root.after(100,self.poll)

    def cancel(self):
        if self.process is not None and self.process.poll() is None:
            self.stopping = True; self.process.terminate()
            self.stop.configure(state="disabled")
            self.status.set("Stopping; completed checkpoints will be kept.")

    def open_folder(self):
        path = Path(self.folder.get())
        if not path.is_dir():
            messagebox.showinfo("Run folder","The folder will be created when generation starts."); return
        if self.scope.get() == "Saved run query" and (path/"planet.json").is_file():
            path = path.parent
        if os.name == "nt": os.startfile(path)
        else: subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open",str(path)])

    def close(self):
        if self.process is not None and self.process.poll() is None:
            if not messagebox.askyesno("Stop generation?","Stop this run and close? Completed checkpoints will be kept."):
                return
            self.cancel()
        self.root.destroy()


def main():
    configure_tk()
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        text = f"Cannot start the desktop launcher: {exc}\nInstall Python with Tcl/Tk support. The command-line workflow remains available."
        if os.name == "nt":
            import ctypes
            ctypes.windll.user32.MessageBoxW(None,text,"Planet Terrain Diffusion",16)
        else:
            print(text,file=sys.stderr)
        return 1
    Launcher(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
