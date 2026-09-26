"""Small desktop launcher. Generation runs in a separate, cancellable process."""
from datetime import datetime
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
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


def build_command(folder, seed, draft="", coarse_height=8, ocean_depth=0., white_metres=4000., device="cpu", refinement=.2):
    folder = Path(folder).expanduser().resolve()
    if (folder/"state").exists() or (folder/"planet-native.tif").exists():
        raise ValueError("This run already has output. Choose New run to preserve it.")
    if coarse_height < 4 or coarse_height % 4:
        raise ValueError("Resolution must use a coarse height divisible by four")
    if device not in ("cpu","cuda"):
        raise ValueError("Device must be cpu or cuda")
    resolved = resolve_seed(seed,folder/"checkpoints")
    cmd = [python_executable(),"-u","-m","planet_diffusion","generate",
           "--seed",str(resolved),"--state",str(folder/"state"),
           "--output",str(folder/"planet-native.tif"),"--checkpoint-dir",str(folder/"checkpoints"),
           "--coarse-height",str(coarse_height),"--device",device,"--upstream",str(ROOT/"upstream")]
    model = ROOT/"models"/"terrain-diffusion-90m"
    if model.is_dir():
        cmd += ["--model",str(model)]
    if draft.strip():
        from .draft import Draft
        draft_path = Path(draft).expanduser().resolve()
        Draft(draft_path,ocean_depth,white_metres,refinement)  # Fail before launching large models.
        cmd += ["--draft",str(draft_path),"--draft-ocean-depth",str(ocean_depth),
                "--draft-white-metres",str(white_metres),"--draft-refinement",str(refinement)]
    return cmd,resolved,folder


class Launcher:
    def __init__(self, root):
        self.root = root
        self.process = None
        self.events = queue.Queue()
        self.stopping = False
        self.controls = []
        root.title("Planet Terrain Diffusion")
        root.geometry("900x790")
        root.minsize(740,610)
        root.protocol("WM_DELETE_WINDOW",self.close)
        main = ttk.Frame(root,padding=16)
        main.pack(fill="both",expand=True)
        main.columnconfigure(1,weight=1)
        self.draft = tk.StringVar()
        self.seed = tk.StringVar(value="random")
        self.folder = tk.StringVar()
        self.resolution = tk.StringVar(value="8192 x 4096")
        self.refinement = tk.StringVar(value="0.2")
        self.ocean = tk.StringVar(value="0")
        self.white = tk.StringVar(value="4000")
        self.device = tk.StringVar(value="cpu")
        self.status = tk.StringVar(value="Choose a PNG draft, or leave it blank for a procedural planet.")
        ttk.Label(main,text="Generate a spherical elevation GeoTIFF",font=("Segoe UI",15)).grid(row=0,column=0,columnspan=3,sticky="w",pady=(0,12))
        self.entry(main,1,"Draft PNG (optional)",self.draft,"Browse…",self.browse_draft)
        ttk.Label(main,text="2:1 global map · north at top · black = ocean · lighter = higher land").grid(row=2,column=0,columnspan=3,sticky="w",pady=(0,12))
        self.entry(main,3,"Seed",self.seed,"Random",self.random_seed)
        self.entry(main,4,"Run folder",self.folder,"Browse…",self.browse_folder)
        button = ttk.Button(main,text="New run",command=self.new_run)
        button.grid(row=5,column=2,sticky="e",pady=(0,8)); self.controls.append(button)
        ttk.Label(main,text="Global resolution").grid(row=6,column=0,sticky="w")
        resolution = ttk.Combobox(main,textvariable=self.resolution,state="readonly",
                                  values=["2048 x 1024","4096 x 2048","8192 x 4096","16384 x 8192"])
        resolution.grid(row=6,column=1,sticky="ew",padx=8,pady=4); self.controls.append(resolution)
        self.entry(main,7,"Ocean depth hint (0 = auto)",self.ocean)
        self.entry(main,8,"White land elevation (m)",self.white)
        self.entry(main,9,"Elevation refinement (0.01–4)",self.refinement)
        ttk.Label(main,text="Default 0.2 · smaller values follow the draft more closely; larger values allow more change.").grid(row=10,column=0,columnspan=3,sticky="w",pady=(4,8))
        ttk.Label(main,text="Compute device").grid(row=11,column=0,sticky="w")
        device = ttk.Combobox(main,textvariable=self.device,values=["cpu","cuda"],state="readonly")
        device.grid(row=11,column=1,sticky="ew",padx=8); self.controls.append(device)
        ttk.Label(main,text="8k/16k CPU runs can take tens of minutes or longer. CUDA needs a CUDA-enabled PyTorch install.").grid(row=12,column=0,columnspan=3,sticky="w",pady=8)
        actions = ttk.Frame(main)
        actions.grid(row=13,column=0,columnspan=3,sticky="ew",pady=8)
        self.generate = ttk.Button(actions,text="Generate / Resume",command=self.start)
        self.generate.pack(side="left"); self.controls.append(self.generate)
        self.stop = ttk.Button(actions,text="Stop",command=self.cancel,state="disabled")
        self.stop.pack(side="left",padx=8)
        ttk.Button(actions,text="Open run folder",command=self.open_folder).pack(side="right")
        ttk.Label(main,textvariable=self.status,wraplength=790).grid(row=14,column=0,columnspan=3,sticky="w",pady=6)
        logframe = ttk.Frame(main)
        logframe.grid(row=15,column=0,columnspan=3,sticky="nsew")
        main.rowconfigure(15,weight=1)
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

    def browse_folder(self):
        path = filedialog.askdirectory(title="Choose an empty run folder, or an interrupted run")
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

    def start(self):
        try:
            if not self.folder.get().strip():
                raise ValueError("Choose a run folder")
            command,seed,folder = build_command(self.folder.get(),self.seed.get(),self.draft.get(),
                int(self.resolution.get().split(" x ")[1])//256,float(self.ocean.get()),float(self.white.get()),self.device.get(),float(self.refinement.get()))
            folder.mkdir(parents=True,exist_ok=True)
            self.seed.set(str(seed))
            self.append(f"\nSeed: {seed}\nRun folder: {folder}\n")
            (folder/"launch.json").write_text(json.dumps({"seed":seed,"command":command},indent=2)+"\n")
            self.process = subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                text=True,encoding="utf-8",errors="replace",creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0),
                env={**os.environ,"PYTHONIOENCODING":"utf-8"})
        except (OSError,ValueError) as exc:
            messagebox.showerror("Cannot start generation",str(exc)); return
        self.stopping = False
        self.busy(True)
        self.status.set(f"Generating with seed {seed}. Progress is saved for resuming.")
        threading.Thread(target=self.read_worker,args=(self.process,folder),daemon=True).start()

    def read_worker(self,process,folder):
        try:
            with (folder/"generation.log").open("a",encoding="utf-8") as log:
                for line in process.stdout:
                    log.write(line); log.flush(); self.events.put(("log",line))
            code = process.wait()
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
                    self.status.set("Stopped. Use the same settings and folder to resume." if self.stopping else
                        ("Complete: planet-native.tif is ready in the run folder." if value == 0 else
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
