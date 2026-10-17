"""Desktop continent generation with optional land coverage and relief controls."""
from datetime import datetime
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog,messagebox,ttk
from toy_meso.gui import configure_tk,python_executable

ROOT=Path(__file__).resolve().parents[1]


def selected_checkpoint():
    selection=ROOT/'outputs/continents/selection.json'
    if selection.exists():
        checkpoint=Path(json.loads(selection.read_text())['checkpoint'])
        if not checkpoint.is_absolute():checkpoint=ROOT/checkpoint
        if checkpoint.is_file():return checkpoint
    return ROOT/'outputs/continents/run1/best.pt'


def new_folder():
    return ROOT/'outputs/continents'/datetime.now().strftime('planet-%Y%m%d-%H%M%S-%f')


def prepare_command(values):
    def path(value):
        p=Path(value).expanduser()
        return (p if p.is_absolute() else ROOT/p).resolve()
    checkpoint,out=path(values['checkpoint']),path(values['out'])
    if not checkpoint.is_file():raise ValueError('Choose a continent model checkpoint.')
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError('Choose a new empty output folder to preserve earlier planets.')
    try:seed=int(values['seed'])
    except ValueError:raise ValueError('Seed must be a whole number.') from None
    if not 0<=seed<=2**63-1:raise ValueError('Seed must be between 0 and 9223372036854775807.')
    try:
        percent=float(values['land_percent']) if values['land_percent'].strip() else None
        relief=float(values['land_relief'])
    except ValueError:raise ValueError('Land coverage and relief must be numbers.') from None
    if percent is not None and (not math.isfinite(percent) or not 0<percent<100):
        raise ValueError('Land coverage must be between 0 and 100 percent, or blank for the model’s natural result.')
    if not math.isfinite(relief) or relief<=0:raise ValueError('Relief must be a positive multiplier; 1 keeps the model’s height.')
    command=[python_executable(),'-u','-m','planet_training.generate','--run',str(checkpoint.parent),
             '--checkpoint',checkpoint.name,'--out',str(out),'--seed',str(seed),'--steps','64',
             '--land-relief',str(relief),'--device','auto']
    if percent is not None:command.extend(('--land-percent',str(percent)))
    return command,out


class Launcher:
    def __init__(self,root):
        self.root=root;self.process=None;self.photo=None;self.output=None;self.events=queue.Queue()
        root.title('Continents');root.geometry('960x790');root.minsize(850,700)
        self.values={k:tk.StringVar(value=v) for k,v in dict(checkpoint=str(selected_checkpoint()),
                     out=str(new_folder()),seed='42',land_percent='',land_relief='1').items()}
        box=ttk.Frame(root,padding=16);box.pack(fill='both',expand=True);box.columnconfigure(1,weight=1)
        ttk.Label(box,text='Generate a continent layout',font=('Segoe UI',16,'bold')).grid(row=0,column=0,columnspan=3,sticky='w',pady=(0,14))
        for row,key,label in [(1,'checkpoint','Continent model'),(2,'out','Output folder'),(3,'seed','Seed'),
                               (4,'land_percent','Sea-level land coverage (%)'),(5,'land_relief','Land relief multiplier')]:
            ttk.Label(box,text=label).grid(row=row,column=0,sticky='w',padx=(0,12),pady=5)
            ttk.Entry(box,textvariable=self.values[key]).grid(row=row,column=1,sticky='ew',pady=5)
            if key in ('checkpoint','out'):
                ttk.Button(box,text='Browse…',command=lambda k=key:self.browse(k)).grid(row=row,column=2,padx=(8,0))
        ttk.Label(box,text='Recommended: leave coverage blank and relief at 1. Coverage lowers/raises sea level and can expose shelves; it does not grow new continents.',
                  wraplength=860).grid(row=6,column=0,columnspan=3,sticky='w',pady=10)
        actions=ttk.Frame(box);actions.grid(row=7,column=0,columnspan=3,sticky='w',pady=5)
        self.generate_button=ttk.Button(actions,text='Generate planet',command=self.generate);self.generate_button.pack(side='left')
        ttk.Button(actions,text='New output folder',command=lambda:self.values['out'].set(str(new_folder()))).pack(side='left',padx=8)
        ttk.Button(actions,text='Open result folder',command=self.open_result).pack(side='left')
        self.status=tk.StringVar(value='Ready. The layout feeds the intermediate stage; finer terrain is generated later.')
        ttk.Label(box,textvariable=self.status,wraplength=860).grid(row=8,column=0,columnspan=3,sticky='w',pady=10)
        self.preview=ttk.Label(box,anchor='center');self.preview.grid(row=9,column=0,columnspan=3,sticky='nsew')
        box.rowconfigure(9,weight=1)
        self.poll_id=root.after(150,self.poll);root.protocol('WM_DELETE_WINDOW',self.close)

    def browse(self,key):
        selected=(filedialog.askopenfilename(title='Choose continent model',initialdir=str(ROOT/'outputs/continents'),filetypes=[('Model checkpoint','*.pt')])
                  if key=='checkpoint' else filedialog.askdirectory(title='Choose a new empty output folder',initialdir=str(ROOT/'outputs/continents')))
        if selected:self.values[key].set(selected)

    def generate(self):
        if self.process is not None:return
        try:command,out=prepare_command({k:v.get() for k,v in self.values.items()})
        except ValueError as exc:messagebox.showerror('Generation settings',str(exc));return
        self.generate_button.configure(state='disabled');self.status.set('Generating continent layout…')
        try:
            self.process=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                text=True,encoding='utf-8',errors='replace',creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        except OSError as exc:
            self.generate_button.configure(state='normal');messagebox.showerror('Cannot start generation',str(exc));return
        process=self.process
        def worker():
            lines=[]
            for line in process.stdout:lines.append(line)
            self.events.put((process.wait(),out,''.join(lines)))
        threading.Thread(target=worker,daemon=True).start()

    def poll(self):
        try:
            code,out,log=self.events.get_nowait();self.process=None;self.generate_button.configure(state='normal')
            if code:
                self.status.set('Generation failed. Earlier results are preserved.')
                messagebox.showerror('Generation failed',log[-3000:])
            else:
                self.output=out
                from PIL import Image,ImageTk
                image=Image.open(out/'layout-preview.png');image.thumbnail((880,420))
                self.photo=ImageTk.PhotoImage(image);self.preview.configure(image=self.photo)
                meta=json.loads((out/'layout.json').read_text());coverage=meta['controls']['achieved_land_percent']
                self.status.set(f'Ready: {coverage:.1f}% land. layout.tif is the input for the intermediate stage. Saved in {out}')
        except queue.Empty:pass
        self.poll_id=self.root.after(150,self.poll)

    def open_result(self):
        out=self.output or Path(self.values['out'].get())
        if out.is_dir():os.startfile(out)

    def close(self):
        if self.process is not None:self.process.terminate()
        self.root.after_cancel(self.poll_id);self.root.destroy()


def main():
    configure_tk();root=tk.Tk();Launcher(root);root.mainloop()


if __name__=='__main__':main()
