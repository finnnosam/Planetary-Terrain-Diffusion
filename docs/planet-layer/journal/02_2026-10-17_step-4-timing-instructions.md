### Step 4: global timing run (your machine)

```powershell
Set-Location 'E:\projects\Infinite_Diffusion'
& '.\.venv\Scripts\python.exe' -m pip install ducc0
& '.\.venv\Scripts\python.exe' -m pytest -q tests\test_sphere_graph.py tests\test_graph_landscape.py
& '.\.venv\Scripts\python.exe' -u -m scripts.planet_landscape_timing
```

**Planet Landscape Timing.cmd** does the install and the run. Output goes to
`outputs\planet-landscape-timing\<timestamp>\`: `timing.json` and `preview.png`.

