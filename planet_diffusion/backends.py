"""Pinned upstream model adapter; the diagnostic backend never claims learned terrain."""
import json
from pathlib import Path
import sys
import numpy as np

UPSTREAM_COMMIT = "e8dcb4b1a834ab2f6b1a6f5256ed7c9f2f3e8230"
MODEL_REVISION = "2bb1a93141140040091d44f0770d4ba22c8cf145"
MEANS = np.array([-37.67916460232751, 2.22578822145657, 18.030293275011356,
                  333.8442390481231, 1350.1259248456176, 52.444339366764396], np.float32)
STDS = np.array([39.68515115440358, 3.0981253981231522, 8.940333096712806,
                 322.25238547630295, 856.3430083394657, 30.982620765341043], np.float32)


class DiagnosticBackend:
    """Fast deterministic stand-in for testing the spatial solver and file formats."""
    name = "diagnostic-NOT-TERRAIN-DIFFUSION"
    residual_mean, residual_std = 0., 1.1678
    means, stds = MEANS, STDS
    cond_snr = np.array([.3, .1, 1., .1, 1.], np.float32)
    metadata = {"backend": name, "model_revision": "none"}

    def start_coarse(self, steps):
        return 1.

    def coarse_predict(self, a, cond, step):
        guide = np.concatenate([cond[:1], np.zeros_like(cond[:1]), cond[1:]])
        return .3*a + .7*guide

    def coarse_advance(self, prediction, state, step):
        return prediction

    def coarse_finish(self, state):
        return state

    def predict(self, stage, a, cond, t):
        # Spatial context is intentionally used so broken halo reads affect tests.
        base = (a + np.roll(a, 1, -1) + np.roll(a, 1, -2))/3
        if stage == "latent":
            base = base.copy()
            base[4] += float(cond[0].mean())/40
            return .25*base
        return .2*base[:1] + .1*cond[:1]


class TerrainBackend:
    name = "terrain-diffusion"

    def __init__(self, upstream, model, revision, device="cpu"):
        import os
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch
        source = Path(upstream).resolve()
        if not (source / "terrain_diffusion/models/edm_unet.py").is_file():
            raise ValueError("--upstream must point to the pinned terrain-diffusion checkout")
        import subprocess
        git = ["git", "-c", f"safe.directory={source.as_posix()}", "-C", str(source)]
        actual = subprocess.check_output(git+["rev-parse", "HEAD"], text=True).strip()
        if actual != UPSTREAM_COMMIT:
            raise ValueError(f"Upstream must be pinned at {UPSTREAM_COMMIT}; found {actual}")
        if subprocess.run(git+["diff", "--quiet", "HEAD", "--", "terrain_diffusion/models", "terrain_diffusion/scheduler"]).returncode:
            raise ValueError("Upstream model/scheduler source has local modifications")
        # Avoid the heavy WorldPipeline imports; only load model and scheduler modules.
        sys.path.insert(0, str(source))
        from terrain_diffusion.models.edm_unet import EDMUnet2D
        from terrain_diffusion.models.mp_layers import mp_concat
        from terrain_diffusion.scheduler.dpmsolver import EDMDPMSolverMultistepScheduler
        from huggingface_hub import snapshot_download
        if Path(model).is_dir():
            model_dir = Path(model).resolve()
            import hashlib
            digest = hashlib.sha256()
            for file in sorted(model_dir.rglob("*")):
                if file.is_file() and file.suffix in (".json", ".safetensors", ".bin"):
                    digest.update(file.relative_to(model_dir).as_posix().encode())
                    with file.open("rb") as f:
                        for chunk in iter(lambda: f.read(1024*1024), b""):
                            digest.update(chunk)
            resolved_revision = "sha256:"+digest.hexdigest()
        else:
            model_dir = Path(snapshot_download(model, revision=revision,
                allow_patterns=["config.json", "coarse_model/*", "base_model/*", "decoder_model/*"]))
            resolved_revision = model_dir.name
        config = json.loads((model_dir/"config.json").read_text())
        if config.get("latent_compression", 8) != 8 or config.get("coarse_pooling", 1) != 1:
            raise ValueError("This adapter requires latent_compression=8 and coarse_pooling=1")
        self.means = np.asarray(config.get("coarse_means") or MEANS, np.float32)
        self.stds = np.asarray(config.get("coarse_stds") or STDS, np.float32)
        self.residual_mean = config.get("residual_mean", 0.)
        self.residual_std = config.get("residual_std", 1.1678)
        self.cond_snr = np.asarray(config.get("cond_snr") or [.3, .1, 1., .1, 1.], np.float32)
        self.histogram = np.asarray(config.get("histogram_raw") or [0.]*5, np.float32)
        self.torch, self.device, self.mp_concat = torch, device, mp_concat
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.models = {}
        for stage, folder in [("coarse", "coarse_model"), ("latent", "base_model"), ("decoder", "decoder_model")]:
            self.models[stage] = EDMUnet2D.from_pretrained(
                str(model_dir/folder), low_cpu_mem_usage=False).eval().to(device)
        self.scheduler = EDMDPMSolverMultistepScheduler(sigma_min=.002, sigma_max=80, sigma_data=.5)
        import importlib.metadata
        self.metadata = {"backend": self.name, "model": model, "model_revision": resolved_revision,
                         "device": device, "torch": torch.__version__, "upstream_commit": UPSTREAM_COMMIT,
                         "torch_threads": torch.get_num_threads(),
                         "python": sys.version.split()[0],
                         "diffusers": importlib.metadata.version("diffusers"),
                         "training_metres_per_pixel": config.get("native_resolution", 90.)}

    def tensor(self, a):
        return self.torch.as_tensor(np.asarray(a).copy(), dtype=self.torch.float32, device=self.device)

    def start_coarse(self, steps):
        self.scheduler.set_timesteps(steps)
        return float(self.scheduler.sigmas[0])

    def coarse_predict(self, a, cond, step):
        torch = self.torch
        sigma = self.scheduler.sigmas[step].to(self.device)
        scaled = self.scheduler.precondition_inputs(self.tensor(a)[None], sigma)
        label = self.scheduler.trigflow_precondition_noise(sigma.reshape(1))
        snr = self.tensor(self.cond_snr)
        inputs = [v.reshape(1) for v in torch.log(snr/8)]
        with torch.inference_mode():
            result = self.models["coarse"](torch.cat([scaled, self.tensor(cond)[None]], 1),
                      noise_labels=label, conditional_inputs=inputs)
        return result[0].cpu().numpy()

    def coarse_advance(self, prediction, state, step):
        # One scheduler/history for the entire sphere, not one per patch.
        torch = self.torch
        with torch.inference_mode():
            return self.scheduler.step(torch.from_numpy(prediction.copy())[None],
                self.scheduler.timesteps[step], torch.from_numpy(state.copy())[None]).prev_sample[0].numpy()

    def coarse_finish(self, state):
        return state/.5

    def predict(self, stage, a, cond, t):
        torch = self.torch
        if stage == "latent":
            raw = np.concatenate([cond, np.ones((1, 4, 4), np.float32)])
            means = np.array([14.99, 11.65, 15.87, 619.26, 833.12, 69.40, .66])
            stds = np.array([21.72, 21.78, 10.40, 452.29, 738.09, 34.59, .47])
            c = self.tensor((raw-means[:, None, None])/stds[:, None, None])[None]
            inputs = [self.mp_concat([c[:, :1].flatten(1), c[:, 1:2].flatten(1),
                c[:, 2:6, 1:3, 1:3].mean((2, 3)), c[:, 6:7].flatten(1),
                self.tensor(self.histogram)[None], self.tensor([[-np.sqrt(3)]])], dim=1)]
            model_input = self.tensor(a)[None]
        else:
            model_input = self.tensor(np.concatenate([a, cond]))[None]
            inputs = []
        with torch.inference_mode():
            result = self.models[stage](model_input, noise_labels=self.tensor([t]), conditional_inputs=inputs)
        return result[0].cpu().numpy()
