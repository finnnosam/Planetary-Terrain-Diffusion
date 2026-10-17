"""Readable training output, with a single updating line in interactive terminals."""
import math
import shutil
import sys


def duration(hours):
    if not math.isfinite(hours):
        return "estimating"
    minutes = max(0, math.ceil(hours * 60))
    if minutes == 0:
        return "0m"
    days, minutes = divmod(minutes, 1440)
    hours, minutes = divmod(minutes, 60)
    if days:
        return f"{days}d {hours}h {minutes:02d}m"
    if hours:
        return f"{hours}h {minutes:02d}m"
    return f"{minutes}m"


class TrainingConsole:
    def __init__(self, total, stream=None):
        self.total = total
        self.stream = sys.stdout if stream is None else stream
        self.live = bool(getattr(self.stream, "isatty", lambda: False)())
        self.line_width = 0

    def clear(self):
        if self.line_width:
            self.stream.write("\r" + " " * self.line_width + "\r")
            self.line_width = 0

    def message(self, text):
        self.clear()
        print(text, file=self.stream, flush=True)

    def start(self, stage, step, batch, device, parameters, run):
        name = {"layout": "CONTINENT MODEL", "meso": "MESO TERRAIN MODEL"}.get(stage, "INTERMEDIATE TERRAIN MODEL")
        scale = {"layout": "Whole-planet layout", "meso": "23.04 km -> 2.88 km refinement"}.get(stage, "161 km -> 23.04 km refinement")
        self.message("\n" + name)
        self.message("-" * len(name))
        self.message(f"{scale} | {str(device).upper()} | Batch {batch} | {parameters/1e6:.2f}M parameters")
        self.message(f"Run folder: {run}")
        self.message(f"{'Resuming' if step else 'Starting'} at update {step:,} of {self.total:,}")
        self.message("Ctrl+C saves a checkpoint and stops safely.\n")

    def record(self, obj):
        if "loss" in obj:
            step, rate = obj["step"], obj["steps_per_s"]
            percent = min(100., step/self.total*100)
            width = max(40, shutil.get_terminal_size((120, 24)).columns - 1)
            text = (f"{step:,}/{self.total:,} ({percent:5.1f}%) | Loss {obj['loss']:.4f} | "
                    f"{rate:.2f} upd/s | ETA {duration(obj['remaining_hours'])}")
            if width >= len(text) + 23:
                filled = min(18, int(percent/100*18))
                text = f"[{'#'*filled}{'-'*(18-filled)}] " + text
            if len(text) > width:
                text = (f"{percent:.1f}% | Update {step:,} | Loss {obj['loss']:.4f} | "
                        f"ETA {duration(obj['remaining_hours'])}")
            if self.live:
                self.stream.write("\r" + text.ljust(self.line_width))
                self.stream.flush()
                self.line_width = len(text)
            else:
                self.message(text)
        elif "val_loss" in obj:
            suffix = "  [new best]" if obj.get("new_best") else ""
            self.message(f"  Validation loss: {obj['val_loss']:.4f}{suffix}")
        elif "peak_allocated_gib" in obj:
            self.message(f"  GPU memory peak: {obj['peak_allocated_gib']:.2f} GiB")
        elif "coverage_preview" in obj:
            check = obj['coverage_preview']
            values = ', '.join(f'{actual:.1f}% (requested {target:.1f}%)'
                               for target,actual in zip(check['requested_land_percent'],check['actual_land_percent']))
            self.message(f'  Preview land coverage: {values}')

    def complete(self, step, run):
        self.message(f"\nTraining complete: {step:,}/{self.total:,} updates.")
        self.message(f"Checkpoints and previews: {run}\n")
