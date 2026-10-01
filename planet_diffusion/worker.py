"""Sequential desktop worker with framed logs and reusable lazy models."""
import contextlib
import json
import sys
from pathlib import Path

PREFIX = "@planet-worker "


def source_signature():
    return tuple((file.name,file.stat().st_size,file.stat().st_mtime_ns)
                 for file in sorted(Path(__file__).parent.glob('*.py')))


class BackendPool:
    def __init__(self):
        self.backend = None

    def __call__(self, *args, **kwargs):
        from .backends import TerrainBackend
        candidate = TerrainBackend(*args, **kwargs)
        if self.backend is not None and self.backend.metadata == candidate.metadata:
            candidate.models = self.backend.models
        else:
            # Retain only one model family/device/precision, bounding GPU memory.
            self.backend = None
            import gc
            gc.collect()
            if candidate.device == 'cuda':
                candidate.torch.cuda.empty_cache()
        self.backend = candidate
        return candidate


class EventStream:
    def __init__(self, emit):
        self.emit = emit

    def write(self, text):
        if text:
            self.emit({'event':'log', 'text':text})
        return len(text)

    def flush(self):
        pass

    def isatty(self):
        return False


def serve(input_stream, output_stream, run=None):
    from .cli import main
    pool = BackendPool()
    run = run or main
    def emit(event):
        output_stream.write(PREFIX+json.dumps(event)+'\n')
        output_stream.flush()
    logs = EventStream(emit)
    signature = source_signature()
    for line in input_stream:
        try:
            if source_signature() != signature:
                raise ValueError('Generation source changed; restart the worker')
            request = json.loads(line)
            argv = request['argv']
            if not isinstance(argv,list) or not all(isinstance(v,str) for v in argv):
                raise ValueError('Worker argv must be a list of strings')
            # The launcher submits generation/query jobs, never recursive GUIs.
            if not argv or argv[0] not in ('generate','query','export','verify'):
                raise ValueError('Unsupported worker command')
            with contextlib.redirect_stdout(logs), contextlib.redirect_stderr(logs):
                try:
                    code = run(argv,backend_factory=pool) or 0
                except SystemExit as exc:
                    code = exc.code if isinstance(exc.code,int) else 1
        except Exception as exc:
            emit({'event':'log','text':f'Worker error: {exc}\n'})
            code = 1
        import gc
        gc.collect()  # Release cyclic regional caches while retaining pool models.
        emit({'event':'done','code':code})


if __name__ == '__main__':
    serve(sys.stdin,sys.stdout)
