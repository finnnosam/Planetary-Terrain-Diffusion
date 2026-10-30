"""Durable console logs for user-launched structure jobs; no hidden processes."""
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from datetime import datetime
from pathlib import Path
import sys


class Tee:
    def __init__(self, stream, file): self.stream, self.file = stream, file
    def write(self, value):
        self.stream.write(value); self.file.write(value); self.file.flush()
        return len(value)
    def flush(self): self.stream.flush(); self.file.flush()
    def isatty(self): return False


@contextmanager
def console_log(folder, label):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    path = folder/(label+'-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f')+'.log')
    with path.open('w', encoding='utf-8') as f:
        with redirect_stdout(Tee(sys.stdout, f)), redirect_stderr(Tee(sys.stderr, f)):
            print(f'Console log: {path.resolve()}', flush=True)
            try: yield path
            except BaseException as error:
                print(f'Job ended: {type(error).__name__}: {error}', flush=True)
                import traceback
                traceback.print_exc(file=f); f.flush()
                raise


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('job', choices=['prepare', 'train', 'review']); p.add_argument('args', nargs=argparse.REMAINDER)
    a = p.parse_args()
    with console_log('outputs/structure/job-logs', a.job):
        if a.job == 'prepare':
            from planet_training.prepare_terrain_structure import main as run
        elif a.job == 'train':
            from planet_training.train_terrain_structure import main as run
        else:
            from scripts.review_terrain_structure import main as run
        result = run(a.args)
        code = result if isinstance(result, int) else 0
        print(f'{a.job.capitalize()} job {"finished" if code == 0 else "stopped/failed"}; exit {code}. No full planet generation or model promotion.', flush=True)
        return code


if __name__ == '__main__':
    try: raise SystemExit(main())
    except KeyboardInterrupt: raise SystemExit(130)
