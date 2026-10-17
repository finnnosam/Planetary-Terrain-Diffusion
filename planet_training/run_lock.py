"""One trainer per run folder; the OS releases the lock even after a crash."""
from contextlib import contextmanager
import os
from pathlib import Path


class RunAlreadyActive(RuntimeError):
    pass


@contextmanager
def training_lock(run):
    run=Path(run);run.mkdir(parents=True,exist_ok=True)
    with (run/'training.lock').open('a+b') as handle:
        if handle.seek(0,2)==0:
            handle.write(b'0');handle.flush()
        handle.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as error:
            raise RunAlreadyActive(f'Training is already active in {run}. Leave that trainer running; do not launch a second copy.') from error
        try:
            yield
        finally:
            handle.seek(0)
            if os.name=='nt': msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
            else: fcntl.flock(handle.fileno(),fcntl.LOCK_UN)
