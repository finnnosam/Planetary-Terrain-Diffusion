"""Prepare and launch one isolated structure pilot. Never starts generation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

from planet_training.adaptation import STRUCTURE_PROFILE
from planet_training.data import atomic_json
from planet_training.prepare_structure import DEFAULT_DATA, prepare
from planet_training.run_lock import RunAlreadyActive, training_lock
from planet_training.train import main as train

RUN = 'outputs/intermediate/structure-pilot-v1'
BASELINE = 'outputs/intermediate/selected-adapted-1750/best.pt'
BASELINE_SHA = 'fd04c930551ddebc809accf75942a37503403bb3f2f0b3b93957b0ddc9585ac9'


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--steps',type=int,default=500)
    p.add_argument('--run',default=RUN)
    p.add_argument('--data',default=DEFAULT_DATA)
    p.add_argument('--batch',type=int,default=16)
    p.add_argument('--device',default='cuda')
    p.add_argument('--check-only',action='store_true')
    p.add_argument('--smoke',action='store_true',help='two real updates in a separate smoke folder')
    a=p.parse_args(argv)
    if a.steps<1 or a.batch<1:p.error('Steps and batch must be positive')
    baseline=Path(BASELINE)
    digest=hashlib.file_digest(baseline.open('rb'),'sha256').hexdigest()
    if digest!=BASELINE_SHA:raise ValueError('Frozen baseline checkpoint hash changed')
    prepare(out=a.data)
    run=Path(a.run)
    if a.smoke:
        run=run/'smoke-test';a.steps=2
        if (run/'latest.pt').exists():raise ValueError('Smoke test already exists; choose a fresh --run for another test')
    args=['bridge','--data',a.data,'--run',str(run),'--init-from',BASELINE,
          '--guide-profile',STRUCTURE_PROFILE,'--steps',str(a.steps),'--batch',str(a.batch),
          '--device',a.device,'--lr','0.00003','--warmup','100','--ema','0.999',
          '--cond-noise','0.03','--seed','42','--eval-every','100','--save-every','50',
          '--log-every','1' if a.smoke else '10','--preview-steps','2' if a.smoke else '32',
          '--validation-patience','0','--threads','4']
    if a.check_only:args+=['--check-only']
    print('\nINTERMEDIATE STRUCTURE PILOT\nTargets keep Earth mountain belts and valleys; coast/relief and meso selections stay fixed.',flush=True)
    print(f'Update ceiling: {a.steps}. Output: {run}\nTraining only. Regional review is a separate command. No full globe starts.',flush=True)
    job={'baseline':str(baseline.resolve()),'baseline_sha256':digest,
         'command':[sys.executable,'-u','-m','planet_training.train',*args],
         'target_change':'unchanged Earth targets instead of translated encoded baseline',
         'full_planet_generation':False}
    # Inspect/update the launcher record only while the run is not active.
    with training_lock(run):atomic_json(run/'training-job.json',job)
    started=time.perf_counter();result=train(args)
    if a.smoke:
        atomic_json(run/'smoke-result.json',{'seconds':time.perf_counter()-started,'updates':2,'exit_code':result or 0})
    if result:return result
    if a.check_only:print('Preflight passed. No training performed.',flush=True)
    elif a.smoke:print('\nTwo-update smoke test passed. The 500-update pilot has not started.',flush=True)
    else:print('\nPilot finished. Inspect previews or run "Review Intermediate Structure Pilot.cmd". Nothing else starts automatically.',flush=True)
    return 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except (ValueError,RunAlreadyActive) as error:
        print(f'Cannot start: {error}',flush=True);raise SystemExit(1)
