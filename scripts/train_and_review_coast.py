"""Train to a review checkpoint, then check coastlines without starting terrain."""
import argparse
from datetime import datetime
import json
from pathlib import Path

from planet_training.train_structured import main as train
from planet_training.structured_data import DEFAULT_COAST_RUN
from planet_training.run_lock import RunAlreadyActive
from scripts.review_coast_checkpoint import main as review


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',default=DEFAULT_COAST_RUN)
    parser.add_argument('--steps',type=int,default=10000)
    parser.add_argument('--batch',type=int,default=16)
    args=parser.parse_args(argv)
    result=train(['coast','--run',args.run,'--steps',str(args.steps),'--batch',str(args.batch),
                  '--eval-every','1000','--preview-steps','32'])
    if result: return result
    print('Training phase finished. Checking four seeds and requested 20/30/35% land...',flush=True)
    run=Path(args.run)
    out=run/datetime.now().strftime('review-%Y%m%d-%H%M%S-%f')
    review(['--checkpoint',str(run/'best.pt'),'--out',str(out)])
    report=json.loads((out/'review.json').read_text())
    errors=[abs(record['actual_land_percent']-record['requested_land_percent']) for record in report['samples']]
    sweep_errors=[abs(record['achieved']-record['requested']) for record in report['coverage_sweep_same_seed']]
    mean_error=sum(errors)/len(errors);max_error=max(errors+sweep_errors)
    passes=mean_error<=3 and max_error<=5
    status=dict(review=str(out.resolve()),reviewed_step=report['step'],checkpoint_sha256=report['checkpoint_sha256'],
                mean_30_percent_error_pp=mean_error,max_coverage_error_pp=max_error,
                preliminary_coverage_gate_passed=passes,visual_review_required=True,
                note='Four seeds are a small coverage screen, not terrain approval. Inspect continent shapes and closest training maps. No relief/intermediate/meso generation started.')
    (run/'coast-quality-status.json').write_text(json.dumps(status,indent=2))
    print(f'Coverage screen: {"passed" if passes else "needs more work"}; average error {mean_error:.1f} points, maximum {max_error:.1f} points.',flush=True)
    print(f'Inspect {out}/coverage-review.png before terrain refinement. Nothing else starts automatically.',flush=True)
    return 0


if __name__=='__main__':
    try: raise SystemExit(main())
    except RunAlreadyActive as error:
        print(str(error),flush=True);raise SystemExit(1)
