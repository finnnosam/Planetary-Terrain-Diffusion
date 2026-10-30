"""Train relief, then make a small end-to-end review with the selected coast."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

from planet_training.data import manifest_id,atomic_json
from planet_training.train_structured import main as train
from planet_training.generate_structured import load
from planet_training.evaluate_structured import main as evaluate
from planet_training.structured_data import DEFAULT_RELIEF_RUN,OBJECTIVE,STRUCTURED_VERSION
from planet_training.run_lock import RunAlreadyActive


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',default=DEFAULT_RELIEF_RUN)
    parser.add_argument('--selection',default='outputs/continents/structured-selection.json')
    parser.add_argument('--steps',type=int,default=10000)
    parser.add_argument('--batch',type=int,default=16)
    parser.add_argument('--data',default='outputs/planet_training/data/layout')
    args=parser.parse_args(argv)
    selected=json.loads(Path(args.selection).read_text())['coast']
    coast_path=Path(selected['checkpoint'])
    if hashlib.sha256(coast_path.read_bytes()).hexdigest()!=selected['checkpoint_sha256']:
        raise ValueError('Selected coast checkpoint changed; review and select it again before training')
    model,checkpoint,_=load(coast_path,'coast','cpu')
    meta=json.loads((Path(args.data)/'manifest.json').read_text())
    identity=manifest_id(dict(source=meta,representation=STRUCTURED_VERSION))
    if checkpoint.get('objective')!=OBJECTIVE or checkpoint['dataset_id']!=identity:
        raise ValueError('Selected coast must use the corrected objective and the same source data')
    del model,checkpoint
    result=train(['relief','--run',args.run,'--data',args.data,'--steps',str(args.steps),
                  '--batch',str(args.batch),'--eval-every','1000','--preview-steps','32'])
    if result: return result
    run=Path(args.run);out=run/datetime.now().strftime('review-%Y%m%d-%H%M%S-%f')
    out.mkdir(parents=True)
    snapshot=out/'reviewed-relief.pt'
    snapshot.write_bytes((run/'best.pt').read_bytes())
    print('Relief training finished. Making four small coast-and-elevation previews...',flush=True)
    evaluate(['--coast',str(coast_path),'--relief',str(snapshot),'--data',args.data,
              '--out',str(out),'--steps','64','--batch',str(args.batch),'--device','cuda'])
    report=json.loads((out/'evaluation.json').read_text())
    status=dict(review=str(out.resolve()),coast_sha256=report['coast_sha256'],relief_sha256=report['relief_sha256'],
                coast_step=report['coast_step'],relief_step=report['relief_step'],visual_review_required=True,
                next_check='Inspect shelves, coastal heights and broad mountain belts, then a few regional refinements.',
                note='No full intermediate or meso generation was started.')
    atomic_json(run/'relief-quality-status.json',status)
    print(f'Review ready: {out}/comparison.png. Nothing else starts automatically.',flush=True)
    return 0


if __name__=='__main__':
    try: raise SystemExit(main())
    except RunAlreadyActive as error:
        print(str(error),flush=True);raise SystemExit(1)
