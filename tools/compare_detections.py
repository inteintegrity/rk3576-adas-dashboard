"""Match each real reference box once, class-aware, and fail on detector regression."""
import argparse
import json
from pathlib import Path
import numpy as np


def compare(reference, candidate):
    rows = []
    for ref, actual in zip(reference['records'], candidate['records']):
        if ref['frame'] != actual['frame']:
            raise ValueError('Reference frames differ')
        used, matches = set(), []
        for box, cls, score in zip(ref['boxes'], ref['classes'], ref['scores']):
            a = np.array(box)
            options = []
            for j, (other, other_cls) in enumerate(zip(actual['boxes'], actual['classes'])):
                if j in used or other_cls != cls: continue
                b = np.array(other)
                inter = np.maximum(np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]),0).prod()
                union = (a[2:]-a[:2]).prod()+(b[2:]-b[:2]).prod()-inter
                options.append((float(inter/max(union,1e-9)),j))
            if not options:
                matches.append({'iou':0.0, 'matched':False}); continue
            iou, j = max(options)
            used.add(j)
            matches.append({'iou':iou,'matched':iou>=.9,
                            'score_error':abs(score-actual['scores'][j])})
        rows.append({'frame':ref['frame'],'reference_count':len(ref['boxes']),
                     'actual_count':len(actual['boxes']),'matches':matches})
    passed = len(reference['records']) == len(candidate['records']) and all(
        r['reference_count']==r['actual_count'] and all(m['matched'] for m in r['matches']) for r in rows)
    return {'passed':passed,'minimum_iou':min(m['iou'] for r in rows for m in r['matches']),
            'max_score_error':max(m.get('score_error',1) for r in rows for m in r['matches']), 'frames':rows}


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('reference');p.add_argument('candidate');p.add_argument('--output',required=True)
    a=p.parse_args()
    report=compare(json.loads(Path(a.reference).read_text()),json.loads(Path(a.candidate).read_text()))
    Path(a.output).write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='frames'}))
    if not report['passed']: raise SystemExit(1)
