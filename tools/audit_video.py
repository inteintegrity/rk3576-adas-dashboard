"""Audit a completed real run and extract actual warning frames from its trace."""
import argparse
import json
from pathlib import Path
import cv2


def main():
    p=argparse.ArgumentParser();p.add_argument('run');a=p.parse_args()
    folder=Path(a.run)
    summary=json.loads((folder/'summary.json').read_text())
    trace=[json.loads(s) for s in (folder/'trace.jsonl').read_text().splitlines()]
    assert len(trace)==summary['frames']
    assert [r['frame'] for r in trace]==list(range(len(trace)))
    assert sum(r['ldw'] for r in trace)==summary['ldw_frames']
    assert sum(r['fcw'] for r in trace)==summary['fcw_frames']
    assert sum(len(r['vehicles']) for r in trace)==summary['vehicle_detections']
    assert all(0<=v['score']<=1 for r in trace for v in r['vehicles'])
    assert abs(len(trace)/sum(r['inference_ms']/1000 for r in trace)-summary['npu_fps'])<1e-6
    cap=cv2.VideoCapture(str(folder/'result.mp4'))
    decoded=0
    while cap.read()[0]: decoded+=1
    fps=cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    assert decoded==len(trace)
    assert abs(fps-summary['source_fps'])<.001
    selected={}
    for kind in ('fcw','ldw'):
        record=next((r for r in trace if r[kind]),None)
        if record:
            cap=cv2.VideoCapture(str(folder/'result.mp4'))
            cap.set(cv2.CAP_PROP_POS_FRAMES,record['frame'])
            ok,frame=cap.read();cap.release()
            assert ok
            cv2.imwrite(str(folder/f'{kind}-actual.jpg'),frame)
            selected[kind]=record['frame']
    report={'passed':True,'decoded_frames':decoded,'fps':fps,'actual_alert_frames':selected,
        'lane_types':sorted({lane['type'] for r in trace for lane in r['lanes']}),
        'vehicle_classes':sorted({v['class'] for r in trace for v in r['vehicles']})}
    (folder/'audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report))


if __name__=='__main__':main()
