"""Compare real model detections on seven original-video frames; also records timings."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from adas.detector import Detector, letterbox


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--backend', choices=['rknn', 'onnx', 'pytorch'], required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--source', default='upstream/data/roads/video1-720p.mp4')
    a = p.parse_args()
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(a.source)
    if a.backend == 'pytorch':
        import torch
        from ultralytics import YOLO
        torch.set_num_threads(4)
        detector = YOLO(a.model)
    else:
        detector = Detector(a.model, a.backend)
    records = []
    try:
        for index in [0, 125, 250, 375, 500, 625, 728]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f'Cannot decode original frame {index}')
            if a.backend == 'pytorch':
                result = detector.predict(frame, imgsz=640, rect=False, classes=[2,3,5,7],
                    conf=.30, iou=.7, verbose=False)[0]
                boxes = result.boxes.xyxy.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
            else:
                image, _ = letterbox(frame)
                if not records:
                    for _ in range(5): detector.run_raw(image)
                found = detector.detect(frame)
                boxes = np.array([b.xyxy[0] for b in found]).reshape(-1,4)
                scores = np.array([b.conf[0] for b in found])
                classes = np.array([b.cls[0] for b in found])
            records.append({'frame':index, 'boxes':boxes.tolist(), 'classes':classes.astype(int).tolist(),
                            'scores':scores.tolist()})
            for box, score, cls in zip(boxes, scores, classes):
                x1,y1,x2,y2 = box.astype(int)
                cv2.rectangle(frame, (x1,y1),(x2,y2),(0,220,255),2)
                cv2.putText(frame,f'{int(cls)} {score:.3f}',(x1,max(20,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,220,255),2)
            cv2.imwrite(str(out / f'frame-{index:04d}.jpg'), frame)
        if a.backend != 'pytorch':
            timings = [detector.run_raw(image)[1] for _ in range(50)]
        else:
            timings = []
        report = {'backend':a.backend,'model':a.model,'records':records,'api_times_s':timings}
        (out/'detections.json').write_text(json.dumps(report, indent=2))
        print(json.dumps({'backend':a.backend,'detections':sum(len(r['boxes']) for r in records),
                          'api_fps':len(timings)/sum(timings) if timings else None}))
    finally:
        cap.release()
        if a.backend != 'pytorch': detector.close()


if __name__ == '__main__': main()
