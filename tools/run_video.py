import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adas.detector import Detector
from adas.pipeline import run_video

p = argparse.ArgumentParser()
p.add_argument('--source', default='upstream/data/roads/video1-720p.mp4')
p.add_argument('--model', default='models/yolov8n_rk3576_fp16.rknn')
p.add_argument('--backend', choices=['rknn', 'onnx'], default='rknn')
p.add_argument('--output', default='outputs/rk3576-original')
p.add_argument('--limit', type=int, default=0)
a = p.parse_args()
d = Detector(a.model, a.backend)
try:
    summary = run_video(a.source, a.output, d, a.limit,
                        lambda x, frame: print(json.dumps(x), flush=True) if x['frames'] % 100 == 0 else None)
    print(json.dumps(summary, indent=2))
finally:
    d.close()
