"""Build an RK3576 FP16 model using existing RKNN Toolkit2. No quantization/training."""
import argparse
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path
from rknn.api import RKNN


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--onnx', default='models/yolov8n_raw.onnx')
    p.add_argument('--output', default='models/yolov8n_rk3576_fp16.rknn')
    a = p.parse_args()
    started = time.perf_counter()
    r = RKNN(verbose=False)
    timings = {'initialize_s': time.perf_counter() - started}
    try:
        for name, op in (
            ('config', lambda: r.config(target_platform='rk3576',
                mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]])),
            ('load', lambda: r.load_onnx(model=a.onnx)),
            ('build', lambda: r.build(do_quantization=False)),
            ('export', lambda: r.export_rknn(a.output))):
            stage_started = time.perf_counter()
            ret = op()
            timings[name + '_s'] = time.perf_counter() - stage_started
            if ret != 0:
                raise RuntimeError(f'{name} failed: {ret}')
        timings['total_s'] = time.perf_counter() - started
        Path(a.output).with_suffix('.json').write_text(json.dumps({
            'target': 'rk3576', 'precision': 'fp16',
            'toolkit2': importlib.metadata.version('rknn-toolkit2'),
            'onnx_sha256': hashlib.sha256(Path(a.onnx).read_bytes()).hexdigest(),
            'rknn_sha256': hashlib.sha256(Path(a.output).read_bytes()).hexdigest(),
            'conversion_timings': timings,
            'input': 'uint8 NHWC RGB, normalization embedded in RKNN'}, indent=2))
        print(json.dumps({'conversion_timings': timings}, indent=2))
    finally:
        r.release()


if __name__ == '__main__':
    main()
