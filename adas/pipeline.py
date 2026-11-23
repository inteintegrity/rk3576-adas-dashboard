"""Headless adaptation of the upstream final video cell; one inference per frame."""
import json
import subprocess
import time
from pathlib import Path

import cv2 as cv
import numpy as np

from . import upstream_lanes as lanes


def process_frame(frame, detector, memory):
    detector.detect(frame)
    uniform = lanes.apply_clahe(frame)
    mask = lanes.combined_color_gradient_mask(uniform, sobel_thresh_min=7,
                    white_l_min=92, white_l_max=255, white_s_min=0)
    gray = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
    clean = lanes.blackout_vehicles_from_mask(gray, frame, detector)
    roi = lanes.proportional_roi(clean, bottom_offset_left=100, bottom_offset_right=1000,
                    top_offset=.55, top_width_factor_right=.70, top_width_factor_left=.45)
    lines = lanes.hough_lines(roi, threshold=50, minLineLength=12, maxLineGap=17)
    fitted = lanes.merging_lines_video(frame, lines, memory)
    line_image = lanes.display_lines(frame, fitted)
    ldw = lanes.check_lane_crossing(frame, fitted, margin=150)
    combo = cv.addWeighted(frame, .8, line_image, 1, 1)
    combo = cv.addWeighted(combo, 1, ldw, 1, 0)
    area = lanes.draw_lane_area(uniform, combo, fitted, color_tolerance=60)
    output = lanes.analyze_traffic_and_alarms(frame, area, detector, fitted)
    telemetry = {'vehicles': detector.events,
                 'lanes': [{'coords': [int(x) for x in coords], 'type': kind} for coords, kind in (fitted or [])],
                 'ldw': bool(np.any(ldw)), 'fcw': any(v['fcw'] for v in detector.events)}
    return output, telemetry


def run_video(source, output_dir, detector, limit=0, progress=None):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cap = cv.VideoCapture(str(source))
    if not cap.isOpened():
        raise ValueError('Cannot open MP4 video')
    fps = cap.get(cv.CAP_PROP_FPS)
    total = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    width, height = (int(cap.get(x)) for x in (cv.CAP_PROP_FRAME_WIDTH, cv.CAP_PROP_FRAME_HEIGHT))
    if not np.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise ValueError('Invalid MP4 metadata')
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    # Warmup is measured separately and excluded from steady-state NPU statistics.
    ok, first = cap.read()
    if not ok:
        cap.release()
        raise ValueError('MP4 contains no decodable frames')
    from .detector import letterbox
    warm_input, _ = letterbox(first)
    warmup = [detector.run_raw(warm_input)[1] for _ in range(5)]
    encoder_log = (out / 'encoder.log').open('w')
    tmp = out / 'result.partial.mp4'
    encoder = subprocess.Popen([ffmpeg, '-hide_banner', '-loglevel', 'error', '-y',
        '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{width}x{height}', '-r', str(fps),
        '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23',
        '-pix_fmt', 'yuv420p', '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
        '-movflags', '+faststart', str(tmp)], stdin=subprocess.PIPE, stderr=encoder_log)
    memory, counts, npu_times = [], {'ldw_frames': 0, 'fcw_frames': 0, 'vehicle_detections': 0, 'frames_with_lanes': 0}, []
    start = time.perf_counter()
    count = 0
    frame = first
    try:
        with (out / 'trace.jsonl').open('w') as trace:
            while frame is not None:
                rendered, record = process_frame(frame, detector, memory)
                count += 1
                npu_times.append(detector.times[-1])
                counts['ldw_frames'] += record['ldw']; counts['fcw_frames'] += record['fcw']
                counts['vehicle_detections'] += len(record['vehicles'])
                counts['frames_with_lanes'] += bool(record['lanes'])
                elapsed = time.perf_counter() - start
                npu_fps = len(npu_times) / sum(npu_times) if detector.backend == 'rknn' else None
                label = f'NPU API: {npu_fps:.1f} FPS' if npu_fps else 'ONNX CPU reference'
                cv.putText(rendered, f'{label} | Processing: {count/elapsed:.1f} FPS',
                           (20, height-20), cv.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2, cv.LINE_AA)
                record.update(frame=count-1, timestamp_s=(count-1)/fps, inference_ms=npu_times[-1]*1000)
                trace.write(json.dumps(record) + '\n')
                encoder.stdin.write(rendered.tobytes())
                if count == 1 or count % max(1, round(fps*5)) == 0:
                    cv.imwrite(str(out / f'frame-{count:05d}.jpg'), rendered)
                if progress and (count == 1 or count % 10 == 0):
                    progress({'frames': count, 'total_frames': min(total, limit) if limit else total,
                              'npu_fps': npu_fps, 'processing_fps': count/(time.perf_counter()-start), **counts}, rendered)
                if limit and count >= limit:
                    break
                ok, frame = cap.read()
                if not ok:
                    frame = None
        encoder.stdin.close()
        status = encoder.wait(timeout=120)
        if status:
            raise RuntimeError(f'H.264 encoding failed, see {out / "encoder.log"}')
    except BaseException:
        if encoder.poll() is None:
            encoder.kill()
            encoder.wait()
        raise
    finally:
        cap.release()
        encoder_log.close()
    elapsed = time.perf_counter() - start
    if not count:
        raise ValueError('No processed frames')
    tmp.replace(out / 'result.mp4')
    summary = {'source': str(source), 'backend': detector.backend, 'frames': count,
        'source_frames': total, 'source_fps': fps, 'width': width, 'height': height,
        'elapsed_s': elapsed, 'processing_fps': count/elapsed,
        'npu_fps': count/sum(npu_times) if detector.backend == 'rknn' else None,
        'inference_mean_ms': float(np.mean(npu_times)*1000),
        'inference_median_ms': float(np.median(npu_times)*1000),
        'warmup_ms': [v*1000 for v in warmup], **counts,
        'timing': 'NPU FPS = frames / sum of synchronous RKNNLite.inference wall time; includes API transfers. '
                  'Processing FPS includes decode, preprocessing, inference, lanes, drawing, trace and H.264 finalization; excludes initialization and warmup. '
                  'Video overlay/live progress is cumulative before finalization. Playback keeps source FPS; audio omitted.'}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    return summary
