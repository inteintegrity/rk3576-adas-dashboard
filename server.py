"""Single-process FastAPI dashboard for actual RK3576 inference jobs."""
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
import json
import logging
import os
from pathlib import Path
import shutil
import threading
import uuid

import cv2
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from adas.detector import Detector
from adas.pipeline import run_video

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / 'outputs'
MODEL = Path(os.environ.get('ADAS_MODEL', ROOT / 'models/yolov8n_rk3576_fp16.rknn'))
SOURCE = ROOT / 'upstream/data/roads/video1-720p.mp4'
MAX_UPLOAD = 512 * 1024 * 1024
jobs = {}
lock = threading.Lock()
active = None
detector = None
executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='adas-npu')


@asynccontextmanager
async def lifespan(app):
    global detector
    RUNS.mkdir(exist_ok=True)
    # Fail startup visibly if hardware/model cannot initialize. Never fall back to CPU.
    detector = Detector(MODEL, 'rknn')
    for folder in sorted(RUNS.iterdir()):
        if not folder.is_dir(): continue
        summary = folder / 'summary.json'
        if summary.exists() and (folder / 'result.mp4').exists():
            data = json.loads(summary.read_text())
            if data.get('backend') == 'rknn':
                jobs[folder.name] = {'id':folder.name,'status':'completed',
                    'filename':Path(data['source']).name, 'metrics':data,
                    'preview':None, 'error':None}
    yield
    executor.shutdown(wait=True)
    detector.close()


app = FastAPI(title='AI-Powered ADAS Dashboard', lifespan=lifespan)
app.mount('/static', StaticFiles(directory=ROOT/'web'), name='static')


@app.get('/')
def index():
    return FileResponse(ROOT/'web/index.html')


@app.get('/api/health')
def health():
    compatible = Path('/proc/device-tree/compatible')
    return {'ready':detector is not None, 'backend':'RKNN Lite2 / RK3576 NPU',
            'model':MODEL.name, 'device':compatible.read_bytes().replace(b'\0',b' ').decode() if compatible.exists() else 'unknown',
            'max_upload_mb':MAX_UPLOAD//1024//1024, 'busy':active is not None}


def get_job(job_id):
    if job_id not in jobs: raise HTTPException(404, 'Job not found')
    return jobs[job_id]


def public(job):
    return {k:v for k,v in job.items() if k != 'preview'}


@app.get('/api/jobs')
def list_jobs():
    with lock: return [public(j) for j in reversed(list(jobs.values()))]


@app.get('/api/jobs/{job_id}')
def status(job_id: str):
    with lock: return public(get_job(job_id))


def work(job_id, source):
    global active
    folder = RUNS/job_id
    try:
        with lock: jobs[job_id]['status']='processing'
        def update(metrics, frame):
            ok, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY,80])
            with lock:
                jobs[job_id]['metrics'] = metrics
                if ok: jobs[job_id]['preview'] = jpg.tobytes()
        summary = run_video(source, folder, detector, progress=update)
        with lock: jobs[job_id].update(status='completed',metrics=summary)
    except Exception:
        logging.exception('ADAS processing failed: %s',job_id)
        with lock: jobs[job_id].update(status='failed',error='Processing failed. See the board server log for details.')
    finally:
        with lock:
            active = None
            (folder/'job.json').write_text(json.dumps(public(jobs[job_id]),indent=2))


def reserve(filename):
    global active
    with lock:
        if active is not None: raise HTTPException(409,'Another video is being processed. Please wait until it finishes.')
        job_id = uuid.uuid4().hex
        active = job_id
        (RUNS/job_id).mkdir()
        jobs[job_id]={'id':job_id,'status':'uploading','filename':filename,'metrics':{},'preview':None,'error':None}
        return job_id


@app.post('/api/jobs/sample', status_code=202)
def sample():
    if not SOURCE.is_file(): raise HTTPException(404,'Original sample video is missing')
    if shutil.disk_usage(RUNS).free < 256*1024*1024: raise HTTPException(507,'Not enough free storage')
    job_id = reserve(SOURCE.name)
    executor.submit(work,job_id,SOURCE)
    return {'id':job_id}


@app.post('/api/jobs', status_code=202)
async def upload(file: UploadFile):
    global active
    if Path(file.filename or '').suffix.lower() != '.mp4':
        await file.close()
        raise HTTPException(415,'Please upload an MP4 video')
    if shutil.disk_usage(RUNS).free < MAX_UPLOAD*3:
        await file.close()
        raise HTTPException(507,'Not enough free storage for a new video')
    job_id = reserve(Path(file.filename).name)
    target = RUNS/job_id/'input.mp4'
    try:
        size = 0
        with target.open('wb') as stream:
            while chunk := await file.read(1024*1024):
                size += len(chunk)
                if size > MAX_UPLOAD: raise HTTPException(413,'MP4 exceeds the 512 MB upload limit')
                stream.write(chunk)
        if not size: raise HTTPException(400,'The uploaded file is empty')
        cap = cv2.VideoCapture(str(target))
        valid = cap.isOpened() and cap.read()[0]
        cap.release()
        if not valid: raise HTTPException(422,'The file does not contain a decodable video')
        executor.submit(work,job_id,target)
        return {'id':job_id}
    except BaseException:
        target.unlink(missing_ok=True)
        with lock:
            active = None
            jobs[job_id].update(status='failed',error='Upload rejected: invalid or oversized video')
        raise
    finally:
        await file.close()


@app.get('/api/jobs/{job_id}/preview')
def preview(job_id: str):
    with lock: jpg = get_job(job_id)['preview']
    if jpg is None: raise HTTPException(404,'Preview is not ready')
    return Response(jpg,media_type='image/jpeg',headers={'Cache-Control':'no-store'})


@app.get('/api/jobs/{job_id}/{artifact}')
def artifact(job_id: str, artifact: str, download: bool=False):
    with lock: job = get_job(job_id)
    if job['status'] != 'completed': raise HTTPException(409,'Results are not ready')
    name, media = {'video':('result.mp4','video/mp4'), 'report':('summary.json','application/json'),
                   'trace':('trace.jsonl','application/x-ndjson')}.get(artifact,(None,None))
    if name is None: raise HTTPException(404,'Artifact not found')
    path = RUNS/job_id/name
    if not path.is_file(): raise HTTPException(404,'Artifact not found')
    return FileResponse(path,media_type=media,filename=f'adas-{job_id}-{name}' if download else None)


if __name__ == '__main__':
    import argparse
    import uvicorn
    p=argparse.ArgumentParser()
    p.add_argument('--host',default='0.0.0.0');p.add_argument('--port',type=int,default=8092)
    args=p.parse_args()
    cv2.setNumThreads(4)
    uvicorn.run(app,host=args.host,port=args.port,workers=1)
