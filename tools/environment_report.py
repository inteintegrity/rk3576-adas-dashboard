"""Record reproducible board dependency versions without credentials or environment variables."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import cv2

packages=['numpy','opencv-python','scikit-learn','scipy','imageio-ffmpeg',
          'rknn-toolkit-lite2','fastapi','uvicorn','python-multipart','starlette','pydantic']
versions={}
for name in packages:
    try: versions[name]=importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError: versions[name]='not installed'
compatible=Path('/proc/device-tree/compatible')
report={'platform':platform.platform(),'python':platform.python_version(),
        'compatible':compatible.read_bytes().replace(b'\0',b' ').decode() if compatible.exists() else None,
        'packages':versions,'cv2_runtime':cv2.__version__,
        'rknn_sha256':hashlib.sha256(Path('models/yolov8n_rk3576_fp16.rknn').read_bytes()).hexdigest()}
Path('outputs/environment.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
