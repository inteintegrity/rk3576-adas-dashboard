"""Verify original inputs using Git blob hashes from the pinned upstream tree."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
COMMIT='58e6f31f5e4cc8c7b1c838920d9da454e2c11627'
FILES={
    'LICENSE':'38f0fbdba22f4816cd44997e66688bd4b7fbca15',
    'README.md':'96dd2b1c8ff3f00e28507daf8737162f6d8b858e',
    'src/project.py':'2f3973b398c963d9d4cb5e7e07de4220368c43b1',
    'notebook/FinalProject.ipynb':'475bc898ec33904fe741b46e634ec71107c7f757',
    'data/roads/video1-720p.mp4':'041a0847acc18ca3581c5d3dde5de663b5898820',
}


def main():
    p=argparse.ArgumentParser();p.add_argument('--fetch',action='store_true');a=p.parse_args()
    records={}
    for name,expected in FILES.items():
        path=ROOT/'upstream'/name
        if not path.exists() and a.fetch:
            path.parent.mkdir(parents=True,exist_ok=True)
            url=f'https://raw.githubusercontent.com/Kiya-rb/vision-based-adas-dashboard/{COMMIT}/{name}'
            with urllib.request.urlopen(url,timeout=600) as r: data=r.read()
            path.write_bytes(data)
        data=path.read_bytes()
        blob=hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()
        if blob!=expected: raise ValueError(f'Upstream integrity check failed: {name}: {blob}')
        records[name]={'bytes':len(data),'git_blob_sha1':blob,'sha256':hashlib.sha256(data).hexdigest()}
    manifest={'repository':'https://github.com/Kiya-rb/vision-based-adas-dashboard',
              'commit':COMMIT,'files':records,
              'weights':{'path':'models/yolov8n.pt','sha256':hashlib.sha256((ROOT/'models/yolov8n.pt').read_bytes()).hexdigest(),
                         'source':'Existing local official COCO YOLOv8n pretrained checkpoint; no training'}}
    (ROOT/'upstream/manifest.json').write_text(json.dumps(manifest,indent=2))
    print('All five upstream files match the pinned Git blobs.')


if __name__=='__main__': main()
