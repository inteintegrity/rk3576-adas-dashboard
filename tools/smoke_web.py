"""End-to-end HTTP test using the unchanged original MP4 and actual NPU, no mocks."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request
import urllib.error
import uuid


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--url',default='http://127.0.0.1:8092')
    p.add_argument('--source',default='upstream/data/roads/video1-720p.mp4')
    p.add_argument('--output',default='outputs/web-validation.json')
    a=p.parse_args()
    def call(path,data=None,headers=None):
        with urllib.request.urlopen(urllib.request.Request(a.url+path,data=data,headers=headers or {}),timeout=30) as r:
            return r.status,r.read(),{k.lower():v for k,v in r.headers.items()}
    def expect_error(path,code,data,headers=None):
        try: call(path,data,headers)
        except urllib.error.HTTPError as e:
            assert e.code==code,(e.code,code)
        else: raise AssertionError(f'Expected HTTP {code}: {path}')
    assert json.loads(call('/api/health')[1])['ready']
    assert b'AI-Powered ADAS Dashboard' in call('/')[1]
    for path in ['/static/style.css','/static/app.js']: assert call(path)[0]==200
    def multipart(name,data):
        boundary='ADAS'+uuid.uuid4().hex
        body=(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{name}"\r\n'
              'Content-Type: video/mp4\r\n\r\n').encode()+data+f'\r\n--{boundary}--\r\n'.encode()
        return body,{'Content-Type':f'multipart/form-data; boundary={boundary}'}
    body,headers=multipart('invalid.txt',b'not a video')
    expect_error('/api/jobs',415,body,headers)
    body,headers=multipart('invalid.mp4',b'not a video')
    expect_error('/api/jobs',422,body,headers)
    source=Path(a.source).read_bytes()
    body,headers=multipart(Path(a.source).name,source)
    job=json.loads(call('/api/jobs',body,headers)[1])['id']
    expect_error('/api/jobs/sample',409,b'')
    expect_error(f'/api/jobs/{job}/video',409,None)
    previews=0
    deadline=time.monotonic()+1200
    while time.monotonic()<deadline:
        state=json.loads(call(f'/api/jobs/{job}')[1])
        if state['status']=='failed': raise AssertionError(state)
        if state['status']=='completed': break
        if state['metrics'].get('frames'):
            status,jpg,_=call(f'/api/jobs/{job}/preview')
            assert status==200 and jpg.startswith(b'\xff\xd8')
            previews+=1
        time.sleep(2)
    else: raise TimeoutError('NPU video job did not finish')
    summary=json.loads(call(f'/api/jobs/{job}/report')[1])
    assert summary['frames']==729 and summary['backend']=='rknn'
    assert summary['npu_fps']>0 and summary['processing_fps']>0
    status,part,_=call(f'/api/jobs/{job}/video',headers={'Range':'bytes=0-1023'})
    assert status==206 and len(part)==1024 and b'ftyp' in part
    _,video,response_headers=call(f'/api/jobs/{job}/video?download=true')
    assert 'attachment' in response_headers.get('content-disposition','')
    trace=call(f'/api/jobs/{job}/trace')[1].decode().splitlines()
    assert len(trace)==729
    result={'passed':True,'job_id':job,'preview_checks':previews,
            'checks':['health','HTML/CSS/JS','invalid extension 415','invalid MP4 422',
                      'original MP4 upload','concurrent job rejection 409','unfinished artifact rejection 409',
                      'real JPEG progress','729 real NPU frames','video range 206',
                      'MP4 attachment download','JSON report','729-line trace'],
            'source_sha256':hashlib.sha256(source).hexdigest(),
            'download_bytes':len(video),'download_sha256':hashlib.sha256(video).hexdigest(),
            'summary':summary}
    Path(a.output).write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
