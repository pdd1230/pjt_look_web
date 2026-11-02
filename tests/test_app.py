"""실제 MediaPipe 모델을 사용하는 통합 테스트. 전용 임시 SQLite만 사용한다."""
import os
import io
import time
import tempfile
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import pytest

TMP = tempfile.TemporaryDirectory(prefix='look_tests_')
os.environ['DATABASE_URL'] = 'sqlite:///'+str(Path(TMP.name)/'test.db')
os.environ['LOOK_STORAGE'] = str(Path(TMP.name)/'storage')
from fastapi.testclient import TestClient
from app.main import app
from app.evaluation import compare, TruthForm
from app.settings import ROOT, MODEL_PATH
from app.db import transaction, Run, Inference

@pytest.fixture(scope='module')
def client():
    with TestClient(app) as c:
        yield c

def blank():
    im=Image.new('RGB',(640,480),'white')
    b=io.BytesIO();im.save(b,format='PNG');return b.getvalue()

def upload(client,data=None,name='테스트.png'):
    r=client.post('/api/images',files={'file':(name,data or blank(),'image/png')})
    assert r.status_code==201,r.text
    return r.json()['id']

def truth(client,iid,**values):
    f={'status':'CONFIRMED','hand_count':0,'total':None,'lighting':'밝음','notes':'테스트'}
    f.update(values)
    return client.post(f'/api/inputs/{iid}/truth',json=f)

def test_home_and_health(client):
    assert '손가락 인식 실험실' in client.get('/').text
    assert client.get('/api/health').json()['model_exists']
    assert client.get('/static/app.js').status_code==200

def test_empty_bad_and_oversized(client):
    assert client.post('/api/images',files={'file':('x.jpg',b'bad')}).status_code==400
    assert client.post('/api/images',files={'file':('x.jpg',b'')}).status_code==400
    assert client.post('/api/images',files={'file':('x.jpg',b'x'*(20*1024**2+1))}).status_code==413
    assert client.get('/files/not-found.jpg').status_code==404

def test_full_image_history_and_exports(client):
    iid=upload(client)
    d=client.get('/api/inputs/'+iid).json()
    assert d['truths'][0]['payload']['status']=='DRAFT'
    assert truth(client,iid).status_code==201
    r=client.post('/api/inputs/'+iid+'/analyze').json()
    assert r['status']=='DONE',r
    d=client.get('/api/inputs/'+iid).json()
    p=d['runs'][0]['inferences'][0]['payload']
    assert p['status']=='NO_HAND' and p['total'] is None and p['hand_count']==0
    assert client.get('/files/'+p['image']).status_code==200
    assert truth(client,iid,hand_count=1,total=0,per_hand=[0]).status_code==201
    e=[e for e in client.get('/api/evaluations').json() if e['input_id']==iid][0]
    assert e['result']['error_type']=='미검출'
    assert e['result']['classification_ok'] is False and e['result']['error_sum'] is None
    assert len(client.get('/api/evaluations?history=true').json())>=2
    reviewed=client.patch(f"/api/evaluations/{e['id']}/review",json={'cause':'수기 원인','reviewed':True})
    assert reviewed.status_code==200
    # A second run preserves first run and truth version references.
    assert client.post('/api/inputs/'+iid+'/analyze').json()['status']=='DONE'
    assert len(client.get('/api/inputs/'+iid).json()['runs'])==2
    for name in ['test_cases.csv','ground_truth.csv','predictions.csv']:
        out=client.get('/api/exports/'+name)
        assert out.status_code==200 and out.content.startswith(b'\xef\xbb\xbf')
    assert '미검출' in client.get('/api/exports/test_cases.csv?errors_only=true').text

def test_validation_and_two_hand_cancellation():
    t={'status':'CONFIRMED','hand_count':2,'total':5,'per_hand':[2,3]}
    p={'status':'DETECTED','hand_count':2,'total':5,'per_hand':[3,2]}
    e=compare(t,p)
    assert e['classification_ok'] and not e['per_hand_ok'] and e['error_sum']==2
    assert e['error_type']=='손별개수불일치'
    with pytest.raises(ValueError):
        TruthForm(status='CONFIRMED',hand_count=0,total=0)
    with pytest.raises(ValueError):
        TruthForm(status='CONFIRMED',hand_count=1,total=6)
    with pytest.raises(ValueError):
        TruthForm(status='CONFIRMED',hand_count=2,total=5,per_hand=[2,2])
    assert compare({'status':'DRAFT'},p)['state']=='PENDING'
    assert compare({'status':'UNJUDGEABLE'},p)['state']=='EXCLUDED'

def test_large_exif_image(client):
    im=Image.new('RGB',(3023,4032),'white')
    ex=Image.Exif();ex[274]=6
    b=io.BytesIO();im.save(b,format='JPEG',exif=ex)
    iid=upload(client,b.getvalue(),'손_01.jpg')
    d=client.get('/api/inputs/'+iid).json()['input']
    assert d['payload']['original_size']==[3023,4032]
    assert d['payload']['exif_orientation']==6
    assert d['payload']['analysis_size'][0]<=960
    assert client.get('/files/'+d['original']).content==b.getvalue()

def test_original_modules_and_actual_model(client):
    for name in ('finger_classifier.py','smoothing.py','visualizer.py'):
        assert (ROOT/'app/vision'/name).read_bytes()==(ROOT/'original_project/src'/name).read_bytes()
    # Supplied examples already contain drawings: smoke testing only, not accuracy evidence.
    for photo in sorted((ROOT/'original_project/docs/repro').glob('*.jpg')):
        iid=upload(client,photo.read_bytes(),photo.name)
        r=client.post('/api/inputs/'+iid+'/analyze').json()
        assert r['status']=='DONE',r
        d=client.get('/api/inputs/'+iid).json()
        inf=d['runs'][0]['inferences'][0]
        assert inf['payload']['status'] in ('DETECTED','NO_HAND')
        if inf['payload']['hands']:
            assert inf['payload']['total']==sum(inf['payload']['per_hand'])
            for kind in ('landmarks','metrics'):
                assert client.get(f"/api/inferences/{inf['id']}/{kind}.csv").status_code==200

def test_model_failure_records_error(client,monkeypatch):
    import app.vision.landmark_extractor as le
    iid=upload(client)
    monkeypatch.setattr(le,'MODEL_PATH',ROOT/'models/missing.task')
    r=client.post('/api/inputs/'+iid+'/analyze').json()
    assert r['status']=='FAILED' and r['error']
    inf=client.get('/api/inputs/'+iid).json()['runs'][0]['inferences'][0]
    assert inf['payload']['status']=='ERROR'

def test_webcam_session_and_capture(client):
    sid=client.post('/api/webcam/sessions').json()['id']
    for ts in [0,0,100,200]: # duplicate client timestamps are safely made monotonic
        r=client.post(f'/api/webcam/sessions/{sid}/frame',data={'timestamp_ms':ts},
            files={'file':('frame.jpg',blank())})
        assert r.status_code==200,r.text
        assert r.json()['prediction']['status']=='NO_HAND'
    r=client.post(f'/api/webcam/sessions/{sid}/capture')
    assert r.status_code==201,r.text
    d=client.get('/api/inputs/'+r.json()['id']).json()
    assert d['input']['session_id']==sid and d['runs'][0]['inferences'][0]['payload']['total'] is None
    assert client.delete(f'/api/webcam/sessions/{sid}').status_code==200
    assert client.post(f'/api/webcam/sessions/{sid}/capture').status_code==404
    # Stop and restart creates a fresh model/filter instance.
    sid2=client.post('/api/webcam/sessions').json()['id']
    assert sid!=sid2
    client.delete('/api/webcam/sessions/'+sid2)

@pytest.mark.parametrize('n',[8,12])
def test_video_all_frames_capture_csv(client,n):
    movie=Path(TMP.name)/f'movie{n}.mp4'
    writer=cv2.VideoWriter(str(movie),cv2.VideoWriter_fourcc(*'mp4v'),10,(320,240))
    assert writer.isOpened()
    for i in range(n):writer.write(np.full((240,320,3),240-i,dtype=np.uint8))
    writer.release()
    r=client.post('/api/videos',files={'file':(movie.name,movie.read_bytes())})
    assert r.status_code==201,r.text
    iid=r.json()['id']
    r=client.post(f'/api/videos/{iid}/analyze')
    assert r.status_code==202,r.text
    rid=r.json()['id']
    for _ in range(200):
        result=client.get('/api/runs/'+rid).json()
        if result['status'] in ('DONE','FAILED'):break
        time.sleep(.05)
    assert result['status']=='DONE',result
    assert result['processed']==n
    assert len(client.get(f'/api/runs/{rid}/frames').json())==n
    csv=client.get('/files/'+result['payload']['csv']).text
    assert len(csv.strip().splitlines())==n+1
    assert client.get('/files/'+result['payload']['video']).status_code==200
    capture=client.post(f'/api/videos/{iid}/capture',data={'frame_number':3})
    assert capture.status_code==201,capture.text
    d=client.get('/api/inputs/'+capture.json()['id']).json()
    assert d['input']['payload']['timestamp_ms']==300
    assert d['runs'][0]['inferences'][0]['frame']==3
    assert d['runs'][0]['payload']['source_run']==rid

def test_invalid_video(client):
    assert client.post('/api/videos',files={'file':('x.mp4',b'bad')}).status_code==400
    assert client.post('/api/videos',files={'file':('x.txt',blank())}).status_code==400

def test_foreign_keys_and_persistence(client):
    from sqlalchemy.exc import IntegrityError
    with pytest.raises(IntegrityError):
        with transaction() as s:
            s.add(Inference(run_id='missing',payload={}))
    from app.db import Session
    with Session() as s:
        assert s.query(Run).count()>0

def test_session_limit_and_isolation(client):
    a=client.post('/api/webcam/sessions').json()['id']
    b=client.post('/api/webcam/sessions').json()['id']
    assert client.post('/api/webcam/sessions').status_code==409
    for ts in (100,200):
        r=client.post(f'/api/webcam/sessions/{a}/frame',data={'timestamp_ms':ts},files={'file':('f.png',blank())})
        assert r.status_code==200
    r=client.post(f'/api/webcam/sessions/{b}/frame',data={'timestamp_ms':0},files={'file':('f.png',blank())})
    assert r.status_code==200 and r.json()['frames']==1
    assert client.delete('/api/webcam/sessions/'+a).status_code==200
    assert client.delete('/api/webcam/sessions/'+b).status_code==200

def test_mysql_ddl_compiles():
    from sqlalchemy.schema import CreateTable
    from sqlalchemy.dialects.mysql import dialect
    from app.db import Base
    for table in Base.metadata.sorted_tables:
        sql=str(CreateTable(table).compile(dialect=dialect()))
        assert 'CREATE TABLE' in sql
    # Compilation verifies schema syntax only, not an actual MySQL connection.
