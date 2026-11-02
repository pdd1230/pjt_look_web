"""LOOK local classroom web service. Run with one Uvicorn worker."""
import base64
import copy
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from uuid import uuid4
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Form
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from .db import (init_db, transaction, record, now, engine, Input, GroundTruth,
                 Run, Inference, Evaluation, CaptureSession)
from .settings import ROOT, MODEL_PATH, MAX_IMAGE_BYTES, MAX_VIDEO_BYTES, ALGORITHM
from .media import path, decode_image, register_image, write_image, read_image
from .evaluation import TruthForm, ReviewForm, compare, evaluate, evaluate_latest, COLUMNS, csv_bytes
from .recognition import Recognizer
from .jobs import analyze_image, video_job, new_run, VIDEO_SLOT, POOL

log = logging.getLogger('look')
streams = {}
streams_lock = Lock()
truth_lock = Lock()

@asynccontextmanager
async def lifespan(app):
    init_db()  # A wrong MySQL setting fails startup clearly; no silent fallback database.
    yield
    with streams_lock:
        for item in streams.values():
            with item['lock']:
                item['recognizer'].close()
        streams.clear()
    POOL.shutdown(wait=True)

app = FastAPI(title='LOOK 손가락 인식 실험실', version='1.0.0', lifespan=lifespan)
app.mount('/static', StaticFiles(directory=ROOT/'app/static'), name='static')
templates = Jinja2Templates(directory=ROOT/'app/templates')

@app.exception_handler(SQLAlchemyError)
async def db_error(request, exc):
    log.error('Database operation failed: %s', type(exc).__name__)
    return JSONResponse(status_code=503, content={'detail':'DB 저장·조회에 실패했습니다. MySQL 실행과 .env 설정을 확인하세요.'})

@app.exception_handler(Exception)
async def server_error(request, exc):
    log.exception('Request failed')
    return JSONResponse(status_code=500, content={'detail':'처리 중 오류가 발생했습니다. 터미널 로그를 확인하세요.'})

@app.get('/')
def home(request: Request):
    return templates.TemplateResponse(request=request, name='index.html')

def require(s, model, key):
    obj = s.get(model,key)
    if obj is None:
        raise HTTPException(404,'기록을 찾을 수 없습니다.')
    return obj

def uploaded_bytes(file, limit):
    data = file.file.read(limit+1)
    file.file.close()
    if not data:
        raise HTTPException(400,'빈 파일입니다.')
    if len(data)>limit:
        raise HTTPException(413, f'파일 용량 제한은 {limit//1024**2}MB입니다.')
    return data

def download_csv(name, columns, rows):
    return Response(csv_bytes(columns, rows), media_type='text/csv; charset=utf-8',
                    headers={'Content-Disposition': f'attachment; filename="{name}"'})

def jpeg_url(image):
    ok, jpg = cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,85])
    if not ok:
        raise ValueError('프레임 인코딩 실패')
    return 'data:image/jpeg;base64,'+base64.b64encode(jpg).decode('ascii')

@app.get('/api/health')
def health():
    with engine.connect() as c:
        c.execute(text('SELECT 1'))
    return {'database':engine.dialect.name, 'model_exists':MODEL_PATH.exists(),
            'algorithm':ALGORITHM, 'mode':'local-single-worker'}

@app.post('/api/images', status_code=201)
def image_upload(file: UploadFile = File(...)):
    iid = register_image(uploaded_bytes(file,MAX_IMAGE_BYTES), file.filename or 'image.jpg')
    return {'id':iid}

@app.get('/api/inputs')
def inputs():
    with transaction() as s:
        result = []
        for i in s.query(Input).order_by(Input.created.desc()).limit(1000):
            item = record(i)
            t = s.query(GroundTruth).filter_by(input_id=i.id).order_by(GroundTruth.version.desc()).first()
            r = s.query(Run).filter_by(input_id=i.id).order_by(Run.created.desc()).first()
            item['truth'] = record(t) if t else None
            item['run'] = record(r) if r else None
            result.append(item)
        return result

@app.get('/api/inputs/{iid}')
def detail(iid: str):
    with transaction() as s:
        inp = require(s,Input,iid)
        truths = s.query(GroundTruth).filter_by(input_id=iid).order_by(GroundTruth.version.desc()).all()
        runs = s.query(Run).filter_by(input_id=iid).order_by(Run.created.desc()).all()
        data = []
        for r in runs:
            rr = record(r)
            # Detailed frame history is paginated through /runs/{rid}/frames for videos.
            rr['inferences'] = [record(i) for i in s.query(Inference).filter_by(run_id=r.id).order_by(Inference.frame).limit(1)]
            data.append(rr)
        return {'input':record(inp), 'truths':[record(t) for t in truths], 'runs':data}

@app.post('/api/inputs/{iid}/truth',status_code=201)
def save_truth(iid: str, form: TruthForm):
    with truth_lock, transaction() as s:
        inp = require(s,Input,iid)
        if inp.kind == 'VIDEO':
            raise HTTPException(400,'영상에서 평가 프레임을 먼저 추출하세요.')
        s.query(Input).filter_by(id=iid).with_for_update().first()
        last = s.query(GroundTruth).filter_by(input_id=iid).order_by(GroundTruth.version.desc()).first()
        truth = GroundTruth(input_id=iid,version=last.version+1 if last else 1,payload=form.model_dump())
        s.add(truth)
        s.flush()
        for inf in s.query(Inference).join(Run).filter(Run.input_id==iid):
            evaluate(s,truth,inf)
        return record(truth)

@app.post('/api/inputs/{iid}/analyze',status_code=201)
def run_image(iid: str):
    with transaction() as s:
        if require(s,Input,iid).kind == 'VIDEO':
            raise HTTPException(400,'영상 분석 버튼을 사용하세요.')
    rid = analyze_image(iid)
    with transaction() as s:
        return record(s.get(Run,rid))

@app.post('/api/videos',status_code=201)
def upload_video(file: UploadFile = File(...)):
    name = Path((file.filename or '').replace('\\','/')).name
    if Path(name).suffix.lower() not in ('.mp4','.mov','.avi','.mkv'):
        raise HTTPException(400,'MP4, MOV, AVI, MKV를 사용하세요. MP4를 권장합니다.')
    data = uploaded_bytes(file,MAX_VIDEO_BYTES)
    iid = str(uuid4())
    fname = iid+'_original'+Path(name).suffix.lower()
    try:
        path(fname).write_bytes(data)
        cap = cv2.VideoCapture(str(path(fname)))
        ok, _ = cap.read()
        cap.release()
        if not ok:
            raise HTTPException(400,'영상을 읽을 수 없습니다. 손상 또는 코덱을 확인하세요.')
        with transaction() as s:
            s.add(Input(id=iid,kind='VIDEO',name=name[:255],original=fname,
                        payload={'bytes':len(data),'mirrored':False}))
    except Exception:
        path(fname).unlink(missing_ok=True)
        raise
    return {'id':iid}

@app.post('/api/videos/{iid}/analyze',status_code=202)
def start_video(iid: str):
    if not VIDEO_SLOT.acquire(blocking=False):
        raise HTTPException(409,'다른 영상이 분석 중입니다. 완료 후 다시 실행하세요.')
    try:
        with transaction() as s:
            if require(s,Input,iid).kind != 'VIDEO':
                raise HTTPException(400,'영상 입력이 아닙니다.')
            r = new_run(s,iid)
            r.status = 'QUEUED'
            rid = r.id
        POOL.submit(video_job,rid)
    except Exception:
        VIDEO_SLOT.release()
        raise
    return {'id':rid}

@app.get('/api/runs/{rid}')
def get_run(rid: str):
    with transaction() as s:
        return record(require(s,Run,rid))

@app.get('/api/runs/{rid}/frames')
def frames(rid: str, offset: int=0, limit: int=30):
    if offset < 0 or not 1 <= limit <= 100:
        raise HTTPException(400,'offset>=0, limit=1~100')
    with transaction() as s:
        require(s,Run,rid)
        return [record(i) for i in s.query(Inference).filter_by(run_id=rid).order_by(Inference.frame).offset(offset).limit(limit)]

@app.post('/api/videos/{iid}/capture',status_code=201)
def capture_video(iid: str, frame_number: int=Form(...)):
    if frame_number < 0:
        raise HTTPException(400,'프레임 번호는 0 이상입니다.')
    with transaction() as s:
        inp = require(s,Input,iid)
        if inp.kind != 'VIDEO':
            raise HTTPException(400,'영상이 아닙니다.')
        name = inp.original
    cap = cv2.VideoCapture(str(path(name)))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            raise HTTPException(400,'FPS를 확인할 수 없습니다.')
        # Sequential decode ensures the same numbered frame as the analysis pass.
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if count and frame_number >= count:
            raise HTTPException(400,'영상 범위 밖의 프레임입니다.')
        image = None
        for _ in range(frame_number+1):
            ok, image = cap.read()
            if not ok:
                raise HTTPException(400,'이 프레임은 읽을 수 없습니다.')
        source_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        frame_ts = round(source_ms) if source_ms > 0 or frame_number == 0 else round(frame_number*1000/fps)
        ok, data = cv2.imencode('.png',image)
        if not ok:
            raise HTTPException(400,'프레임 변환 실패')
        captured_id = register_image(data.tobytes(),f'frame_{frame_number}.png','VIDEO_FRAME',
            parent_id=iid,extra={'frame':frame_number,'timestamp_ms':frame_ts})
        # Carry each already-stored video prediction to this evaluation input. No image re-inference.
        with transaction() as s:
            source = s.query(Inference).join(Run).filter(Run.input_id==iid,Inference.frame==frame_number,
                                                       Run.status=='DONE').all()
            for original_inf in source:
                r = new_run(s,captured_id)
                r.status,r.processed,r.expected = 'DONE',1,1
                r.payload = {**r.payload,'source_run':original_inf.run_id,'source_inference':original_inf.id}
                pred = copy.deepcopy(original_inf.payload)
                inf=Inference(run_id=r.id,frame=frame_number,timestamp_ms=original_inf.timestamp_ms,payload=pred)
                s.add(inf)
                s.flush()
                evaluate_latest(s,captured_id,inf)
        return {'id':captured_id}
    finally:
        cap.release()

@app.get('/files/{name}')
def get_file(name: str, download: bool=False):
    p = path(name)
    if not p.is_file():
        raise HTTPException(404,'파일이 없습니다.')
    return FileResponse(p,filename=p.name if download else None)

def end_stream(sid, item):
    with item['lock']:
        item['recognizer'].close()
        with transaction() as s:
            row = s.get(CaptureSession,sid)
            if row:
                row.ended,row.frames = now(),item['frames']
                row.payload = {'elapsed_seconds':round(time.monotonic()-item['started'],2),
                    'processing_ms_sum':round(item['processing_ms'],2),'mirrored':False}

@app.post('/api/webcam/sessions',status_code=201)
def start_webcam():
    with streams_lock:
        expired = [sid for sid,item in streams.items() if time.monotonic()-item['active']>120]
        for sid in expired:
            end_stream(sid,streams.pop(sid))
        if len(streams)>=2:
            raise HTTPException(409,'카메라 세션은 최대 2개입니다. 다른 세션을 정지하세요.')
        rec = Recognizer(video=True)
        sid = str(uuid4())
        try:
            with transaction() as s:
                s.add(CaptureSession(id=sid,payload={'mirrored':False}))
        except Exception:
            rec.close()
            raise
        streams[sid] = {'recognizer':rec,'lock':Lock(),'frames':0,'last':None,
            'started':time.monotonic(),'active':time.monotonic(),'processing_ms':0.0}
        return {'id':sid,'mirrored':False}

def stream_item(sid):
    with streams_lock:
        if sid not in streams:
            raise HTTPException(404,'세션이 종료되었습니다. 카메라를 다시 시작하세요.')
        item = streams[sid]
        if not item['lock'].acquire(blocking=False):
            raise HTTPException(429,'이전 프레임 처리 중입니다.')
        return item

@app.post('/api/webcam/sessions/{sid}/frame')
def webcam_frame(sid: str, file: UploadFile=File(...), timestamp_ms:int=Form(...)):
    item = stream_item(sid)
    try:
        image, _ = decode_image(uploaded_bytes(file,MAX_IMAGE_BYTES))
        pred, annotated = item['recognizer'].process(image,timestamp_ms)
        item['frames']+=1
        item['active']=time.monotonic()
        item['processing_ms']+=pred['processing_ms']
        item['last']=(image.copy(),annotated.copy(),copy.deepcopy(pred),timestamp_ms)
        return {'prediction':pred,'image':jpeg_url(annotated),'frames':item['frames']}
    finally:
        item['lock'].release()

@app.post('/api/webcam/sessions/{sid}/capture',status_code=201)
def webcam_capture(sid: str):
    item = stream_item(sid)
    try:
        if item['last'] is None:
            raise HTTPException(400,'첫 분석 결과가 나온 뒤 캡처하세요.')
        image,annotated,pred,ts = item['last']
        ok,data = cv2.imencode('.png',image)
        if not ok:
            raise HTTPException(400,'캡처 저장 실패')
        iid = register_image(data.tobytes(),'webcam_capture.png','WEBCAM_CAPTURE',session_id=sid,
                             extra={'frame':item['frames']-1,'timestamp_ms':ts,'mirrored':False})
        out = None
        try:
            with transaction() as s:
                r=new_run(s,iid)
                r.status,r.processed,r.expected='DONE',1,1
                out=f'{r.id}_result.jpg'
                write_image(out,annotated)
                result=copy.deepcopy(pred)
                result['image']=out
                inf=Inference(run_id=r.id,frame=item['frames']-1,timestamp_ms=ts,payload=result)
                s.add(inf)
                s.flush()
                evaluate_latest(s,iid,inf)
                s.get(CaptureSession,sid).frames=item['frames']
        except Exception:
            if out:
                path(out).unlink(missing_ok=True)
            raise
        return {'id':iid}
    finally:
        item['lock'].release()

@app.delete('/api/webcam/sessions/{sid}')
def stop_webcam(sid: str):
    with streams_lock:
        item=streams.pop(sid,None)
        if item:
            end_stream(sid,item)
    return {'stopped':True}

@app.get('/api/evaluations')
def evaluations(history: bool=False):
    with transaction() as s:
        rows=[]
        for e,t,inf,r,i in s.query(Evaluation,GroundTruth,Inference,Run,Input).join(
            GroundTruth,Evaluation.truth_id==GroundTruth.id).join(Inference,Evaluation.inference_id==Inference.id).join(
            Run,Inference.run_id==Run.id).join(Input,Run.input_id==Input.id).order_by(Evaluation.id.desc()):
            if not history:
                latest_t=s.query(GroundTruth).filter_by(input_id=i.id).order_by(GroundTruth.version.desc()).first()
                latest_r=s.query(Run).filter_by(input_id=i.id).order_by(Run.created.desc()).first()
                if t.id!=latest_t.id or r.id!=latest_r.id:
                    continue
            rows.append({'id':e.id,'input_id':i.id,'name':i.name,'run_id':r.id,
                'truth_version':t.version,'truth':t.payload,'prediction':inf.payload,
                'frame':inf.frame,'timestamp_ms':inf.timestamp_ms,'result':e.payload,'review':e.review})
        return rows

@app.patch('/api/evaluations/{eid}/review')
def review(eid:int, form:ReviewForm):
    with transaction() as s:
        e=require(s,Evaluation,eid)
        e.review=form.model_dump()
        return record(e)

@app.get('/api/exports/test_cases.csv')
def test_csv(errors_only:bool=False,history:bool=False):
    rows=[]
    for e in evaluations(history):
        t,p,v=e['truth'],e['prediction'],e['result']
        for mode in ('검출','분류'):
            success=v['detection_ok'] if mode=='검출' else v['classification_ok']
            # Per-hand errors must remain visible even when the total happens to match.
            if errors_only and not (success is False or (mode=='분류' and v['per_hand_ok'] is False)):
                continue
            pred=p['hand_count'] if mode=='검출' else p['total']
            actual=t.get('hand_count') if mode=='검출' else t.get('total')
            error=abs(pred-actual) if mode=='검출' and actual is not None and pred is not None else v['error_sum']
            if v['state']!='EVALUATED':
                error=None
            rows.append([f"t{e['id']}-{mode}",e['input_id'],mode,e['name'],pred,actual,error,
                '' if success is None else ('y' if success else 'n'),v['error_type'],e['review'].get('cause',''),
                e['input_id'],e['run_id'],e['truth_version'],e['frame'],e['timestamp_ms'],v['state'],
                t.get('lighting',''),t.get('conditions',''),t.get('occlusion',''),t.get('notes',''),
                '검토완료' if e['review'].get('reviewed') else '미검토',e['review'].get('improvement',''),
                '/'.join(map(str,p['per_hand'])),'/'.join(map(str,t.get('per_hand') or [])),v['per_hand_ok'],e['review'].get('notes','')])
    return download_csv('errors.csv' if errors_only else 'test_cases.csv',COLUMNS,rows)

@app.get('/api/exports/ground_truth.csv')
def truths_csv():
    columns=['입력ID','입력파일명','버전','작성시각','상태','실제손개수','정답(개)','손별개수',
             '조명','촬영조건','가림·배경','비고','작성자']
    with transaction() as s:
        rows=[]
        for t,i in s.query(GroundTruth,Input).join(Input,GroundTruth.input_id==Input.id).order_by(GroundTruth.id):
            p=t.payload
            rows.append([i.id,i.name,t.version,t.created,p['status'],p.get('hand_count'),p.get('total'),
                '/'.join(map(str,p.get('per_hand') or [])),p.get('lighting',''),p.get('conditions',''),
                p.get('occlusion',''),p.get('notes',''),p.get('author','')])
    return download_csv('ground_truth.csv',columns,rows)

@app.get('/api/exports/predictions.csv')
def predictions_csv():
    with transaction() as s:
        rows=[]
        for inf,r,i in s.query(Inference,Run,Input).join(Run,Inference.run_id==Run.id).join(Input,Run.input_id==Input.id):
            p=inf.payload
            rows.append([i.id,i.name,r.id,r.status,inf.frame,inf.timestamp_ms,p['status'],
                p['hand_count'],p['total'],'/'.join(map(str,p['per_hand'])),p.get('processing_ms'),
                p.get('model_sha256'),p.get('algorithm')])
    return download_csv('predictions.csv',['입력ID','입력파일명','run_id','run_status','frame','timestamp_ms',
        'status','hands','finger_count','per_hand','processing_ms','model_sha256','algorithm'],rows)

@app.get('/api/inferences/{inf_id}/landmarks.csv')
def landmarks_csv(inf_id:int):
    with transaction() as s:
        p=require(s,Inference,inf_id).payload
        rows=[]
        for h in p.get('hands',[]):
            for j,point in enumerate(h['landmarks']):
                world=h['world_landmarks'][j] if h['world_landmarks'] else [None]*3
                rows.append([h['slot'],j,*point,*world])
    return download_csv('landmarks.csv',['hand','index','x','y','z','wx','wy','wz'],rows)

@app.get('/api/inferences/{inf_id}/metrics.csv')
def metrics_csv(inf_id:int):
    with transaction() as s:
        p=require(s,Inference,inf_id).payload
        rows=[]
        for h in p.get('hands',[]):
            for d in h['diagnosis']:
                rows.append([h['slot'],d['finger'],d['open'],h['state'][d['finger']],d['reason'],
                    *[d['metrics'].get(k) for k in ('pip2d','pip3d','ext3d','wrist_ratio','spread2d','center2d','ip2d','spread3d')]])
    return download_csv('metrics.csv',['hand','finger','raw_open','stable_open','reason','pip2d','pip3d','ext3d',
        'wrist_ratio','spread2d','center2d','ip2d','spread3d'],rows)
