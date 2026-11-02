"""단일 프로세스 영상 작업. 한 작업만 허용해 웹캠용 실행 여유를 남긴다."""
import csv
import time
from concurrent.futures import ThreadPoolExecutor
from threading import BoundedSemaphore
from uuid import uuid4
import cv2
from .db import Input, Run, Inference, transaction, now
from .media import path, read_image, write_image, resize_frame
from .recognition import Recognizer
from .evaluation import evaluate_latest
from .settings import MAX_VIDEO_SECONDS, ALGORITHM
POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix='look-video')
VIDEO_SLOT = BoundedSemaphore(1)

def new_run(s, input_id):
    r = Run(id=str(uuid4()), input_id=input_id, status='RUNNING', payload={'algorithm': ALGORITHM})
    s.add(r)
    s.flush()
    return r

def analyze_image(iid):
    result_file = None
    with transaction() as s:
        inp = s.get(Input, iid)
        image_name = inp.image
        rid = new_run(s, iid).id
    recognizer = None
    try:
        recognizer = Recognizer()
        result, annotated = recognizer.process(read_image(image_name))
        result_file = f'{rid}_result.jpg'
        write_image(result_file, annotated)
        result['image'] = result_file
        with transaction() as s:
            inf = Inference(run_id=rid, payload=result)
            s.add(inf)
            s.flush()
            evaluate_latest(s, iid, inf)
            r = s.get(Run, rid)
            r.status, r.processed, r.expected = 'DONE', 1, 1
            r.payload = {**r.payload, 'finished': now(), 'result_image': result_file}
    except Exception as e:
        if result_file:
            path(result_file).unlink(missing_ok=True)
        with transaction() as s:
            r = s.get(Run, rid)
            r.status, r.error = 'FAILED', str(e)
            inf = Inference(run_id=rid, payload={'status':'ERROR', 'error':str(e), 'total':None,
                                               'per_hand':[], 'hand_count':None})
            s.add(inf)
            s.flush()
            evaluate_latest(s, iid, inf)
    finally:
        if recognizer:
            recognizer.close()
    return rid

VIDEO_COLUMNS = ['frame_index','time_sec','hands','finger_count','per_hand','status','processing_ms','display_smoothed_total']

def video_job(rid):
    cap = writer = recognizer = None
    outputs = []
    started = time.perf_counter()
    try:
        with transaction() as s:
            r = s.get(Run, rid)
            r.status = 'RUNNING'
            inp = s.get(Input, r.input_id)
            original = inp.original
        cap = cv2.VideoCapture(str(path(original)))
        if not cap.isOpened():
            raise ValueError('영상 코덱을 열 수 없습니다. MP4(H.264) 파일을 확인하세요.')
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not 0 < fps <= 240:
            raise ValueError('유효한 영상 FPS를 읽지 못했습니다.')
        expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if expected and expected/fps > MAX_VIDEO_SECONDS:
            raise ValueError(f'영상은 {MAX_VIDEO_SECONDS}초 이하만 가능합니다.')
        recognizer = Recognizer(video=True)
        out_name, csv_name = f'{rid}_result.mp4', f'{rid}_frames.csv'
        outputs = [out_name, csv_name]
        with transaction() as s:
            r = s.get(Run, rid)
            r.expected = max(0, expected)
            r.payload = {**r.payload, 'fps':fps, 'frame_interval':1, 'audio_preserved':False}
        index, batch, last_source_ms, timestamp_fallbacks = 0, [], -1, 0
        with path(csv_name).open('w', encoding='utf-8-sig', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(VIDEO_COLUMNS)
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if index/fps > MAX_VIDEO_SECONDS:
                    raise ValueError('실제 읽은 영상 길이가 제한을 초과했습니다.')
                frame = resize_frame(frame)
                if writer is None:
                    h,w = frame.shape[:2]
                    writer = cv2.VideoWriter(str(path(out_name)), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))
                    if not writer.isOpened():
                        raise ValueError('MP4 결과 인코더를 열 수 없습니다.')
                source_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                if source_ms >= 0 and (index == 0 or source_ms > last_source_ms):
                    ts = round(source_ms)
                else:
                    ts = max(round(index*1000/fps),last_source_ms+1)
                    timestamp_fallbacks += 1
                last_source_ms = ts
                pred, annotated = recognizer.process(frame, ts)
                writer.write(annotated)
                csv_writer.writerow([index, round(ts/1000,3), pred['hand_count'], pred['total'],
                    '/'.join(map(str,pred['per_hand'])), pred['status'], pred['processing_ms'], pred['display_smoothed_total']])
                batch.append(Inference(run_id=rid, frame=index, timestamp_ms=ts, payload=pred))
                index += 1
                if len(batch) >= 15:
                    with transaction() as s:
                        s.add_all(batch)
                        s.get(Run, rid).processed = index
                    batch = []
            if not index:
                raise ValueError('영상에 읽을 수 있는 프레임이 없습니다.')
            # Truncated decode is a failure instead of pretending the whole movie was analyzed.
            if expected > index+1:
                raise ValueError(f'영상 디코딩이 중단되었습니다: {index}/{expected}프레임')
            with transaction() as s:
                s.add_all(batch)
                s.get(Run,rid).processed = index
        writer.release()
        writer = None
        check = cv2.VideoCapture(str(path(out_name)))
        readable, _ = check.read()
        check.release()
        if not readable:
            raise ValueError('저장된 결과 영상을 다시 읽을 수 없습니다.')
        seconds = time.perf_counter()-started
        with transaction() as s:
            r = s.get(Run,rid)
            r.status = 'DONE'
            r.payload = {**r.payload, 'video':out_name, 'csv':csv_name, 'finished':now(),
                'elapsed_seconds':round(seconds,3), 'analysis_fps':round(index/seconds,2),
                'decoded_frames':index, 'timestamp_basis':'OpenCV POS_MSEC; frame/fps fallback',
                'timestamp_fallbacks':timestamp_fallbacks, 'browser_playback':'mp4v: download if browser cannot play'}
    except Exception as e:
        if writer:
            writer.release()
            writer = None
        for name in outputs:
            path(name).unlink(missing_ok=True)
        with transaction() as s:
            r = s.get(Run,rid)
            r.status, r.error = 'FAILED', str(e)
    finally:
        if cap:
            cap.release()
        if writer:
            writer.release()
        if recognizer:
            recognizer.close()
        VIDEO_SLOT.release()
