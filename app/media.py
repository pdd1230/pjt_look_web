"""업로드 검증. 원본은 그대로 보존하고 EXIF 적용/분석용 파일을 따로 생성."""
import io
import warnings
from uuid import uuid4
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from fastapi import HTTPException
from .settings import STORAGE
Image.MAX_IMAGE_PIXELS = 30_000_000

def path(name):
    p = (STORAGE / name).resolve()
    if STORAGE not in p.parents:
        raise HTTPException(400, '잘못된 파일 경로')
    return p

def write_image(name, image):
    ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise ValueError('이미지 저장 실패')
    path(name).write_bytes(encoded.tobytes())

def decode_image(data):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as im:
                if im.format not in ('JPEG', 'PNG', 'BMP', 'WEBP'):
                    raise ValueError('JPG, PNG, BMP, WEBP 사진을 사용하세요.')
                original_size = list(im.size)
                orientation = im.getexif().get(274, 1)
                im = ImageOps.exif_transpose(im).convert('RGB')
                im.thumbnail((960, 1280), Image.Resampling.LANCZOS)
                bgr = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
        return bgr, {'original_size': original_size, 'analysis_size': [bgr.shape[1], bgr.shape[0]],
                     'exif_orientation': orientation, 'mirrored': False}
    except (ValueError, UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as e:
        raise HTTPException(400, f'사진을 읽을 수 없습니다: {e}') from e

def register_image(data, name, kind='IMAGE', parent_id=None, session_id=None, extra=None):
    from .db import Input, GroundTruth, transaction
    frame, meta = decode_image(data)
    iid = str(uuid4())
    suffix = Path(name).suffix.lower()
    if suffix not in ('.jpg', '.jpeg', '.png', '.bmp', '.webp'):
        suffix = '.jpg'
    original, analysis = f'{iid}_original{suffix}', f'{iid}_input.jpg'
    files = [original, analysis]
    try:
        path(original).write_bytes(data)
        write_image(analysis, frame)
        meta.update(extra or {})
        with transaction() as s:
            s.add(Input(id=iid, kind=kind, name=Path(name.replace('\\', '/')).name[:255],
                original=original, image=analysis, parent_id=parent_id, session_id=session_id, payload=meta))
            s.flush()
            s.add(GroundTruth(input_id=iid, version=1, payload={
                'status':'DRAFT', 'hand_count':None, 'total':None, 'per_hand':None,
                'lighting':'', 'conditions':'', 'occlusion':'', 'notes':'', 'author':''}))
    except Exception:
        for f in files:
            path(f).unlink(missing_ok=True)
        raise
    return iid

def read_image(name):
    im = cv2.imdecode(np.frombuffer(path(name).read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    if im is None:
        raise ValueError('저장된 사진을 읽지 못했습니다.')
    return im

def resize_frame(frame):
    h, w = frame.shape[:2]
    scale = min(1, 960/w, 1280/h)
    return cv2.resize(frame, (max(2, int(w*scale)//2*2), max(2, int(h*scale)//2*2)))
