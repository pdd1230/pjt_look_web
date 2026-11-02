"""[담당 2: 이다영 / 입력 데이터] 사진/영상 읽기, 크기 조정, 프레임 추출 간격.

이 파일을 맡은 사람이 설명할 것:
    - read_image : 사진 파일을 읽어 이미지로 만든다.
    - resize_image : 가로가 RESIZE_MAX_WIDTH 보다 크면 비율을 유지한 채 줄인다.
    - 사진이 들어가는 폴더는 data/inputs/images, 영상은 data/inputs/videos 다.

알고 있어야 할 것:
    - cv2.imread 는 한글·네트워크 경로에서 실패해서, 파일을 바이트로 읽은 뒤 이미지로 푼다.
    - 평가용 사진은 한 장에 손을 크게, 가급적 한 손만. 손가락 0~5개가 골고루 있어야 한다.
    - .heic 는 읽지 못한다. jpg 또는 png 로 넣는다.

점수용으로 바꿀 값:
    RESIZE_MAX_WIDTH: 지금 960. 480으로 바꾸면 저장되는 사진이 더 작아진다.
    FRAME_INTERVAL: 지금 1(모든 장면). 3으로 바꾸면 영상에서 3장마다 1장만 본다.
    INPUT_MAX_BYTES: 지금 1MB. 입력 폴더의 jpg 가 이보다 크면 파일을 약 500KB 로 줄여 덮어쓴다.
"""
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np

from config import IMAGE_DIR, VIDEO_DIR

RESIZE_MAX_WIDTH = 960   # 0 이면 크기 조정을 하지 않는다
FRAME_INTERVAL = 1       # 1 = 모든 프레임, 3 = 3프레임마다 1장
INPUT_MAX_BYTES = 1024 * 1024       # 이 용량을 넘는 jpg 만 줄인다
INPUT_TARGET_BYTES = 500 * 1024     # 줄인 뒤 목표 용량
INPUT_MAX_WIDTH = 1280              # 저장 파일의 가로 상한 (판정용 960 과 별개)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def list_files(folder, exts):
    """폴더 안에서 확장자가 맞는 파일 목록(이름순)."""
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in exts)


def shrink_input_image(path):
    """입력 폴더의 jpg/jpeg 가 1MB 를 넘으면 약 500KB 로 줄여 같은 파일에 덮어쓴다."""
    path = Path(path)
    if path.suffix.lower() not in {".jpg", ".jpeg"} or not path.is_file():
        return
    try:
        if not path.resolve().is_relative_to(Path(IMAGE_DIR).resolve()):
            return
        if path.stat().st_size <= INPUT_MAX_BYTES:
            return
    except OSError:
        return

    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return
    h, w = img.shape[:2]
    if w > INPUT_MAX_WIDTH:
        img = cv2.resize(img, (INPUT_MAX_WIDTH, int(h * INPUT_MAX_WIDTH / w)), interpolation=cv2.INTER_AREA)

    best = None
    for quality in range(90, 39, -5):
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            continue
        best = buf
        if len(buf) <= INPUT_TARGET_BYTES:
            break
    while best is not None and len(best) > INPUT_MAX_BYTES and img.shape[1] > 640:
        img = cv2.resize(img, (int(img.shape[1] * 0.85), int(img.shape[0] * 0.85)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            best = buf
    if best is None:
        return
    path.write_bytes(best.tobytes())
    print(f"{path.name}: {len(data) / 1024:.0f}KB -> {path.stat().st_size / 1024:.0f}KB 로 줄여 저장했습니다.")


def list_images(folder=IMAGE_DIR):
    images = list_files(folder, IMAGE_EXTS)
    if Path(folder).resolve() == Path(IMAGE_DIR).resolve():
        for path in images:
            shrink_input_image(path)
    return images


def list_videos(folder=VIDEO_DIR):
    return list_files(folder, VIDEO_EXTS)


def resize_image(img, max_width=None):
    """가로가 max_width 보다 크면 비율을 유지하며 줄인다."""
    max_width = RESIZE_MAX_WIDTH if max_width is None else max_width
    h, w = img.shape[:2]
    if max_width and w > max_width:
        scale = max_width / w
        img = cv2.resize(img, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def read_image(path):
    """이미지 읽기 (한글 경로에서 cv2.imread 가 실패하므로 numpy 로 우회)."""
    shrink_input_image(path)
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {path}")
    return resize_image(img)


def open_video(path):
    """비디오 파일 열기. 한글/네트워크 경로에서 실패하면 임시 폴더로 복사해서 연다.

    반환: (VideoCapture, 임시폴더 또는 None)  - 임시폴더는 다 쓴 뒤 삭제해야 한다.
    """
    cap = cv2.VideoCapture(str(path))
    if cap.isOpened():
        return cap, None
    cap.release()
    tmp = Path(tempfile.mkdtemp(prefix="finger_"))
    local = tmp / ("input" + Path(path).suffix)
    shutil.copy(path, local)
    return cv2.VideoCapture(str(local)), tmp


def open_webcam(index=0):
    """웹캠 열기. 프레임이 실제로 한 장 들어와야 성공이다. 실패하면 None.

    Windows 는 DirectShow 로 먼저 연다. 인덱스로 열리지 않으면 Media Foundation 으로 다시 시도한다.
    열리기만 하고 첫 프레임이 실패하는 경우(다른 프로그램이 카메라를 쓰는 중 등)가 있어서 몇 번 읽는다.
    """
    for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF):
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            continue
        for _ in range(15):
            ok, frame = cap.read()
            if ok and frame is not None:
                return cap
        cap.release()
    return None
