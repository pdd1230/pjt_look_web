"""고3 초급 프로젝트 공통 입출력/모델 로딩 함수. 교사가 검토 후 수정하세요."""
from pathlib import Path
import csv
import cv2
import numpy as np

PACK_ROOT = Path(__file__).resolve().parents[1]
MODELS = PACK_ROOT / "shared" / "models"


def read_image(path):
    """한글 Windows 경로에도 대응하는 이미지 읽기."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"이미지 파일이 없습니다: {path}")
    raw = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"이미지 형식/내용을 확인하세요: {path}")
    return image


def save_image(path, image):
    """원본을 보존한 채 이미지 결과만 출력 경로에 저장."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    extension = path.suffix.lower() or ".jpg"
    ok, encoded = cv2.imencode(extension, image)
    if not ok:
        raise OSError(f"인코딩 실패: {path}")
    encoded.tofile(str(path))
    return path


def image_files(folder):
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"이미지 폴더를 확인하세요: {folder}")
    images = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"))
    if not images:
        raise FileNotFoundError(f"이미지 폴더에 jpg/png/bmp 파일이 없습니다: {folder}")
    return images


def save_csv(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def yolo():
    weights = MODELS / "yolo11n.pt"
    if not weights.is_file():
        raise FileNotFoundError("shared/models/yolo11n.pt가 없습니다. 교사가 먼저 download_models.py --only yolo 실행 또는 기존 가중치 복사")
    from ultralytics import YOLO
    return YOLO(str(weights))


def get_cascade():
    filename = MODELS / "haarcascade_frontalface_default.xml"
    if not filename.is_file():
        filename = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(str(filename))
    if cascade.empty():
        raise RuntimeError("얼굴 검출기 파일을 읽지 못했습니다")
    return cascade


def detect_faces(cascade, image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))


def video_input(path, stride, max_frames):
    """(frame_index, frame, fps)를 출력. 분석 프레임은 stride마다 하나씩."""
    if stride < 1 or max_frames < 1:
        raise ValueError("stride, max_frames는 양수여야 합니다")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"영상 파일이 없습니다: {path}")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"영상 열기 실패(코덱/파일을 확인): {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps < 1:
        fps = 30.0
    try:
        selected = 0
        frame_no = -1
        while selected < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            frame_no += 1
            if frame_no % stride:
                continue
            selected += 1
            yield frame_no, frame, fps
    finally:
        cap.release()


def video_writer(path, frame, fps):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frame.shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(max(1.0, fps)), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"영상 저장 초기화 실패(코덱 확인): {path}")
    return writer


def safe_count(boxes, wanted_names):
    """YOLO 결과의 이름과 신뢰도로 필터링. boxes는 해당 result.boxes."""
    counts = {name: 0 for name in wanted_names}
    return counts
