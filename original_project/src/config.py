"""프로젝트 공통 경로 설정 (모두가 같이 쓰는 파일, 보통 수정하지 않는다)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR = ROOT / "data" / "inputs"          # 입력 데이터 (담당 2)
IMAGE_DIR = INPUT_DIR / "images"              # 손 사진
VIDEO_DIR = INPUT_DIR / "videos"              # 손 영상
TRUTH_CSV = ROOT / "data" / "ground_truth.csv"  # 정답표 (담당 6)
OUTPUT_DIR = ROOT / "output"                  # 결과 저장 폴더
MODEL_PATH = ROOT / "models" / "hand_landmarker.task"
