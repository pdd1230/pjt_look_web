"""환경 변수와 프로젝트 내부 저장 경로. 한글 경로도 지원한다."""
import os
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import URL
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / '.env')
STORAGE = Path(os.getenv('LOOK_STORAGE', str(ROOT / 'storage'))).resolve()
STORAGE.mkdir(parents=True, exist_ok=True)
MODEL_PATH = ROOT / 'models' / 'hand_landmarker.task'
DATABASE_URL = os.getenv('DATABASE_URL') or URL.create(
    'mysql+pymysql', username=os.getenv('DB_USER', 'look_user'),
    password=os.getenv('DB_PASSWORD', ''), host=os.getenv('DB_HOST', '127.0.0.1'),
    port=int(os.getenv('DB_PORT', '3306')), database=os.getenv('DB_NAME', 'look_db'),
    query={'charset': 'utf8mb4'})
MAX_IMAGE_BYTES = int(os.getenv('MAX_IMAGE_MB', '20')) * 1024**2
MAX_VIDEO_BYTES = int(os.getenv('MAX_VIDEO_MB', '250')) * 1024**2
MAX_VIDEO_SECONDS = int(os.getenv('MAX_VIDEO_SECONDS', '300'))
ALGORITHM = 'LOOK-main-2-web-v1'
