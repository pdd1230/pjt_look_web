"""프로젝트 루트에서 python scripts/check_environment.py 실행."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import hashlib
import importlib.metadata
from sqlalchemy import text
from app.db import engine
from app.settings import MODEL_PATH
from app.recognition import Recognizer

def main():
    print('Python:',sys.version)
    for p in ['fastapi','uvicorn','SQLAlchemy','PyMySQL','mediapipe','opencv-contrib-python','numpy','Pillow']:
        print(p,importlib.metadata.version(p))
    print('Model exists:',MODEL_PATH.is_file())
    if not MODEL_PATH.is_file():
        raise SystemExit('models/hand_landmarker.task가 없습니다. ZIP 전체를 풀었는지 확인하세요.')
    print('Model SHA256:',hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest())
    r=Recognizer();r.close();print('Model load: OK')
    with engine.connect() as c:
        c.execute(text('SELECT 1'))
    print('Database connection: OK ('+engine.dialect.name+')')
    print('이 검사는 실제 웹캠·브라우저·인식 정확도 테스트를 대신하지 않습니다.')
if __name__=='__main__':
    main()
