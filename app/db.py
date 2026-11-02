"""SQLAlchemy 테이블. 예측/정답 JSON은 별도 테이블에 저장한다."""
from contextlib import contextmanager
from datetime import datetime, timezone
from sqlalchemy import create_engine, event, Column, Integer, String, Text, ForeignKey, JSON, UniqueConstraint
from sqlalchemy.orm import declarative_base, sessionmaker
from .settings import DATABASE_URL

def now():
    return datetime.now(timezone.utc).isoformat()
engine = create_engine(DATABASE_URL, pool_pre_ping=True,
    **({'connect_args': {'check_same_thread': False}} if str(DATABASE_URL).startswith('sqlite') else {}))
if engine.dialect.name == 'sqlite':
    @event.listens_for(engine, 'connect')
    def foreign_keys(conn, _):
        conn.execute('PRAGMA foreign_keys=ON')
Base = declarative_base()
Session = sessionmaker(bind=engine, expire_on_commit=False)

class CaptureSession(Base):
    __tablename__ = 'capture_sessions'
    id = Column(String(36), primary_key=True)
    created = Column(String(40), default=now)
    ended = Column(String(40))
    frames = Column(Integer, default=0)
    payload = Column(JSON, default=dict)

class Input(Base):
    __tablename__ = 'inputs'
    id = Column(String(36), primary_key=True)
    kind = Column(String(20), nullable=False)
    name = Column(String(255), nullable=False)
    original = Column(String(255), nullable=False)
    image = Column(String(255))
    parent_id = Column(String(36), ForeignKey('inputs.id'))
    session_id = Column(String(36), ForeignKey('capture_sessions.id'))
    created = Column(String(40), default=now)
    payload = Column(JSON, default=dict)

class GroundTruth(Base):
    __tablename__ = 'ground_truths'
    id = Column(Integer, primary_key=True, autoincrement=True)
    input_id = Column(String(36), ForeignKey('inputs.id'), nullable=False, index=True)
    version = Column(Integer, nullable=False)
    created = Column(String(40), default=now)
    payload = Column(JSON, nullable=False)
    __table_args__ = (UniqueConstraint('input_id', 'version'),)

class Run(Base):
    __tablename__ = 'analysis_runs'
    id = Column(String(36), primary_key=True)
    input_id = Column(String(36), ForeignKey('inputs.id'), nullable=False, index=True)
    status = Column(String(20), default='QUEUED')
    processed = Column(Integer, default=0)
    expected = Column(Integer, default=0)
    created = Column(String(40), default=now)
    error = Column(Text)
    payload = Column(JSON, default=dict)

class Inference(Base):
    __tablename__ = 'vision_inferences'
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(36), ForeignKey('analysis_runs.id'), nullable=False, index=True)
    frame = Column(Integer, default=0)
    timestamp_ms = Column(Integer, default=0)
    payload = Column(JSON, nullable=False)

class Evaluation(Base):
    __tablename__ = 'evaluations'
    id = Column(Integer, primary_key=True, autoincrement=True)
    truth_id = Column(Integer, ForeignKey('ground_truths.id'), nullable=False)
    inference_id = Column(Integer, ForeignKey('vision_inferences.id'), nullable=False)
    payload = Column(JSON, nullable=False)
    review = Column(JSON, default=dict)
    created = Column(String(40), default=now)
    __table_args__ = (UniqueConstraint('truth_id', 'inference_id'),)

@contextmanager
def transaction():
    with Session() as s:
        with s.begin():
            yield s

def init_db():
    Base.metadata.create_all(engine)
    # In-process workers cannot resume after a server restart: mark interrupted jobs honestly.
    with transaction() as s:
        for r in s.query(Run).filter(Run.status.in_(['QUEUED', 'RUNNING'])):
            r.status = 'FAILED'
            r.error = '서버가 종료되어 작업이 중단되었습니다. 다시 분석하세요.'
        for c in s.query(CaptureSession).filter(CaptureSession.ended.is_(None)):
            c.ended = now()

def record(obj):
    return {c.name: getattr(obj, c.name) for c in obj.__table__.columns}
