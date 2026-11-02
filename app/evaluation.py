"""정답·예측 비교: 손 없음과 주먹 0개는 다르다. 기존 test_cases.csv 열을 유지."""
import csv
import io
from pydantic import BaseModel, Field, model_validator
from .db import GroundTruth, Evaluation, Inference, Run

class TruthForm(BaseModel):
    status: str = 'DRAFT'
    hand_count: int | None = Field(default=None, ge=0, le=2)
    total: int | None = Field(default=None, ge=0, le=10)
    per_hand: list[int] | None = None
    lighting: str = Field(default='', max_length=200)
    conditions: str = Field(default='', max_length=500)
    occlusion: str = Field(default='', max_length=500)
    notes: str = Field(default='', max_length=2000)
    author: str = Field(default='', max_length=100)
    @model_validator(mode='after')
    def check(self):
        if self.status not in ('DRAFT', 'CONFIRMED', 'UNJUDGEABLE'):
            raise ValueError('정답 상태가 올바르지 않습니다.')
        if self.per_hand is not None:
            if len(self.per_hand) > 2 or any(n < 0 or n > 5 for n in self.per_hand):
                raise ValueError('손별 개수는 각 0~5개입니다.')
            if self.hand_count is not None and len(self.per_hand) != self.hand_count:
                raise ValueError('손 개수와 손별 개수의 항목 수가 다릅니다.')
            if self.hand_count and sum(self.per_hand) != self.total:
                raise ValueError('손별 개수 합계와 전체 개수가 다릅니다.')
        if self.status == 'CONFIRMED':
            if self.hand_count is None:
                raise ValueError('실제 손 개수를 작성하세요.')
            if self.hand_count == 0 and self.total is not None:
                raise ValueError('손이 없을 때 손가락 개수는 빈칸입니다.')
            if self.hand_count and (self.total is None or self.total > self.hand_count*5):
                raise ValueError('실제 손가락 개수를 확인하세요.')
        if self.status == 'UNJUDGEABLE' and not self.notes.strip():
            raise ValueError('판정 불가 사유를 비고에 작성하세요.')
        return self

class ReviewForm(BaseModel):
    cause: str = Field(default='', max_length=500)
    notes: str = Field(default='', max_length=2000)
    improvement: str = Field(default='', max_length=2000)
    reviewed: bool = False

def compare(truth, pred):
    if truth['status'] != 'CONFIRMED':
        return {'state': 'EXCLUDED' if truth['status'] == 'UNJUDGEABLE' else 'PENDING',
                'detection_ok': None, 'classification_ok': None, 'per_hand_ok': None, 'hand_matches':0, 'hand_samples':0,
                'error_sum': None, 'error_type': ''}
    if pred.get('status') == 'ERROR':
        return {'state': 'EVALUATED', 'detection_ok': False, 'classification_ok': False if truth.get('hand_count',0) else None,
                'per_hand_ok': None, 'hand_matches':0, 'hand_samples':0, 'error_sum': None, 'error_type': '분석오류'}
    actual, detected = truth['hand_count'], pred['hand_count']
    detection_ok = actual == detected
    if actual == 0:
        return {'state': 'EVALUATED', 'detection_ok': detection_ok, 'classification_ok': None,
                'per_hand_ok': None, 'hand_matches':0, 'hand_samples':0, 'error_sum': None, 'error_type': '' if detection_ok else '오검출'}
    predicted = pred['total']
    if detected == 0:
        error_type = '미검출'
    elif not detection_ok:
        error_type = '손개수불일치'
    elif predicted != truth['total']:
        error_type = '과대계수' if predicted > truth['total'] else '과소계수'
    else:
        error_type = ''
    per_ok, err = None, abs(predicted-truth['total']) if predicted is not None else None
    if truth.get('per_hand') is not None and detection_ok:
        per_ok = truth['per_hand'] == pred['per_hand']
        err = sum(abs(a-b) for a,b in zip(truth['per_hand'], pred['per_hand']))
        if not per_ok and not error_type:
            error_type = '손별개수불일치'
    return {'state': 'EVALUATED', 'detection_ok': detection_ok,
            'classification_ok': detection_ok and predicted == truth['total'],
            'per_hand_ok': per_ok,
            'hand_matches':sum(a==b for a,b in zip(truth['per_hand'],pred['per_hand'])) if truth.get('per_hand') is not None and detection_ok else 0,
            'hand_samples':len(truth['per_hand']) if truth.get('per_hand') is not None and detection_ok else 0,
            'error_sum': err, 'error_type': error_type}

def evaluate(s, truth, inf):
    existing = s.query(Evaluation).filter_by(truth_id=truth.id, inference_id=inf.id).first()
    if not existing:
        s.add(Evaluation(truth_id=truth.id, inference_id=inf.id,
                         payload=compare(truth.payload, inf.payload)))

def evaluate_latest(s, input_id, inf):
    truth = s.query(GroundTruth).filter_by(input_id=input_id).order_by(GroundTruth.version.desc()).first()
    if truth:
        evaluate(s, truth, inf)

COLUMNS = ['테스트id','시나리오_id','기대동작','입력파일명','예측(개)','정답(개)',
    '개수오차합','성공여부(y/n)','오류유형','원인분류','입력ID','분석실행ID','정답버전',
    '프레임번호','시각(ms)','평가상태','조명','촬영조건','가림·배경','비고','검토상태','개선의견',
    '손별예측','손별정답','손별일치','검토비고']

def csv_bytes(columns, rows):
    out = io.StringIO(newline='')
    writer = csv.writer(out)
    writer.writerow(columns)
    for row in rows:
        # Prevent a user's notes from executing as spreadsheet formulas.
        writer.writerow(["'"+v if isinstance(v,str) and v.startswith(('=','+','-','@')) else v for v in row])
    return out.getvalue().encode('utf-8-sig')
