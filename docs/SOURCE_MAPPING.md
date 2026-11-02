# 첨부 소스 반영표

기준: pjt_LOOK(2).zip / src/main-2.py. original_project에 첨부 전체 보존.

| 원본 | 웹 구현 및 유지 내용 |
|---|---|
| finger_classifier.py | 파일 전체 원본과 동일, 엄지/손가락 판정·가림·주먹 예외·임계값 유지 |
| smoothing.py | 파일 전체 동일, One Euro/MultiHandSmoother 적용 |
| visualizer.py | 파일 전체 동일, 펴짐 파랑/접힘 빨강/관절 0~20/가려진 번호 숨김/범례 |
| landmark_extractor.py | 모델 import만 app.settings로 변경, 메모리 모델 로딩·IMAGE/VIDEO 추출 유지 |
| _to_pixels | x×폭·y×높이·USE_DEPTH=False 유지 |
| _depths / USE_HIDDEN_THUMB | 가림 판정용 z×폭 별도 사용 |
| USE_WORLD=True | 3D world 좌표를 판정에 사용 |
| FingerStabilizer | 원본 window=7, need=4, 스트림별 별도 인스턴스 |
| SMOOTH_FRAMES=5 | display_smoothed_total 진단 필드로 보존. 평가와 주 표시값은 손별 안정화 합계로 통일하여 손별 수·합계 불일치 방지 |
| 손 개수 변화 | 원본 좌표 필터 초기화 + 손가락 상태/합계 이력 초기화로 이전 손 상태 혼입 방지 |
| finger_diagnosis / d | 모든 검출 손의 판정 이유·측정값 JSON/CSV, 안정화 전후 구분 |
| s 캡처 | 같은 최근 분석 프레임 원본·결과·좌표·예측 연결 저장 |
| 사진 원본 압축 | 원본 덮어쓰기 제거, 분석본에 EXIF·축소 적용 |
| 미검출 0개 | NO_HAND+total=NULL로 상태 의미 분리, 주먹 0개와 구분 |
| 서버 OpenCV 카메라 | 사용자 브라우저 getUserMedia로 변경 |
| 웹캠 cv2.flip | 반전 없음, 정답·예측 모두 화면 왼쪽 순서 |
| 영상 interval=1 | 모든 프레임 순서대로, 증가 timestamp, 원본 fps로 결과 저장 |
| evaluate.py | 입력 FK·정답 버전·추론 실행별 비교로 확장 |
| 과거 CSV | 원본 자료로만 보존, 신규 정확도에 사용하지 않음 |

판정 임계값 재튜닝이나 큰 판정 알고리즘 변경은 하지 않았다. 웹 입력·원본 보존·저장 일관성·평가를 위한 상태 처리를 추가했다.

DB 연결: inputs → ground_truths(여러 버전), inputs → analysis_runs(여러 실행) → vision_inferences(사진/영상 프레임별), ground_truths+vision_inferences → evaluations. capture_sessions는 선택 웹캠 캡처와 연결한다. 영상 캡처는 inputs.parent_id로 원본 영상과 연결하며 기존 예측의 원본 실행/추론 ID·프레임 번호·시각을 보존한다.
