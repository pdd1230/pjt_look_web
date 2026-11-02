# 09. 손가락 개수 알아맞히기 — 학생용 실행 안내

**대상**: 고3·복사 실행 수준 / 5주 / 조별 6~7명 / CPU / 웹캠 불필요

## 1. 목적
손 관절 위치를 찾고 펼친 손가락 개수를 휴리스틱으로 추정한다.

## 2. 모델과 데이터
- 사전 학습 모델/검출기: `shared/models/hand_landmarker.task`
- 입력: 손 사진 20장
- 보관 위치: `data/images/` 또는 `data/videos/` (현재 실제 사진·영상은 포함하지 않음)
- 데이터 조건: 0~5개 손가락과 다른 각도
- 개인정보·저작권 및 촬영 동의 확인. 원본 외부 공개 금지.

## 3. 처음 실행 (Windows PowerShell)
1. 최상위 폴더로 이동: `cd C:\CV_StudentProjects` (실제 위치에 맞게 수정)
2. 최상위 `README_먼저읽기.md`를 보고 Python 3.11용 `.venv_mp` 가상환경을 먼저 생성.
3. 필요한 경우 `python download_models.py --only all`로 모델 받기(각 프로젝트별 모델표 참조).
4. `data/images`에 실제 입력 파일을 넣고, 프로젝트 폴더의 `run.bat`를 실행.

```powershell
# 직접 실행할 경우 (프로젝트 폴더에서):
..\.venv_mp\Scripts\python.exe source\main.py
```

- 현재 `data/`에는 연습용 실물 사진·영상이 없습니다. **교사가 준비한 허가된 데이터를 넣어야 합니다.**
- 입력 파일 이름이 다르면 `--input` 옵션을 이용하거나 프로그램 상단의 기본 경로를 바꾸세요.
- 자세한 옵션: `..\.venv_mp\Scripts\python.exe source\main.py --help`

## 4. 출력과 성공 기준
- 결과는 `outputs/`에 **원본과 다른 이름**으로 생성됩니다.
- 사진별 실제 수 vs 예측 수를 학생이 직접 기록하거나 비교합니다.
- 2주차까지 입력→분석→파일 저장을 재현해야 합니다.
- 한계: 손 회전, 가림, 엄지 판별로 오차 발생

## 5. 검증 방법
1. `test_cases.csv`의 시나리오별 입력을 준비합니다.
2. `data/ground_truth.csv`의 `actual_...` 칸에 사람이 센 **진짜 정답만** 작성합니다.
3. 프로그램을 실행하고 `outputs/`의 CSV·이미지·영상을 확인합니다.
4. `test_cases.csv`의 `actual_result`, `pass_fail`, `note`를 팀이 채웁니다.
5. 실패 사례 3개와 사용한 입력 조건을 발표에 포함합니다.

## 6. 필수와 선택 구분
- 필수(A): **입력→검출/분석→저장** 정상 실행.
- 권장(B): 주제별 핵심 기능을 적용하고 실제 정답과 비교.
- 선택(C): 여러 입력 처리, 그래프, GUI, 정밀 지표.

## 7. 오류 처리 팁
- `ModuleNotFoundError`: `run.bat`가 올바른 가상환경을 가리키는지 확인.
- `FileNotFoundError`: `data/`와 모델 경로를 확인.
- 이미지가 검출되지 않으면 밝고 크게 나온 입력으로 먼저 테스트.
- MP4 파일이 재생되지 않으면 Windows 코덱/출력 코덱을 확인하고 교사에게 문의.

## 8. 제출물
`source/main.py`, `outputs/` 결과물 2개 이상, 정답표, `test_cases.csv`, 수정 기록, 3분 시연.

공통 내용은 `../README_먼저읽기.md`, 과제 기능은 `PROJECT.md`를 참조합니다.

## 9. 예상 출력 파일과 첫 번째 수정 실습

입력명이 기본 예시(`img_001.jpg` 또는 `video_01.mp4`)일 때 `outputs/`에 다음 파일이 생성됩니다.

- `img_001_hand.jpg`
- `finger_results.csv`

**학생 수정 실습:** 손가락 판정 함수의 `150`(도)을 바꿔 추정 결과를 비교하세요.
