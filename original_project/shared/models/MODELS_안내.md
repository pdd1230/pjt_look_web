# 사전 학습 모델 파일 안내

| 프로젝트 | 필요한 파일 | 현재 상태 | 취득 방법 |
|---|---|---|---|
| 01/02/04/05/06/07 | `yolo11n.pt` | **포함하지 않음** | `python download_models.py --only yolo` 또는 이전 실습 파일 복사 |
| 03/08 | `haarcascade_frontalface_default.xml` | **이 패키지에 포함** | OpenCV 패키지의 사전 학습 검출기 복사본 |
| 09 | `hand_landmarker.task` | **포함하지 않음** | `python download_models.py --only hand` |
| 10 | `pose_landmarker_lite.task` | **포함하지 않음** | `python download_models.py --only pose` |

이 파일들은 `shared/models/`에 한 번만 보관하고, 10개 프로그램이 공유합니다. 
모델은 **새로 학습시키지 않습니다**. 온라인에서 받은 모델 파일의 라이선스·재배포 조건은 원 제공처에서 확인하세요.

- YOLO11: https://docs.ultralytics.com/models/yolo11/
- Hand: https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
- Pose: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
- Haar: https://github.com/opencv/opencv/tree/4.x/data/haarcascades

배포 후에는 실행 전 교사용 PC에서 최신 의존성 및 라이선스 정책을 확인하세요.
