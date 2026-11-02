"""[담당 3: 당현청 / MediaPipe 랜드마크 추출] 손 21개 관절의 (x, y, z) 좌표를 뽑는다.

이 파일을 맡은 사람이 설명할 것:
    - extract_from_image : 사진 한 장에서 손 관절 좌표를 뽑는다.
    - _to_hands : 손이 없으면 None, 있으면 손마다 (x, y, z) 21개를 돌려준다.
    - create_landmarker : 미리 받아 둔 손 모델(models/hand_landmarker.task)을 메모리로 읽어 실행한다.

알고 있어야 할 것:
    - 좌표 x, y 는 0~1 로, 이미지 가로·세로에 대한 비율이다. 픽셀이 아니다.
    - 손이 가려져도 모델은 점을 21개 항상 채운다. 점이 가려졌는지는 알려 주지 않는다.
    - 신뢰도를 낮추면 손이 더 잡히지만, 작은 손·겹친 손은 관절 위치가 틀리기 쉽다.
    - 모델을 우리가 학습한 것이 아니다. 이미 학습된 모델을 불러 쓰기만 한다.

점수용으로 바꿀 값:
    MIN_DETECTION_CONFIDENCE: 지금 0.5. 0.3으로 낮추면 애매한 손도 잡힌다.
    MAX_NUM_HANDS: 지금 2. 한 사진에서 찾을 손의 최대 개수다.

반환 형식:
    손이 없으면 None,
    있으면 [손0, 손1, ...] 이고 각 손은 [(x, y, z) x 21개] (x, y 는 0~1 로 정규화된 좌표)
"""
import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

from config import MODEL_PATH

MIN_DETECTION_CONFIDENCE = 0.5
MAX_NUM_HANDS = 2


def create_landmarker(video=False):
    """HandLandmarker 생성. video=True 면 영상(프레임 연속) 모드, False 면 사진 모드."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"모델 파일이 없습니다: {MODEL_PATH}")
    # MediaPipe 는 한글/네트워크(UNC) 경로를 직접 못 여는 경우가 있어
    # 파이썬으로 모델을 읽은 뒤 메모리(buffer)로 넘긴다.
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_buffer=MODEL_PATH.read_bytes()),
        running_mode=vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE,
        num_hands=MAX_NUM_HANDS,
        min_hand_detection_confidence=MIN_DETECTION_CONFIDENCE,
    )
    return vision.HandLandmarker.create_from_options(options)


def _to_mp_image(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def _to_hands(result):
    """MediaPipe 결과를 [[(x,y,z)*21, ...]] 로 바꾼다. 손이 없으면 None."""
    if not result.hand_landmarks:
        return None
    return [[(lm.x, lm.y, lm.z) for lm in hand] for hand in result.hand_landmarks]


def _to_worlds(result):
    """실제 3차원 좌표(hand_world_landmarks, 단위 m, 손목 기준)를 [[(x,y,z)*21, ...]] 로 바꾼다. 없으면 None.

    화면 좌표의 z 와 달리, 손이 어느 방향으로 돌아도 길이·각도가 실제 손에 가깝게 나온다.
    (손을 옆으로 돌리면 화면 x, y 만으로는 손가락이 짧아 보이지만 이 좌표는 그대로 길다.)
    """
    if not result.hand_world_landmarks:
        return None
    return [[(lm.x, lm.y, lm.z) for lm in hand] for hand in result.hand_world_landmarks]


def extract_both_from_image(landmarker, bgr):
    """사진 한 장에서 (화면 좌표, 실제 3D 좌표) 를 같이 뽑는다. 손이 없으면 (None, None)."""
    result = landmarker.detect(_to_mp_image(bgr))
    return _to_hands(result), _to_worlds(result)


def extract_both_from_frame(landmarker, bgr, timestamp_ms):
    """영상/웹캠 프레임에서 (화면 좌표, 실제 3D 좌표) 를 같이 뽑는다. 손이 없으면 (None, None)."""
    result = landmarker.detect_for_video(_to_mp_image(bgr), int(timestamp_ms))
    return _to_hands(result), _to_worlds(result)


def extract_from_image(landmarker, bgr):
    """사진 한 장에서 추출. 손이 없으면 None."""
    return _to_hands(landmarker.detect(_to_mp_image(bgr)))


def extract_from_frame(landmarker, bgr, timestamp_ms):
    """영상/웹캠 프레임에서 추출. timestamp_ms 는 계속 증가해야 한다. 손이 없으면 None."""
    return _to_hands(landmarker.detect_for_video(_to_mp_image(bgr), int(timestamp_ms)))
