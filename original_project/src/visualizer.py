"""[담당 5: 시각화 & 결과 저장] 관절 뼈대 그리기, 숫자 표시, 이미지/CSV 저장.
    정민서 = 화면에 그리는 함수, 박현준 = 파일로 저장하는 함수. 같은 파일을 둘로 나눈다.

정민서가 설명할 것:
    - draw_skeleton : 관절 21개를 점과 선으로 그린다. HAND_CONNECTIONS 가 어떤 점끼리 잇는지 정한다.
    - draw_skeleton(img, hand, state, show_ids) : state 를 주면 펴진 손가락은 파랑, 접힌 손가락은 빨강으로 그리고,
      show_ids=True 면 관절 번호(0~20)를 작게 쓴다. 다른 관절에 가려진 관절은 번호를 숨기고, 글씨끼리 겹치면 카메라에 가까운 쪽만 쓴다.
      (가려짐 판단: 화면에서 가까이 겹치고(OCCLUDE_RADIUS) 그쪽이 카메라에 더 가까우면(OCCLUDE_DEPTH) 가려진 것)
    - draw_legend : 오른쪽 위에 'Open(파랑) / Closed(빨강)' 설명을 쓴다.
    - draw_total : 화면 왼쪽 위에 전체 손가락 수, 손 개수, FPS 를 쓴다.
    - 알고 있어야 할 것: cv2.putText 는 한글을 그리지 못해 글씨는 영문·숫자만 쓴다.
      좌표는 0~1 이라 그릴 때 이미지 가로·세로를 곱해 픽셀로 바꾼다.
    - 점수용으로 바꿀 값: TEXT_COLOR: 지금 (0, 255, 0) 초록. (0, 0, 255)로 바꾸면 글씨가 빨강이 된다.

박현준이 설명할 것:
    - save_image : 결과 그림을 output/ 에 jpg 로 저장한다. 1MB 가 넘지 않게 약 500KB 로 줄인다. 한글 경로라 cv2.imwrite 대신 바이트로 쓴다.
    - save_csv : 표의 칸 이름(columns)과 줄(rows)을 output/ 의 csv 로 저장한다.
    - 알고 있어야 할 것: 사진 예측표는 finger_results.csv (예전 이름 image_results.csv 도 같이 저장), 영상 결과는 영상 이름_result.csv 다.
      칸 이름을 바꾸면 표의 열 제목이 바뀐다. 정답표(ground_truth.csv)와는 다른 파일이다.
    - 점수용으로 바꿀 값: IMAGE_CSV_COLUMNS: 표의 칸 이름. 이름을 하나 더 넣으면 저장되는 표에 칸이 늘어난다.
"""
import csv
from pathlib import Path

import cv2
import numpy as np

# ---- 글씨 설정 (cv2.putText 는 한글을 못 그리므로 영문/숫자만) ----
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 1.2
TEXT_COLOR = (0, 255, 0)      # BGR (초록)
TEXT_POS = (10, 40)           # 화면 왼쪽 위 (x, y)
HAND_TEXT_COLOR = (255, 255, 0)

# ---- CSV 컬럼 ----
IMAGE_CSV_COLUMNS = ["file", "hands", "finger_count", "per_hand"]
FINGER_CSV_COLUMNS = ["입력파일명", "예측(손개수)", "예측(손가락개수)", "손별개수"]
VIDEO_CSV_COLUMNS = ["frame_index", "time_sec", "hands", "finger_count"]

# 손 뼈대 연결선 (랜드마크 번호 쌍)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def put_text(img, text, org, scale=1.0, color=TEXT_COLOR, thickness=2):
    """반투명한 어두운 배경 상자 위에 글씨를 쓴다 (뒤 배경과 겹쳐도 잘 보이도록)."""
    (tw, th), base = cv2.getTextSize(text, FONT, scale, thickness)
    x, y = org
    pad = max(2, int(4 * scale))
    img_h, img_w = img.shape[:2]
    x0, y0 = max(x - pad, 0), max(y - th - pad, 0)
    x1, y1 = min(x + tw + pad, img_w), min(y + base + pad, img_h)
    if x1 > x0 and y1 > y0:
        roi = img[y0:y1, x0:x1]
        img[y0:y1, x0:x1] = cv2.addWeighted(roi, 0.35, np.zeros_like(roi), 0.65, 0)
    cv2.putText(img, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)


# ---- 펴짐/접힘 색 (BGR). draw_skeleton 에 state 를 넘기면 적용된다 ----
OPEN_COLOR = (255, 90, 0)      # 파랑: 펴진 손가락
CLOSED_COLOR = (0, 0, 255)     # 빨강: 접힌 손가락
PALM_COLOR = (200, 200, 200)   # 손바닥 선(손목~손가락 뿌리)은 회색

# 손가락별 관절 번호 / 선 (엄지 1~4, 검지 5~8, 중지 9~12, 약지 13~16, 새끼 17~20)
FINGER_JOINTS = {"thumb": (1, 2, 3, 4), "index": (5, 6, 7, 8), "middle": (9, 10, 11, 12),
                 "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}
_FINGER_OF_JOINT = {j: name for name, js in FINGER_JOINTS.items() for j in js}
# ---- 관절 번호 표시 설정 ----
ID_FONT_SCALE = 0.45          # 관절 번호 글씨 크기 (작게)
OCCLUDE_RADIUS = 0.16         # 화면에서 다른 관절이 이 거리(손바닥 크기 x 값) 안에 있고
OCCLUDE_DEPTH = 0.10          # 그 관절이 카메라에 이만큼(손바닥 크기 x 값) 더 가까우면 '가려진 관절'로 보고 번호를 숨긴다


def _joint_color(a, b, state):
    """선 (a, b) 의 색: 손가락 선이면 펴짐 파랑 / 접힘 빨강, 손바닥 선이면 회색."""
    finger = _FINGER_OF_JOINT.get(b if b in _FINGER_OF_JOINT else a)
    is_palm_line = (a, b) in ((0, 5), (5, 9), (9, 13), (13, 17), (0, 17))
    if is_palm_line or finger is None:
        return PALM_COLOR
    return OPEN_COLOR if state.get(finger) else CLOSED_COLOR


def _visible_joint_ids(pts, hand):
    """가려지지 않은 관절 번호 목록. 다른 손가락·손바닥에 가려진 관절의 번호를 숨겨 화면이 지저분해지는 걸 막는다.

    판단: 화면에서 가까이(OCCLUDE_RADIUS) 겹쳐 있고, 그쪽 관절이 카메라에 더 가까우면(z 가 더 작으면) 가려진 것으로 본다.
    (MediaPipe z 는 작을수록 카메라에 가깝다. 손목이 0.)
    """
    palm = max(1.0, float(np.hypot(pts[0][0] - pts[9][0], pts[0][1] - pts[9][1])))
    palm_z = max(1e-6, float(np.hypot(hand[0][0] - hand[9][0], hand[0][1] - hand[9][1])))   # z 는 정규화 좌표 단위
    visible = []
    for i in range(21):
        hidden = False
        for j in range(21):
            if i == j:
                continue
            near = np.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]) < OCCLUDE_RADIUS * palm
            closer = hand[j][2] < hand[i][2] - OCCLUDE_DEPTH * palm_z
            if near and closer:
                hidden = True
                break
        if not hidden:
            visible.append(i)
    return visible


def draw_joint_ids(img, pts, hand):
    """보이는 관절 옆에 작은 번호(0~20)를 쓴다. 가려진 관절은 건너뛰고, 글씨끼리 겹치면 카메라에 가까운 쪽만 쓴다."""
    s = _big_scale(img)
    scale = ID_FONT_SCALE * max(0.8, s)
    th = max(1, int(s))
    img_h, img_w = img.shape[:2]
    placed = []   # 이미 쓴 글씨 상자 (x0, y0, x1, y1)
    # 카메라에 가까운(z 작은) 관절부터 자리를 잡는다
    for i in sorted(_visible_joint_ids(pts, hand), key=lambda k: hand[k][2]):
        text = str(i)
        (tw, tht), _ = cv2.getTextSize(text, FONT, scale, th)
        x, y = pts[i]
        g = int(6 * s)
        for dx, dy in ((g, -g // 2), (g, tht + g), (-tw - g, -g // 2), (-tw - g, tht + g)):   # 오른쪽위, 오른쪽아래, 왼쪽위, 왼쪽아래
            x0, y0 = x + dx, y + dy - tht
            box = (x0 - 1, y0 - 1, x0 + tw + 1, y0 + tht + 3)
            if box[0] < 0 or box[1] < 0 or box[2] > img_w or box[3] > img_h:
                continue
            if any(box[0] < p[2] and box[2] > p[0] and box[1] < p[3] and box[3] > p[1] for p in placed):
                continue
            placed.append(box)
            org = (x0, y0 + tht)
            cv2.putText(img, text, org, FONT, scale, (0, 0, 0), th + 2, cv2.LINE_AA)       # 검은 테두리
            cv2.putText(img, text, org, FONT, scale, (255, 255, 255), th, cv2.LINE_AA)     # 흰 글씨
            break


def draw_skeleton(img, hand, state=None, show_ids=False):
    """21개 관절 점과 뼈대 선을 그린다. 반환: 픽셀 좌표 리스트.

    state: {"thumb": True/False, "index": ...} 를 주면 펴진 손가락은 파랑, 접힌 손가락은 빨강으로 그린다.
           (안 주면 예전처럼 초록 선 + 빨간 점)
    show_ids: True 면 관절 번호를 작게 쓴다. 가려진 관절의 번호는 숨긴다.
    """
    h, w = img.shape[:2]
    pts = [(int(x * w), int(y * h)) for x, y, _ in hand]
    s = _scale(img)  # 작은 이미지에서는 점과 선도 작게
    line_w = max(1, int(2 * s))
    dot_r = max(1, int(4 * s))
    if state is None:
        for a, b in HAND_CONNECTIONS:
            cv2.line(img, pts[a], pts[b], (0, 255, 0), line_w)
        for p in pts:
            cv2.circle(img, p, dot_r, (0, 0, 255), -1)
    else:
        s = _big_scale(img)
        line_w, dot_r = max(2, int(3 * s)), max(3, int(4 * s))
        for a, b in HAND_CONNECTIONS:
            cv2.line(img, pts[a], pts[b], _joint_color(a, b, state), line_w)
        for i, p in enumerate(pts):
            finger = _FINGER_OF_JOINT.get(i)
            color = PALM_COLOR if finger is None or i in (5, 9, 13, 17) else (OPEN_COLOR if state.get(finger) else CLOSED_COLOR)
            cv2.circle(img, p, dot_r + 1, (255, 255, 255), -1)   # 흰 테두리
            cv2.circle(img, p, dot_r - 1 if dot_r > 2 else dot_r, color, -1)
    if show_ids:
        draw_joint_ids(img, pts, hand)
    return pts


def draw_legend(img):
    """오른쪽 위에 색 설명(파랑=펴짐, 빨강=접힘)을 쓴다."""
    s = _big_scale(img)
    scale = 0.55 * s
    th = max(1, int(2 * s))
    w = img.shape[1]
    (tw, _), _ = cv2.getTextSize("Closed", FONT, scale, th)
    x = w - tw - int(14 * s)
    put_text(img, "Open", (x, int(26 * s)), scale, OPEN_COLOR, th)
    put_text(img, "Closed", (x, int(52 * s)), scale, CLOSED_COLOR, th)


def _big_scale(img):
    """색 표시·관절 번호용 배율: 640px 에서 1.0, 큰 사진(1280px 등)은 가로 크기에 비례해 키운다(최대 2.5배)."""
    return max(0.3, min(2.5, img.shape[1] / 640))


def _scale(img):
    """이미지 가로 크기에 맞춘 글씨 배율 (640px 이상이면 1.0, 작은 이미지는 축소)."""
    return max(0.3, min(1.0, img.shape[1] / 640))


def draw_hand_count(img, pts, n):
    """손목 근처에 그 손의 손가락 수를 표시."""
    s = _scale(img)
    put_text(img, str(n), (pts[0][0] - 10, pts[0][1] + int(50 * s)), 1.5 * s,
             HAND_TEXT_COLOR, max(1, int(2 * s)))


def draw_total(img, total, hands, fps=None):
    """화면 상단에 전체 손가락 수(와 손 개수, FPS)를 표시."""
    s = _scale(img)
    x, y = int(TEXT_POS[0] * s) + 2, int(TEXT_POS[1] * s) + 6
    th = max(1, int(2 * s))
    put_text(img, f"Fingers: {total}", (x, y), FONT_SCALE * s, TEXT_COLOR, th)
    info = f"Hands: {hands}" + (f"  FPS: {fps:.0f}" if fps is not None else "")
    put_text(img, info, (x, y + int(40 * s)), 0.7 * s, HAND_TEXT_COLOR, th)


OUTPUT_TARGET_BYTES = 500 * 1024   # 결과 사진 목표 용량
OUTPUT_MAX_BYTES = 1024 * 1024      # 이 용량은 넘기지 않는다
OUTPUT_MAX_WIDTH = 1280


def save_image(path, img):
    """결과 사진을 jpg 로 저장한다. 약 500KB, 최대 1MB. 한글 경로에서도 동작."""
    path = Path(path)
    if path.suffix.lower() != ".jpg":
        path = path.with_suffix(".jpg")
    path.parent.mkdir(parents=True, exist_ok=True)

    h, w = img.shape[:2]
    if w > OUTPUT_MAX_WIDTH:
        img = cv2.resize(img, (OUTPUT_MAX_WIDTH, int(h * OUTPUT_MAX_WIDTH / w)), interpolation=cv2.INTER_AREA)

    best = None
    for quality in range(90, 39, -5):
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            continue
        best = buf
        if len(buf) <= OUTPUT_TARGET_BYTES:
            break
    while best is not None and len(best) > OUTPUT_MAX_BYTES and img.shape[1] > 640:
        img = cv2.resize(img, (int(img.shape[1] * 0.85), int(img.shape[0] * 0.85)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            best = buf
    if best is None:
        raise IOError(f"이미지 인코딩 실패: {path}")
    best.tofile(str(path))
    return path


def save_csv(path, columns, rows):
    """CSV 저장 (엑셀에서 한글이 안 깨지도록 utf-8-sig)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(columns)
        w.writerows(rows)
