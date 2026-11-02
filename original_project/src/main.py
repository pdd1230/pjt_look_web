"""09: MediaPipe Tasks HandLandmarker + 교육용 손가락 개수 추정 규칙."""
from pathlib import Path
import argparse
import sys
import math
import cv2

ROOT = Path(__file__).resolve().parents[1]
# sys.path.insert(0, str(ROOT.parent))

# 수정: 프로젝트 루트 경로를 등록
sys.path.insert(0, str(ROOT))

from shared.common import image_files, read_image, save_csv, save_image, MODELS


def angle(a, b, c):
    """b를 중심으로 한 3점 각도 (도)."""
    u = (a.x-b.x, a.y-b.y)
    v = (c.x-b.x, c.y-b.y)
    denom = max(1e-8, math.hypot(*u)*math.hypot(*v))
    cosv = max(-1, min(1, (u[0]*v[0]+u[1]*v[1])/denom))
    return math.degrees(math.acos(cosv))


def dist(a, b):
    return math.hypot(a.x-b.x, a.y-b.y)


def count_extended(points):
    """기본 휴리스틱(0~5): 방향/가림에 따라 틀릴 수 있음."""
    wrist = points[0]
    count = 0
    for mcp, pip, tip in [(5,6,8),(9,10,12),(13,14,16),(17,18,20)]:
        if angle(points[mcp], points[pip], points[tip]) > 150 and dist(points[tip],wrist) > 1.15*dist(points[pip],wrist):
            count += 1
    # 엄지는 직선에 가깝게 펴져 있고 새끼손가락 뿌리에서 팁이 멀리 있는지 확인
    if angle(points[2], points[3], points[4]) > 150 and dist(points[4], points[17]) > 1.2*dist(points[3], points[17]):
        count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="손가락 개수 추정 (사진 1장씩)")
    parser.add_argument("--input", default=str(ROOT / "data/inputs/images"))
    args = parser.parse_args()
    model_path = MODELS / "hand_landmarker.task"
    if not model_path.is_file():
        raise FileNotFoundError("교사가 download_models.py --only hand로 hand_landmarker.task를 준비하세요")
    import mediapipe as mp
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE, num_hands=1)
    rows = []
    with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:
        for path in image_files(args.input):
            frame = read_image(path)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_image)
            count = ""
            if result.hand_landmarks:
                hand = result.hand_landmarks[0]
                count = count_extended(hand)
                h, w = frame.shape[:2]
                for p in hand:
                    cv2.circle(frame, (int(p.x*w), int(p.y*h)), 3, (0,255,0), -1)
                cv2.putText(frame, f"Fingers (estimated): {count}", (10,35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255),2)
            else:
                cv2.putText(frame, "NO HAND DETECTED", (10,35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
            save_image(ROOT / "outputs" / f"{path.stem}_hand.jpg", frame)
            rows.append({"filename":path.name, "hand_detected":int(bool(result.hand_landmarks)), "estimated_fingers":count})
            print(path.name, "손 검출:", bool(result.hand_landmarks), "추정:", count)
    save_csv(ROOT / "outputs/finger_results.csv", ["filename", "hand_detected", "estimated_fingers"], rows)
    print("규칙 기반 추정값입니다. 실제 손가락 개수와 비교해 오류를 기록하세요.")

if __name__ == "__main__":
    main()
