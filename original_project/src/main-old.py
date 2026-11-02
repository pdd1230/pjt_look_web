"""[담당 1: 당형청 / 조장·통합·발표] 각 담당자의 모듈을 하나로 합치는 메인 실행 파일.

[2주차] 사진 모드 완성: 사진 -> 랜드마크 -> 손가락 개수 판정 -> 뼈대·숫자 그리기 -> output/ 저장.
    - 영상/웹캠, 정확도 검증은 다음 주차에 추가한다.

이 파일을 맡은 사람이 설명할 것:
    - analyze : 손 좌표를 받아 손가락 수를 세고, 그림이 있으면 뼈대와 숫자를 그린다.
    - 데이터 흐름: data_loader -> landmark_extractor -> finger_classifier -> visualizer
    - 손이 없으면 landmark_extractor 가 None 을 주고, analyze 가 그 None 을 받아 0개로 처리한다.
    - 모델은 우리가 학습한 것이 아니라 MediaPipe 가 미리 학습해 둔 손 모델을 실행한 것이다.

알고 있어야 할 것:
    - 이 파일은 검출·판정 규칙을 직접 만들지 않고, 다른 파일의 함수를 순서대로 호출만 한다.

사용법 (프로젝트 루트에서):
    python src/main.py                                     # data/inputs/images 전체 -> output/ + image_results.csv
    python src/main.py --input data/inputs/images/img.png  # 사진 1장
"""
import argparse
from pathlib import Path

import data_loader
import landmark_extractor
import finger_classifier
import visualizer
from config import OUTPUT_DIR, ROOT


def analyze(hands, img=None):
    """손 좌표 목록(또는 None)을 판정하고, img 가 있으면 뼈대와 숫자를 그린다.

    반환: (전체 손가락 수, 손별 개수 리스트)
    """
    if hands is None:          # 손 미검출 처리
        return 0, []
    per_hand = []
    for hand in hands:
        n, _ = finger_classifier.count_fingers(hand)
        per_hand.append(n)
        if img is not None:
            pts = visualizer.draw_skeleton(img, hand)
            visualizer.draw_hand_count(img, pts, n)
    return sum(per_hand), per_hand


def run_image(input_path):
    if input_path:
        path = Path(input_path)
        images = [path if path.is_absolute() else ROOT / path]
    else:
        images = data_loader.list_images()
    if not images:
        raise SystemExit("data/inputs/images 폴더에 손 사진을 먼저 넣어 주세요.")

    rows = []
    with landmark_extractor.create_landmarker() as lm:
        for path in images:
            img = data_loader.read_image(path)
            hands = landmark_extractor.extract_from_image(lm, img)
            total, per_hand = analyze(hands, img)
            visualizer.draw_total(img, total, len(per_hand))
            visualizer.save_image(OUTPUT_DIR / f"{path.stem}_result.jpg", img)
            rows.append([path.name, len(per_hand), total, "/".join(map(str, per_hand))])
            print(f"{path.name}: 손 {len(per_hand)}개, 손가락 {total}개 {per_hand}")

    csv_path = OUTPUT_DIR / "image_results.csv"
    visualizer.save_csv(csv_path, visualizer.IMAGE_CSV_COLUMNS, rows)
    pred_path = OUTPUT_DIR / "finger_results.csv"
    visualizer.save_csv(pred_path, visualizer.FINGER_CSV_COLUMNS, rows)
    print(f"CSV 저장: {pred_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=None, help="사진 1장 경로 (생략하면 images 폴더 전체)")
    args = p.parse_args()
    run_image(args.input)


if __name__ == "__main__":
    main()
