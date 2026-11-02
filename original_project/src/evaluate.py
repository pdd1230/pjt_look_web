"""[담당 6: 함현준 / 검증·오류 분석] 정답표와 예측표를 비교해 오차분석표(test_cases.csv)를 만든다.

함현준이 설명할 것: read_rows (CSV 읽기), main (정답과 예측 비교, 절대오차 계산)
바꿀 값: 정답표(data/ground_truth.csv)의 값 하나, 또는 오차 계산 방식(abs 부분)
맡은 파일: src/evaluate.py, data/ground_truth.csv, data/test_cases.csv

사용법 (먼저 main-2.py --mode IMAGE 로 사진 전체를 처리해 둔다):
    python src\\evaluate.py

표 구성 (수업 양식, 손가락 개수 과제에 맞게 고친 것):
    - data/ground_truth.csv : 정답표. 사람이 센 손가락 개수
    - output/finger_results.csv : 예측표. 프로그램이 센 손 개수, 손가락 개수
    - data/test_cases.csv : 오차분석표. 사진마다 검출/분류 두 줄 + 오류 처리 5건
절대오차 = |예측 수 - 실제 수|
"""
import csv

from config import OUTPUT_DIR, ROOT, TRUTH_CSV

CASE_COLUMNS = [
    "테스트id", "시나리오_id", "기대동작", "입력파일명",
    "예측(개)", "정답(개)", "개수오차합", "성공여부(y/n)", "오류유형", "원인분류",
]
CASES_CSV = ROOT / "data" / "test_cases.csv"

# 오류 처리 검증. 개수를 세는 사진이 아니라서 예측/정답/오차는 비운다.
# 성공여부 y = 프로그램이 안내 후 멈추거나 건너뛴 것까지 확인함.
EXCEPTION_ROWS = [
    ["t37", "s_37", "오류처리", "(입력 폴더가 비어 있음)", "", "", "", "y", "",
     "안내 메시지 출력 후 종료"],
    ["t38", "s_38", "오류처리", "(없는 파일)", "", "", "", "y", "",
     "입력 파일이 없습니다 출력 후 종료"],
    ["t39", "s_39", "오류처리", "(사진이 아닌 파일)", "", "", "", "y", "",
     "읽을 수 없다고 알리고 다음 파일로 건너뜀"],
    ["t40", "s_40", "오류처리", "(모델 파일 없음)", "", "", "", "y", "",
     "모델 파일이 없습니다 출력"],
    ["t41", "s_41", "검출", "(손이 없는 이미지)", "0", "0", "0", "y", "미검출",
     "손이 검출되지 않았다고 알리고 손 0개로 저장"],
]


def read_rows(path):
    if not path.exists():
        raise SystemExit(f"파일이 없습니다: {path}")
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_truth(path):
    truth = {}
    for row in read_rows(path):
        name = row.get("입력파일명") or row.get("file")
        count = row.get("정답(손가락개수)") or row.get("true_count")
        truth[name] = int(count)
    return truth


def load_pred(path):
    pred = {}
    for row in read_rows(path):
        name = row.get("입력파일명") or row.get("file")
        hands = row.get("예측(손개수)") or row.get("hands")
        fingers = row.get("예측(손가락개수)") or row.get("finger_count")
        pred[name] = (int(hands), int(fingers))
    return pred


def judge(pred_n, true_n, miss_label):
    err = abs(pred_n - true_n)
    if err == 0:
        return err, "y", "", ""
    return err, "n", miss_label, "예측 개수가 정답과 다름"


def build_cases(truth, pred):
    rows, missing = [], []
    for i, (name, true_fingers) in enumerate(truth.items(), start=1):
        if name not in pred:
            missing.append(name)
            continue
        hands, fingers = pred[name]
        test_id = f"t{i:02d}"
        scene_id = f"s_{i:02d}"
        hand_err, hand_ok, hand_kind, hand_why = judge(hands, 1, "미검출")
        finger_err, finger_ok, finger_kind, finger_why = judge(fingers, true_fingers, "오류검출")
        rows.append([test_id, scene_id, "검출", name, hands, 1, hand_err, hand_ok, hand_kind, hand_why])
        rows.append(["", "", "분류", name, fingers, true_fingers, finger_err, finger_ok, finger_kind, finger_why])
    rows.extend(EXCEPTION_ROWS)
    return rows, missing


def write_csv(path, columns, rows):
    path.parent.mkdir(exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)


def main():
    truth = load_truth(TRUTH_CSV)
    if not truth:
        raise SystemExit("정답표가 비어 있습니다. data/ground_truth.csv 를 먼저 작성하세요.")
    pred_path = OUTPUT_DIR / "finger_results.csv"
    if not pred_path.exists():
        pred_path = OUTPUT_DIR / "image_results.csv"
    pred = load_pred(pred_path)

    rows, missing = build_cases(truth, pred)
    photo_rows = [r for r in rows if r[2] == "분류" and r[3] in truth]
    for r in photo_rows:
        print(f"{r[3]}: 정답 {r[5]}, 예측 {r[4]}, 절대오차 {r[6]} ({'일치' if r[7] == 'y' else '오차'})")
    if missing:
        print("예측표에 없는 사진(먼저 main-2.py --mode IMAGE 를 실행하세요):", ", ".join(missing))

    if photo_rows:
        same = sum(1 for r in photo_rows if r[7] == "y")
        mae = sum(int(r[6]) for r in photo_rows) / len(photo_rows)
        print(f"\n비교한 사진 {len(photo_rows)}장 중 일치 {same}장 ({same / len(photo_rows) * 100:.1f}%), 평균 절대오차 {mae:.2f}")

    write_csv(OUTPUT_DIR / "evaluation.csv",
              ["file", "true_count", "pred_count", "abs_error", "result"],
              [[r[3], r[5], r[4], r[6], "일치" if r[7] == "y" else "오차"] for r in photo_rows])
    write_csv(CASES_CSV, CASE_COLUMNS, rows)
    print(f"오차분석표 저장: {CASES_CSV}")
    print(f"비교표 저장: {OUTPUT_DIR / 'evaluation.csv'}")


if __name__ == "__main__":
    main()
