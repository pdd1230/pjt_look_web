"""[담당 1: 당형청 / 조장·통합·발표] 각 담당자의 모듈을 하나로 합치는 메인 실행 파일.

[버전] main-2.py = 최신 버전 (사진 + 영상 + 웹캠, 손가락별 안정화 FingerStabilizer, 픽셀 좌표 변환, s 키 원본/좌표 저장).
       사진 모드만 있는 구버전은 main-1.py.

이 파일을 맡은 사람이 설명할 것:
    - analyze : 손 좌표를 받아 손가락 수를 세고, 그림이 있으면 뼈대와 숫자를 그린다.
    - 데이터 흐름: data_loader -> landmark_extractor -> finger_classifier -> visualizer -> evaluate
    - MODE 를 바꾸면 입력이 사진 / 영상 / 웹캠 중 어디로 바뀌는지.

알고 있어야 할 것:
    - 이 파일은 검출·판정 규칙을 직접 만들지 않고, 다른 파일의 함수를 순서대로 호출만 한다.
    - 손이 없으면 landmark_extractor 가 None 을 주고, analyze 가 그 None 을 받아 0개로 처리한다.
    - 모델은 우리가 학습한 것이 아니라 MediaPipe 가 미리 학습해 둔 손 모델을 실행한 것이다.

점수용으로 바꿀 값:
    MODE: 지금 "IMAGE"(사진). "VIDEO"로 바꾸면 영상으로 실행된다.

흐름:  입력(data_loader) -> 랜드마크 추출(landmark_extractor) -> 손가락 판정(finger_classifier)
       -> 표시·저장(visualizer) -> (사진 모드) 검증(evaluate)

사용법 (프로젝트 루트에서):
    python src/main-2.py                       # 위의 MODE 값대로 실행
    python src/main-2.py --mode WEBCAM         # MODE 를 명령줄로 덮어쓰기
    python src/main-2.py --mode IMAGE          # data/inputs/images 전체 -> output/ + image_results.csv
    python src/main-2.py --mode VIDEO --input data/inputs/videos/a.mp4 --save
    python src/main-2.py --mode WEBCAM --camera 1

키 (VIDEO/WEBCAM): q 또는 ESC 종료, s 현재 화면 저장
"""
import argparse
import shutil
import tempfile
import time
from collections import Counter, deque
from pathlib import Path

import cv2

import data_loader
import landmark_extractor
import finger_classifier
import smoothing
import visualizer
from config import OUTPUT_DIR, ROOT

MODE = "VIDEO"      # 'IMAGE' / 'VIDEO' / 'WEBCAM'
USE_WORLD = True        # 실제 3D 좌표(hand_world_landmarks)로 접힘을 추가 판정. 옆모습에서 펴진 손가락이 접힘으로 나오면 False 로 비교해 본다
SHOW_JOINT_IDS = True   # 관절마다 작은 번호(0~20) 표시. 가려진 관절 번호는 자동으로 숨김. False 면 끔
USE_SMOOTHING = True  # 영상/웹캠 관절 좌표를 One Euro Filter 로 평활화 (뼈대 떨림 제거). False 면 끔
USE_HIDDEN_THUMB = True   # True 면 손끝(엄지·검지~새끼)이 다른 관절 뒤에 가려져 있을 때(화면의 번호가 숨겨지는 경우) 그 손가락을 접힘으로 본다
USE_DEPTH = False   # True 면 판정에 깊이(z)도 쓴다. 웹캠 z 는 불안정해서 기본은 끔
SMOOTH_FRAMES = 5   # 영상/웹캠에서 숫자 깜빡임을 줄이기 위해 최근 N프레임의 최빈값 사용


def _to_pixels(hand, img):
    """MediaPipe 좌표(0~1)는 가로·세로를 각각 이미지 크기로 나눈 값이라, 가로로 긴 웹캠(640x480)에서는
    가로 거리가 실제보다 짧게 계산된다. 판정 전에 픽셀 단위로 바꿔서 거리·각도가 왜곡되지 않게 한다.

    USE_DEPTH 가 False 이면 깊이(z)는 버리고 화면 위의 x, y 만으로 판정한다.
    웹캠의 z 값은 손 크기만큼 크고 프레임마다 크게 흔들려서, 엄지 올린 주먹처럼 '접힌 손가락'도 거리·각도가 뒤틀려
    펴진 것으로 나왔다. (사진 7장은 z 를 쓰나 안 쓰나 결과가 같다.)"""
    if img is None:
        return hand
    h, w = img.shape[:2]
    return [(p[0] * w, p[1] * h, p[2] * w if USE_DEPTH else 0.0) for p in hand]


def _depths(hand, img):
    """관절별 깊이(z)를 화면 x 와 같은 단위(픽셀)로 돌려준다. 손가락이 다른 관절 뒤에 가려졌는지 보는 데만 쓴다.
    (USE_DEPTH 와 상관없다. USE_HIDDEN_THUMB 가 False 면 None 을 돌려줘서 가림 판정을 끈다.)"""
    if not USE_HIDDEN_THUMB:
        return None
    w = img.shape[1] if img is not None else 1.0
    return [p[2] * w for p in hand]


def analyze(hands, img=None, stabilizer=None, worlds=None):
    """손 좌표 목록(또는 None)을 판정하고, img 가 있으면 뼈대와 숫자를 그린다.

    stabilizer: 영상/웹캠에서만 넘긴다. 손가락별 상태를 몇 프레임 모아서 떨림으로 뒤집히는 걸 막는다.
    worlds: 실제 3D 좌표 목록(선택). 주면 옆모습에서 화면상 곧게 보이는 접힌 손가락을 걸러낸다.
    반환: (전체 손가락 수, 손별 개수 리스트)
    """
    if hands is None:          # 손 미검출 처리
        if stabilizer is not None:
            stabilizer.reset()
        return 0, []
    # 손 번호는 손목의 가로 위치 순서로 매겨서, 프레임이 바뀌어도 같은 손이 같은 번호를 갖게 한다
    order = sorted(range(len(hands)), key=lambda i: hands[i][0][0])
    per_hand = []
    for i, hand in enumerate(hands):
        world = worlds[i] if worlds and USE_WORLD else None
        n, state = finger_classifier.count_fingers(_to_pixels(hand, img), world, _depths(hand, img))
        if stabilizer is not None:
            state = stabilizer.update(order.index(i), state)
            n = sum(state.values())
        per_hand.append(n)
        if img is not None:
            pts = visualizer.draw_skeleton(img, hand, state, SHOW_JOINT_IDS)   # 펴짐 파랑 / 접힘 빨강 + 관절 번호
            visualizer.draw_hand_count(img, pts, n)
    return sum(per_hand), per_hand


def finger_diagnosis(hands, worlds, img):
    """첫 번째 손의 손가락별 (이름, 펴짐 여부, 이유, 측정값). 손이 없으면 빈 목록. 'd' 화면과 's' 저장에 쓴다."""
    if not hands:
        return []
    hand = _to_pixels(hands[0], img)
    world = worlds[0] if worlds and USE_WORLD else None
    depth = _depths(hands[0], img)
    others = []
    for name, (mcp, pip, dip, tip) in finger_classifier.FINGERS.items():
        is_open, reason, metrics = finger_classifier.diagnose_finger(hand, mcp, tip, pip, dip, world, depth)
        others.append((name, is_open, reason, metrics))
    # 엄지는 주먹(다른 손가락 3개 이상 접힘)인지에 따라 꺾임 기준이 달라서 count_fingers 와 같은 순서로 마지막에 판정한다
    fist = sum(1 for _n, o, _r, _m in others if not o) >= 3
    t_open, t_reason, t_metrics = finger_classifier.diagnose_thumb(hand, world, depth, fist)
    return [("thumb", t_open, t_reason, t_metrics)] + others


def draw_debug(img, diag):
    """'d' 키: 화면 왼쪽 아래에 손가락별 판정 이유와 측정값을 쓴다.
    why = 접힘 이유. 손가락: hidden 다른 손가락 뒤에 가려짐 / touch 엄지에 닿음 / 3D-ext, 3D-pip 3D 기준 / wrist 손목쪽으로 말림 / 2D-pip 화면 각도 (2D/3D = PIP 각도, e = 3D 거리비)
    엄지: 3D-spread 3D 로 손에 붙음 / spread 검지뿌리에 붙음 / center 손바닥 위 / near, bent 꺾임 (sp, ct = 2D 거리비, IP = 엄지 각도, s3 = 3D 거리비)"""
    y = img.shape[0] - 45
    for name, is_open, reason, m in reversed(diag):
        if name == "thumb":
            text = "%-6s %-6s why=%-9s sp%.2f ct%.2f IP%3.0f s3 %.2f" % (
                name, "open" if is_open else "closed", reason, m["spread2d"], m["center2d"], m["ip2d"], m.get("spread3d", 0))
        else:
            text = "%-6s %-6s why=%-9s 2D%3.0f 3D%3.0f e%.2f" % (
                name, "open" if is_open else "closed", reason, m["pip2d"], m.get("pip3d", 0), m.get("ext3d", 0))
        visualizer.put_text(img, text, (10, y), 0.45, visualizer.OPEN_COLOR if is_open else (80, 80, 255), 1)
        y -= 20


# ---------------------------------------------------------------- IMAGE
def run_image(input_path):
    if input_path and not Path(input_path).exists():
        raise SystemExit(f"입력 파일이 없습니다: {input_path} (경로와 파일 이름을 확인하세요)")
    images = [Path(input_path)] if input_path else data_loader.list_images()
    if not images:
        raise SystemExit("data/inputs/images 폴더에 손 사진을 먼저 넣어 주세요. (jpg, jpeg, png, bmp)")

    rows = []
    started = time.perf_counter()
    with landmark_extractor.create_landmarker(video=False) as lm:
        for path in images:
            try:
                img = data_loader.read_image(path)
            except FileNotFoundError:
                # 사진이 아니거나 깨진 파일: 멈추지 않고 알려 준 뒤 다음 사진으로 넘어간다
                print(f"{path.name}: 사진으로 읽을 수 없어 건너뜁니다. (jpg, png 등 이미지 파일인지 확인하세요)")
                continue
            hands, worlds = landmark_extractor.extract_both_from_image(lm, img)
            if not hands:
                print(f"{path.name}: 손이 검출되지 않았습니다. (손을 밝고 크게, 화면에 꽉 차지 않게 찍은 사진으로 시험하세요)")
            total, per_hand = analyze(hands, img, worlds=worlds)
            visualizer.draw_total(img, total, len(per_hand))
            visualizer.draw_legend(img)
            visualizer.save_image(OUTPUT_DIR / f"{path.stem}_result.jpg", img)
            rows.append([path.name, len(per_hand), total, "/".join(map(str, per_hand))])
            print(f"{path.name}: 손 {len(per_hand)}개, 손가락 {total}개 {per_hand}")

    csv_path = OUTPUT_DIR / "image_results.csv"
    visualizer.save_csv(csv_path, visualizer.IMAGE_CSV_COLUMNS, rows)
    pred_path = OUTPUT_DIR / "finger_results.csv"
    visualizer.save_csv(pred_path, visualizer.FINGER_CSV_COLUMNS, rows)
    print(f"CSV 저장: {pred_path} ({time.perf_counter() - started:.1f}초)")


# ------------------------------------------------------- VIDEO / WEBCAM
def run_stream(mode, input_path, camera, save, show):
    tmp_dirs = []
    started = time.perf_counter()
    is_video = mode == "VIDEO"

    if is_video:
        if input_path:
            video_path = Path(input_path)
            if not video_path.is_absolute():
                video_path = ROOT / video_path
        else:
            vids = data_loader.list_videos()
            if not vids:
                raise SystemExit("data/inputs/videos 폴더에 영상을 넣거나 --input 으로 지정하세요.")
            video_path = vids[0]
        if not video_path.exists():
            raise SystemExit(f"비디오 파일이 없습니다: {video_path}")
        cap, tmp = data_loader.open_video(video_path)
        if tmp:
            tmp_dirs.append(tmp)
        if not cap.isOpened():
            raise SystemExit(f"비디오를 열 수 없습니다: {video_path}")
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    else:
        cap = data_loader.open_webcam(camera)
        if cap is None:
            raise SystemExit(
                f"카메라 {camera}번에서 화면을 읽지 못했습니다. 줌·카카오톡·브라우저 등 "
                "카메라를 쓰는 프로그램을 끄고, 이전 실행 창도 종료한 뒤 다시 실행하세요. "
                "(영상 파일은 --mode VIDEO)")
        src_fps = 30.0

    show_debug = False
    history = deque(maxlen=SMOOTH_FRAMES)
    stabilizer = finger_classifier.FingerStabilizer()   # 손가락별 안정화 (웹캠 떨림 대응)
    smoother = smoothing.MultiHandSmoother()            # 관절 좌표 평활화 (One Euro Filter)
    rows, writer, writer_tmp = [], None, None
    start = prev = time.time()
    last_ts, frame_index = -1, 0

    with landmark_extractor.create_landmarker(video=True) as lm:
        while True:
            ok, frame = cap.read()
            if not ok:
                if is_video:
                    print("입력이 끝났거나 프레임을 읽지 못했습니다.")
                else:
                    print("웹캠 프레임을 읽지 못했습니다. 다른 프로그램이 카메라를 쓰고 있으면 끄고 다시 실행하세요.")
                break
            this_index = frame_index
            frame_index += 1

            if is_video:
                if this_index % data_loader.FRAME_INTERVAL != 0:  # 프레임 추출 간격
                    continue
                frame = data_loader.resize_image(frame)
                ts = int(this_index * 1000 / src_fps)             # 영상 안의 시간
            else:
                frame = cv2.flip(frame, 1)                        # 웹캠은 거울 모드
                ts = int((time.time() - start) * 1000)            # 실제 시간
            ts = max(ts, last_ts + 1)                             # timestamp 는 항상 증가
            last_ts = ts

            raw_frame = frame.copy() if show and not is_video else None   # 's' 저장용 원본(뼈대 없음)
            hands, worlds = landmark_extractor.extract_both_from_frame(lm, frame, ts)
            if USE_SMOOTHING:                          # One Euro Filter: 뼈대 떨림 제거
                hands, worlds = smoother.update(hands, worlds, ts / 1000.0)
            total, per_hand = analyze(hands, frame, stabilizer, worlds)

            history.append(total)
            smooth_total = Counter(history).most_common(1)[0][0]
            now = time.time()
            fps = 1.0 / max(now - prev, 1e-6)
            prev = now
            visualizer.draw_total(frame, smooth_total, len(per_hand), fps)
            visualizer.draw_legend(frame)
            visualizer.put_text(frame, "q: quit  s: save  d: debug", (10, frame.shape[0] - 15),
                                0.6, (200, 200, 200))
            diag = finger_diagnosis(hands, worlds, frame)
            if show_debug:
                draw_debug(frame, diag)

            if is_video:
                rows.append([this_index, round(ts / 1000, 3), len(per_hand), smooth_total])
                if save:
                    if writer is None:
                        # 한글/네트워크 경로 문제를 피하려고 임시 폴더에 먼저 쓴다
                        writer_tmp = Path(tempfile.mkdtemp(prefix="finger_out_"))
                        h, w = frame.shape[:2]
                        writer = cv2.VideoWriter(
                            str(writer_tmp / "result.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                            src_fps / data_loader.FRAME_INTERVAL, (w, h))
                        if not writer.isOpened():
                            raise SystemExit("결과 영상을 만들 수 없습니다.")
                    writer.write(frame)

            if show:
                cv2.imshow("Finger Counter", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("d"):
                    show_debug = not show_debug      # 손가락별 판정 이유/측정값 표시 켜기/끄기
                if key == ord("s"):
                    name = time.strftime("capture_%Y%m%d_%H%M%S.jpg")
                    visualizer.save_image(OUTPUT_DIR / name, frame)
                    print(f"저장: output/{name}")
                    if raw_frame is not None:        # 뼈대 없는 원본도 같이 저장 -> 오프라인 재판정/튜닝용
                        raw_name = name.replace(".jpg", "_raw.jpg")
                        visualizer.save_image(OUTPUT_DIR / raw_name, raw_frame)
                        print(f"저장: output/{raw_name}  (원본, 뼈대 없음)")
                    if hands:                        # 관절 좌표(x, y, z + 실제 3D)도 저장 -> 어느 값 때문에 틀렸는지 분석용
                        lm_name = name.replace(".jpg", "_landmarks.csv")
                        lm_rows = []
                        for h_i, hand in enumerate(hands):
                            for p_i, p in enumerate(hand):
                                wp = worlds[h_i][p_i] if worlds else (0, 0, 0)
                                lm_rows.append([h_i, p_i, *(round(v, 5) for v in p), *(round(v, 5) for v in wp)])
                        visualizer.save_csv(OUTPUT_DIR / lm_name,
                                            ["hand", "index", "x", "y", "z", "wx", "wy", "wz"], lm_rows)
                        print(f"저장: output/{lm_name}  (관절 좌표)")
                        if diag:                     # 손가락별 판정 이유/측정값
                            mt_name = name.replace(".jpg", "_metrics.csv")
                            keys = ("pip2d", "pip3d", "ext3d", "wrist_ratio", "spread2d", "center2d", "ip2d", "spread3d")
                            mt_rows = [[n, o, r, *(round(m.get(k, float("nan")), 2) for k in keys)] for n, o, r, m in diag]
                            visualizer.save_csv(OUTPUT_DIR / mt_name, ["finger", "open", "reason", *keys], mt_rows)
                            print(f"저장: output/{mt_name}  (손가락별 판정 이유)")

    cap.release()
    if show:
        cv2.destroyAllWindows()

    if is_video and save:
        if writer is not None:
            writer.release()
            out_video = OUTPUT_DIR / f"{video_path.stem}_result.mp4"
            out_video.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(writer_tmp / "result.mp4"), str(out_video))
            shutil.rmtree(writer_tmp, ignore_errors=True)
            print(f"결과 영상 저장: {out_video}")
        out_csv = OUTPUT_DIR / f"{video_path.stem}_result.csv"
        visualizer.save_csv(out_csv, visualizer.VIDEO_CSV_COLUMNS, rows)
        print(f"CSV 저장: {out_csv} ({time.perf_counter() - started:.1f}초)")
    for d in tmp_dirs:
        shutil.rmtree(d, ignore_errors=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["IMAGE", "VIDEO", "WEBCAM"], default=MODE.upper())
    p.add_argument("--input", default=None, help="IMAGE: 사진 1장 / VIDEO: 영상 파일 경로")
    p.add_argument("--camera", type=int, default=0, help="WEBCAM 카메라 번호")
    p.add_argument("--save", action="store_true", help="VIDEO: 결과 영상과 CSV 저장")
    p.add_argument("--no-show", action="store_true", help="VIDEO: 화면 창 없이 실행")
    args = p.parse_args()

    if args.mode == "IMAGE":
        run_image(args.input)
    else:
        run_stream(args.mode, args.input, args.camera, args.save, show=not args.no_show)


if __name__ == "__main__":
    main()
