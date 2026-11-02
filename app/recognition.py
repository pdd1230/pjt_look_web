"""main-2.py의 처리 순서를 웹 입력에 맞게 연결. 판정 기준은 변경하지 않는다."""
import hashlib
import math
import time
from collections import Counter, deque
from .vision import landmark_extractor as le, finger_classifier as fc, visualizer as vis, smoothing
from .settings import MODEL_PATH, ALGORITHM

class Recognizer:
    def __init__(self, video=False):
        self.video = video
        self.landmarker = le.create_landmarker(video=video)
        self.smoother = smoothing.MultiHandSmoother()
        self.stabilizer = fc.FingerStabilizer()
        self.history = deque(maxlen=5)
        self.previous_count = 0
        self.last_ts = -1
        self.model_hash = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()

    def close(self):
        self.landmarker.close()

    def process(self, frame, timestamp_ms=0):
        start = time.perf_counter()
        ts = max(int(timestamp_ms), self.last_ts+1)
        self.last_ts = ts
        if self.video:
            hands, worlds = le.extract_both_from_frame(self.landmarker, frame, ts)
            hands, worlds = self.smoother.update(hands, worlds, ts/1000)
        else:
            hands, worlds = le.extract_both_from_image(self.landmarker, frame)
        output = frame.copy()
        h, w = frame.shape[:2]
        # Left-to-right screen order is used for both predictions and manual truth.
        order = sorted(range(len(hands or [])), key=lambda i: hands[i][0][0])
        if len(order) != self.previous_count:
            self.stabilizer.reset()
            self.history.clear()
        self.previous_count = len(order)
        details = []
        for slot, i in enumerate(order):
            hand = hands[i]
            world = worlds[i] if worlds else None
            pixels = [(p[0]*w, p[1]*h, 0.0) for p in hand]  # USE_DEPTH=False
            depths = [p[2]*w for p in hand]  # USE_HIDDEN_THUMB=True, USE_WORLD=True
            n, raw = fc.count_fingers(pixels, world, depths)
            state = self.stabilizer.update(slot, raw) if self.video else raw
            pts = vis.draw_skeleton(output, hand, state, True)
            vis.draw_hand_count(output, pts, sum(state.values()))
            diagnosis = []
            for name, (mcp, pip, dip, tip) in fc.FINGERS.items():
                o, reason, metrics = fc.diagnose_finger(pixels, mcp, tip, pip, dip, world, depths)
                diagnosis.append({'finger': name, 'open': o, 'reason': reason, 'metrics': metrics})
            fist = sum(not d['open'] for d in diagnosis) >= 3
            o, reason, metrics = fc.diagnose_thumb(pixels, world, depths, fist)
            # count_fingers also applies the thumb-over-curled-fingers rule after diagnosis.
            if o and not raw['thumb']:
                reason = 'thumb_over_curled_fingers'
            diagnosis.insert(0, {'finger': 'thumb', 'open': raw['thumb'], 'reason': reason, 'metrics': metrics})
            details.append({'slot': slot, 'count': sum(state.values()), 'state': state, 'raw_state': raw,
                'raw_count': n, 'landmarks': hand, 'world_landmarks': world, 'diagnosis': diagnosis})
        per_hand = [d['count'] for d in details]
        total = sum(per_hand) if details else None
        if total is None:
            self.history.clear()
            self.stabilizer.reset()
            display_total = None
            vis.put_text(output, 'NO HAND', (10, 40), 1)
        else:
            self.history.append(total)
            display_total = Counter(self.history).most_common(1)[0][0] if self.video else total
            # Keep per-hand sum and the evaluated count consistent; display smoothing is separate.
            vis.draw_total(output, total, len(details))
        vis.draw_legend(output)
        data = {'status': 'DETECTED' if details else 'NO_HAND', 'hand_count': len(details),
            'total': total, 'per_hand': per_hand, 'hands': details, 'display_smoothed_total': display_total,
            'processing_ms': round((time.perf_counter()-start)*1000, 2),
            'model_sha256': self.model_hash, 'algorithm': ALGORITHM,
            'settings': {'use_world': True, 'use_depth': False, 'hidden_fingers': True,
                         'one_euro_filter': self.video, 'finger_stabilizer': self.video,
                         'joint_ids': True, 'screen_order': 'left_to_right'},
            'analysis_size': [w, h]}
        return clean(data), output

def clean(x):
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x
