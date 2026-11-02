"""웹캠/영상용 관절 좌표 흔들림 제거: One Euro Filter (1 유로 필터).

왜 필요한가:
    MediaPipe 가 프레임마다 따로 찍는 관절 좌표는 손을 가만히 둬도 조금씩 떨린다. 그대로 그리면 뼈대 선이 흔들거리고,
    기준선 근처의 손가락은 떨림만으로 펴짐/접힘이 뒤집힌다.

왜 평균이 아니라 One Euro Filter 인가:
    단순 이동평균은 흔들림을 줄이는 만큼 손을 빠르게 움직일 때 뼈대가 손을 늦게 따라온다(지연).
    One Euro Filter 는 '움직임 속도'에 따라 필터 세기를 바꾼다.
      - 손이 거의 멈춰 있으면 강하게 평활화 -> 떨림이 사라진다.
      - 손이 빠르게 움직이면 약하게 평활화 -> 지연이 거의 없다.
    (Casiez et al., "1 Euro Filter", CHI 2012. VR 컨트롤러/손 추적에서 널리 쓰는 방법)

바꿀 값 (손 크기 = 손목~중지 뿌리 거리를 1 로 놓은 단위):
    MIN_CUTOFF: 작을수록 멈춰 있을 때 더 안 흔들리지만 느려진다. (지금 0.2)
    BETA      : 클수록 빠르게 움직일 때 지연이 줄지만 떨림이 조금 남는다. (지금 0.5)
    2026-10-30: 0.5/8.0 은 관절이 몇 픽셀만 떨려도 속도가 큰 것으로 봐서 필터가 풀렸다.
    웹캠 20fps 에서 손을 가만히 두면 뼈대가 부들거렸다. 0.2/0.5 로 그 잔떨림을 더 걸러 낸다.
"""
import math

MIN_CUTOFF = 0.2     # Hz. 멈춰 있을 때의 기본 차단 주파수
BETA = 0.5           # 속도에 비례해 차단 주파수를 올리는 정도. 작을수록 잔떨림을 더 죽인다
D_CUTOFF = 1.0       # Hz. 속도 추정에 쓰는 차단 주파수


def _alpha(cutoff, dt):
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """숫자 하나(좌표 하나)를 평활화한다."""

    def __init__(self, min_cutoff=MIN_CUTOFF, beta=BETA, d_cutoff=D_CUTOFF):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0

    def __call__(self, x, dt):
        if self.x_prev is None:
            self.x_prev = x
            return x
        dt = max(dt, 1e-3)
        dx = (x - self.x_prev) / dt
        dx_hat = self.dx_prev + _alpha(self.d_cutoff, dt) * (dx - self.dx_prev)
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        x_hat = self.x_prev + _alpha(cutoff, dt) * (x - self.x_prev)
        self.x_prev, self.dx_prev = x_hat, dx_hat
        return x_hat


class HandSmoother:
    """손 하나의 21개 관절 x 3좌표를 One Euro Filter 로 평활화한다.

    좌표 단위가 제각각(픽셀, 미터)이라, 손 크기(손목~중지 뿌리 거리)로 나눈 값에 필터를 걸고 다시 곱한다.
    그러면 손이 카메라에서 멀든 가깝든 같은 설정으로 동작한다.
    """

    def __init__(self, min_cutoff=MIN_CUTOFF, beta=BETA):
        self.min_cutoff, self.beta = min_cutoff, beta
        self.reset()

    def reset(self):
        self.filters = None
        self.scale = None
        self.last_t = None

    def update(self, points, t):
        """points: [(x, y, z) x 21], t: 초. 평활화된 같은 모양의 목록을 돌려준다."""
        wrist, mid = points[0], points[9]
        size = math.dist(wrist, mid) or 1e-6
        if self.filters is None:
            self.filters = [[OneEuroFilter(self.min_cutoff, self.beta) for _ in range(3)] for _ in points]
            self.scale = size
            self.last_t = t - 1.0 / 30
        # 손 크기도 천천히만 바뀌게 해서 기준 단위가 흔들리지 않게 한다
        self.scale += 0.1 * (size - self.scale)
        dt = t - self.last_t
        self.last_t = t
        s = self.scale
        out = []
        for p, fs in zip(points, self.filters):
            out.append(tuple(f(v / s, dt) * s for f, v in zip(fs, p)))
        return out


class MultiHandSmoother:
    """여러 손을 다루는 평활화기. 손 번호는 손목의 가로 위치 순서로 매겨서, 프레임이 바뀌어도 같은 손이 같은 필터를 쓴다.

    손 개수가 바뀌거나 손이 사라지면 필터를 처음부터 다시 시작한다(옛 위치로 끌려가지 않게).
    """

    def __init__(self, min_cutoff=MIN_CUTOFF, beta=BETA):
        self.min_cutoff, self.beta = min_cutoff, beta
        self.screen, self.world, self.count = [], [], 0

    def reset(self):
        self.screen, self.world, self.count = [], [], 0

    def update(self, hands, worlds, t):
        """hands: 화면 좌표 목록 또는 None, worlds: 3D 좌표 목록 또는 None, t: 초.
        반환: (평활화된 hands, 평활화된 worlds). 입력이 None 이면 (None, None)."""
        if hands is None:
            self.reset()
            return None, None
        if len(hands) != self.count:
            self.screen = [HandSmoother(self.min_cutoff, self.beta) for _ in hands]
            self.world = [HandSmoother(self.min_cutoff, self.beta) for _ in hands]
            self.count = len(hands)
        order = sorted(range(len(hands)), key=lambda i: hands[i][0][0])
        out_h, out_w = list(hands), (list(worlds) if worlds else None)
        for slot, i in enumerate(order):
            out_h[i] = self.screen[slot].update(hands[i], t)
            if worlds:
                out_w[i] = self.world[slot].update(worlds[i], t)
        return out_h, out_w
