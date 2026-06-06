"""
trajectory_plan.py — Clamped Cubic B-Spline 4D 軌跡規劃物件（獨立工廠）

職責：
    接收起點（當前手臂 TCP）與終點（TargetZCompute 輸出），
    建立 Clamped Cubic B-Spline 軌跡，同步插值 4D（X, Y, Z, Rz）。

特性：
    - Clamped cubic（k=3）：起終點速度為 0，產生對稱 S-curve 加減速
    - Yaw 最短路徑插值：避免 +179° → -179° 繞遠路
    - 持續時間自動推算：依距離/角度與速度上限決定
    - 純計算物件，無 ROS / 無執行緒，任何地方可 import

外部使用方式：
    tp = TrajectoryPlan()
    info = tp.plan(
        start  = {'x_mm': 100, 'y_mm': 200, 'z_mm': -300, 'yaw_deg': 0},
        target = {'x_mm': 150, 'y_mm': 220, 'z_mm': -150, 'yaw_deg': 35},
    )
    print(f'軌跡時間: {info["duration_s"]:.2f}s')

    # 以 500Hz 取樣（每 0.002s 呼叫一次）
    for t in np.arange(0, info["duration_s"], 0.002):
        wp = tp.sample(t)
        # wp['x_mm'], wp['y_mm'], wp['z_mm'], wp['yaw_deg']
"""

import math

import numpy as np
from scipy.interpolate import BSpline


# ── 預設速度限制 ──────────────────────────────────────────────────────────────
_NOMINAL_VEL_MM_S  = 15.0   # 巡航線速度（mm/s），決定軌跡持續時間
_NOMINAL_ROT_DEG_S = 8.0    # 巡航轉速（deg/s）
_MIN_DURATION_S    = 0.6    # 最短軌跡時間（秒）


def _normalize_yaw(deg: float) -> float:
    """將角度規範化到 (-180, 180]。"""
    while deg >  180.0: deg -= 360.0
    while deg <= -180.0: deg += 360.0
    return deg


def _make_clamped_cubic_bspline(start_4d: np.ndarray,
                                 end_4d:   np.ndarray,
                                 duration: float) -> BSpline:
    """
    建立 clamped cubic B-spline（k=3）。

    knots = [0,0,0,0, T,T,T,T]  → 起終點各重複 4 次
    ctrl  = [S, S, E, E]         → 起終點各重複 2 次 → 端點速度為 0
    """
    T     = float(duration)
    knots = np.array([0, 0, 0, 0, T, T, T, T], dtype=np.float64)
    ctrl  = np.stack([start_4d, start_4d, end_4d, end_4d], axis=0)
    return BSpline(knots, ctrl, 3, extrapolate=False)


# ══════════════════════════════════════════════════════════════════════════════

class TrajectoryPlan:
    """
    Clamped Cubic B-Spline 4D 軌跡規劃物件。

    plan() 建立軌跡，sample(t) 取任意時刻的目標位姿。
    內外隔離：sample() 回傳新 dict，不暴露內部 spline 物件。
    """

    def __init__(self,
                 nominal_vel_mm_s:  float = _NOMINAL_VEL_MM_S,
                 nominal_rot_deg_s: float = _NOMINAL_ROT_DEG_S,
                 min_duration_s:    float = _MIN_DURATION_S):
        self._nominal_vel   = float(nominal_vel_mm_s)
        self._nominal_rot   = float(nominal_rot_deg_s)
        self._min_dur       = float(min_duration_s)

        self._spline     = None          # BSpline 物件（內部，不對外暴露）
        self._duration   = 0.0
        self._start_4d   = None
        self._end_4d     = None

    # ── 軌跡規劃 ──────────────────────────────────────────────────────────────

    def plan(self, start: dict, target: dict) -> dict:
        """
        建立從 start 到 target 的 B-Spline 軌跡。

        參數：
            start  : {'x_mm', 'y_mm', 'z_mm', 'yaw_deg'} — 當前手臂 TCP
            target : {'x_mm', 'y_mm', 'z_mm', 'yaw_deg'} — 目標位姿
                     （來自 TargetZCompute 輸出）

        回傳 dict：
            'duration_s'  : float  — 軌跡總時間（秒）
            'dist_mm'     : float  — XYZ 直線距離（mm）
            'dyaw_deg'    : float  — Yaw 旋轉量（度，最短路徑）
            'start_4d'    : list   — [x, y, z, yaw] 起點
            'end_4d'      : list   — [x, y, z, yaw] 終點（含最短路徑 yaw）
        """
        sx = float(start['x_mm']); sy = float(start['y_mm'])
        sz = float(start['z_mm']); sw = float(start.get('yaw_deg', 0.0))

        tx = float(target['x_mm']); ty = float(target['y_mm'])
        tz = float(target['z_mm']); tw = float(target.get('yaw_deg', 0.0))

        # Yaw 最短路徑：把終點 yaw 調整為「從起點出發不繞遠路的等效角度」
        tw_adjusted = sw + _normalize_yaw(tw - sw)

        start_4d = np.array([sx, sy, sz, sw],          dtype=np.float64)
        end_4d   = np.array([tx, ty, tz, tw_adjusted], dtype=np.float64)

        dist = float(np.linalg.norm(end_4d[:3] - start_4d[:3]))
        dyaw = abs(tw_adjusted - sw)

        T = max(dist / self._nominal_vel,
                dyaw / self._nominal_rot,
                self._min_dur)

        self._spline   = _make_clamped_cubic_bspline(start_4d, end_4d, T)
        self._duration = T
        self._start_4d = start_4d
        self._end_4d   = end_4d

        return {
            'duration_s': T,
            'dist_mm':    round(dist, 1),
            'dyaw_deg':   round(dyaw, 2),
            'start_4d':   start_4d.tolist(),
            'end_4d':     end_4d.tolist(),
        }

    # ── 軌跡取樣 ──────────────────────────────────────────────────────────────

    def sample(self, t: float):
        """
        取得軌跡在時刻 t（秒）的目標位姿。

        t 自動 clamp 到 [0, duration]，保證不超出範圍。

        回傳 dict：
            'x_mm'    : float
            'y_mm'    : float
            'z_mm'    : float
            'yaw_deg' : float

        軌跡未規劃時回傳 None。
        """
        if self._spline is None:
            return None

        t_clamped = min(max(float(t), 0.0), self._duration)
        wp = self._spline(t_clamped)

        return {
            'x_mm':    float(wp[0]),
            'y_mm':    float(wp[1]),
            'z_mm':    float(wp[2]),
            'yaw_deg': float(wp[3]),
        }

    # ── 狀態查詢 ──────────────────────────────────────────────────────────────

    def reset(self):
        """清除當前軌跡規劃。"""
        self._spline   = None
        self._duration = 0.0
        self._start_4d = None
        self._end_4d   = None

    @property
    def is_ready(self) -> bool:
        """是否已有可用的軌跡規劃。"""
        return self._spline is not None

    @property
    def duration(self) -> float:
        """軌跡總時間（秒）；未規劃時為 0。"""
        return self._duration

    def is_complete(self, t: float) -> bool:
        """時刻 t 是否已到達軌跡終點（t >= duration）。"""
        return self._spline is not None and float(t) >= self._duration

    # ── 設定 ──────────────────────────────────────────────────────────────────

    def set_nominal_vel(self, vel_mm_s: float):
        """調整巡航線速度（影響下一次 plan()，不影響當前軌跡）。"""
        self._nominal_vel = float(vel_mm_s)

    def set_nominal_rot(self, rot_deg_s: float):
        """調整巡航轉速（影響下一次 plan()，不影響當前軌跡）。"""
        self._nominal_rot = float(rot_deg_s)

    def set_min_duration(self, s: float):
        """調整最短軌跡時間（影響下一次 plan()）。"""
        self._min_dur = float(s)
