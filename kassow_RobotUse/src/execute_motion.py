"""
execute_motion.py — JogLinear P-control 執行物件（獨立工廠）

職責：
    接收 TrajectoryPlan 和手臂當前位姿，以 P-control 計算每個時刻
    應發送給手臂的 JogLinear 速度命令（vel_xyz + rot_z）。

設計：
    - 純計算物件，無 ROS 依賴
    - 呼叫方（auto_grasp）負責 500Hz 定時呼叫 compute() 並發布 ROS topic
    - compute() 回傳速度命令 dict；done=True 時呼叫方應停止定時器並發布 (0,0,0)

P-control 邏輯：
    err = desired(t) - current
    vel = clamp(gain × err, -limit, +limit)
    當所有軸誤差連續 stable_frames 幀都在容差內 → done=True

外部使用方式：
    em = ExecuteMotion()
    em.start()   # 重置計時與穩定計數

    # 500Hz 定時迴圈（由 auto_grasp QTimer 驅動）：
    t = time.monotonic() - t0
    cmd = em.compute(t, current_pos, current_rot, trajectory_plan)
    ros_node.publish_jog(cmd['vel'], cmd['rot'])
    if cmd['done']:
        ros_node.publish_jog([0,0,0], [0,0,0])
        stop_timer()
"""

import math


# ── 預設控制參數 ──────────────────────────────────────────────────────────────
_VEL_LIMIT_MM_S  = 30.0    # 線速度上限 mm/s
_ROT_LIMIT_DEG_S = 15.0    # 轉速上限 deg/s
_GAIN_XYZ        = 5.0     # P-gain（mm/s per mm 誤差）
_GAIN_YAW        = 5.0     # P-gain（deg/s per deg 誤差）
_POS_THRESH_MM   = 1.0     # 位置容差 mm
_ROT_THRESH_DEG  = 1.0     # 角度容差 deg
_STABLE_FRAMES   = 10      # 連續幾幀在容差內才算到達


def _clamp(val: float, limit: float) -> float:
    return max(-limit, min(limit, val))


def _normalize_angle(deg: float) -> float:
    while deg >  180.0: deg -= 360.0
    while deg <= -180.0: deg += 360.0
    return deg


# ══════════════════════════════════════════════════════════════════════════════

class ExecuteMotion:
    """
    JogLinear P-control 執行物件。

    呼叫方維護 500Hz 定時器，每幀呼叫 compute()，
    根據回傳的 vel/rot 發布 JogLinear；done=True 時停止。
    """

    def __init__(self,
                 vel_limit_mm_s:  float = _VEL_LIMIT_MM_S,
                 rot_limit_deg_s: float = _ROT_LIMIT_DEG_S,
                 gain_xyz:        float = _GAIN_XYZ,
                 gain_yaw:        float = _GAIN_YAW,
                 pos_thresh_mm:   float = _POS_THRESH_MM,
                 rot_thresh_deg:  float = _ROT_THRESH_DEG,
                 stable_frames:   int   = _STABLE_FRAMES):
        self._vel_limit   = float(vel_limit_mm_s)
        self._rot_limit   = float(rot_limit_deg_s)
        self._gain_xyz    = float(gain_xyz)
        self._gain_yaw    = float(gain_yaw)
        self._pos_thresh  = float(pos_thresh_mm)
        self._rot_thresh  = float(rot_thresh_deg)
        self._stable_req  = int(stable_frames)

        self._stable_count = 0
        self._done         = False
        self._start_t      = None   # 第一次 compute() 時記錄 t，用於最小啟動時間判斷

    # ── 生命週期 ──────────────────────────────────────────────────────────────

    def start(self):
        """開始新一段移動前呼叫，重置穩定計數與完成旗標。"""
        self._stable_count = 0
        self._done         = False
        self._start_t      = None

    def reset(self):
        """清除所有狀態（等同 start）。"""
        self.start()

    # ── 主計算 ────────────────────────────────────────────────────────────────

    def compute(self,
                t:              float,
                current_pos:    list,
                current_rot:    list,
                trajectory) -> dict:
        """
        計算當前時刻應發送的 JogLinear 速度命令。

        參數：
            t            : 距軌跡開始的秒數（time.monotonic() - t0）
            current_pos  : [x, y, z]（mm），來自 Kassow SystemState.pos
            current_rot  : [roll, pitch, yaw]（deg），來自 SystemState.rot
            trajectory   : TrajectoryPlan 物件

        回傳 dict：
            'vel'  : [vx, vy, vz]  mm/s  — 發布到 JogLinear.vel
            'rot'  : [0, 0, rz]    deg/s — 發布到 JogLinear.rot
            'done' : bool           — True 表示已到達目標，呼叫方應停止發送
            'err_pos_mm'  : float   — 當前 XYZ 誤差距離（診斷用）
            'err_yaw_deg' : float   — 當前 Yaw 誤差（診斷用）

        trajectory 未就緒或 done=True 時，回傳零速命令。
        """
        # 已完成 → 持續回傳零速（防呼叫方忘記停止）
        if self._done:
            return self._zero_cmd(done=True)

        if not trajectory.is_ready:
            return self._zero_cmd(done=False)

        # ── 取得期望位姿 ──────────────────────────────────────────────────────
        wp = trajectory.sample(t)
        if wp is None:
            return self._zero_cmd(done=False)

        # ── 計算誤差 ──────────────────────────────────────────────────────────
        err_x = wp['x_mm']    - float(current_pos[0])
        err_y = wp['y_mm']    - float(current_pos[1])
        err_z = wp['z_mm']    - float(current_pos[2])
        # Yaw 誤差取最短路徑
        err_yaw = _normalize_angle(wp['yaw_deg'] - float(current_rot[2]))

        err_pos = math.sqrt(err_x**2 + err_y**2 + err_z**2)

        # ── P-control → 速度命令 ──────────────────────────────────────────────
        vx = _clamp(self._gain_xyz * err_x,   self._vel_limit)
        vy = _clamp(self._gain_xyz * err_y,   self._vel_limit)
        vz = _clamp(self._gain_xyz * err_z,   self._vel_limit)
        rz = _clamp(self._gain_yaw * err_yaw, self._rot_limit)

        # ── 到達判斷（所有軸誤差在容差內連續 stable_frames 幀）────────────────
        # 記錄第一次呼叫的 t，作為最小啟動時間基準
        if self._start_t is None:
            self._start_t = t
        min_active_s = trajectory.duration * 0.8   # 至少跑 80% 軌跡時間才能判定到達
        elapsed = t - self._start_t

        within = (err_pos < self._pos_thresh and
                  abs(err_yaw) < self._rot_thresh and
                  elapsed >= min_active_s)
        if within:
            self._stable_count += 1
        else:
            self._stable_count = 0

        if self._stable_count >= self._stable_req:
            self._done = True
            return self._zero_cmd(done=True,
                                  err_pos_mm=err_pos,
                                  err_yaw_deg=abs(err_yaw))

        return {
            'vel':         [vx, vy, vz],
            'rot':         [0.0, 0.0, rz],
            'done':        False,
            'err_pos_mm':  round(err_pos, 2),
            'err_yaw_deg': round(abs(err_yaw), 2),
        }

    # ── 狀態查詢 ──────────────────────────────────────────────────────────────

    @property
    def is_done(self) -> bool:
        """是否已到達目標（連續 stable_frames 幀在容差內）。"""
        return self._done

    @property
    def stable_count(self) -> int:
        """目前連續在容差內的幀數。"""
        return self._stable_count

    # ── 設定 ──────────────────────────────────────────────────────────────────

    def set_vel_limit(self, mm_s: float):
        self._vel_limit = float(mm_s)

    def set_gain(self, gain_xyz: float, gain_yaw: float):
        self._gain_xyz = float(gain_xyz)
        self._gain_yaw = float(gain_yaw)

    def set_thresholds(self, pos_mm: float, rot_deg: float):
        self._pos_thresh = float(pos_mm)
        self._rot_thresh = float(rot_deg)

    # ── 內部工具 ──────────────────────────────────────────────────────────────

    @staticmethod
    def _zero_cmd(done: bool, err_pos_mm: float = 0.0,
                  err_yaw_deg: float = 0.0) -> dict:
        return {
            'vel':         [0.0, 0.0, 0.0],
            'rot':         [0.0, 0.0, 0.0],
            'done':        done,
            'err_pos_mm':  err_pos_mm,
            'err_yaw_deg': err_yaw_deg,
        }
