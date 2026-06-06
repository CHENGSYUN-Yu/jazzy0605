"""
check_arrive.py — 手臂到位確認物件（獨立工廠）

職責：
    在 ExecuteMotion 發送 (0,0,0) 停止命令後，
    持續觀察手臂位姿變化。若位姿在指定時間內保持不動，
    判定手臂已實際停止並到達目標位姿，可進入下一階段。

設計動機：
    ExecuteMotion.done=True 代表「誤差已在容差內，已送出停止命令」。
    CheckArrive 則是物理確認：「手臂確實靜止了」。
    兩者分工：
        ExecuteMotion → 送停止命令
        CheckArrive   → 確認手臂不再動

判定邏輯：
    每幀記錄位姿變化量：
        Δpos = |current_pos - prev_pos| (mm)
        Δrot = |current_rot[2] - prev_rot[2]| (Yaw deg)
    若 Δpos < pos_stable_mm 且 Δrot < rot_stable_deg，
    累積穩定時間；連續穩定超過 stable_duration_s → is_arrived=True

外部使用方式：
    ca = CheckArrive(stable_duration_s=0.3)
    ca.reset()   # 每次移動前重置

    # 500Hz 定時迴圈（ExecuteMotion done 之後開始呼叫）：
    arrived = ca.update(current_pos, current_rot, now=time.monotonic())
    if arrived:
        proceed_to_next_stage()
"""

import math


class CheckArrive:
    """
    手臂到位物理確認物件。

    reset() 開始觀察，update() 每幀餵入位姿；
    is_arrived=True 表示手臂已確實靜止。
    """

    def __init__(self,
                 pos_stable_mm:    float = 0.5,
                 rot_stable_deg:   float = 0.5,
                 stable_duration_s: float = 0.3):
        """
        pos_stable_mm    : 位置變化量低於此值視為靜止（mm）
        rot_stable_deg   : Yaw 變化量低於此值視為靜止（deg）
        stable_duration_s: 連續靜止多久才確認到位（秒）
        """
        self._pos_thresh   = float(pos_stable_mm)
        self._rot_thresh   = float(rot_stable_deg)
        self._stable_dur   = float(stable_duration_s)

        self._prev_pos     = None
        self._prev_rot     = None
        self._stable_since = None   # 第一次進入穩定狀態的時間戳
        self._arrived      = False

    # ── 生命週期 ──────────────────────────────────────────────────────────────

    def reset(self):
        """開始新一輪觀察前呼叫，清除所有狀態。"""
        self._prev_pos     = None
        self._prev_rot     = None
        self._stable_since = None
        self._arrived      = False

    # ── 主更新 ────────────────────────────────────────────────────────────────

    def update(self, current_pos: list, current_rot: list,
               now: float) -> bool:
        """
        餵入當前手臂位姿，判斷是否已到位。

        參數：
            current_pos : [x, y, z]（mm），來自 Kassow SystemState.pos
            current_rot : [roll, pitch, yaw]（deg），來自 SystemState.rot
            now         : 當前時間戳（time.monotonic()）

        回傳：
            True  — 手臂已靜止超過 stable_duration_s，確認到位
            False — 仍在移動或穩定時間不足
        """
        if self._arrived:
            return True

        # 首幀：記錄基準，尚無法比較
        if self._prev_pos is None:
            self._prev_pos = list(current_pos)
            self._prev_rot = list(current_rot)
            return False

        # 計算位姿變化量
        pos_cur = list(current_pos)[:3]
        pos_prv = list(self._prev_pos)[:3]
        dp = math.sqrt(sum((a - b) ** 2 for a, b in zip(pos_cur, pos_prv)))
        dr = abs(float(current_rot[2]) - float(self._prev_rot[2])) \
             if len(current_rot) > 2 and len(self._prev_rot) > 2 else 0.0

        # 更新上一幀記錄
        self._prev_pos = list(current_pos)
        self._prev_rot = list(current_rot)

        stable_now = (dp < self._pos_thresh and dr < self._rot_thresh)

        if stable_now:
            if self._stable_since is None:
                self._stable_since = now           # 開始計時
            elif now - self._stable_since >= self._stable_dur:
                self._arrived = True               # 穩定夠久，確認到位
                return True
        else:
            self._stable_since = None              # 有移動，重新計時

        return False

    # ── 狀態查詢 ──────────────────────────────────────────────────────────────

    @property
    def is_arrived(self) -> bool:
        """手臂是否已確認到位。"""
        return self._arrived

    @property
    def stable_elapsed_s(self) -> float:
        """已連續靜止的秒數（尚未到位時）；未進入穩定狀態則為 0。"""
        if self._stable_since is None:
            return 0.0
        import time as _time
        return _time.monotonic() - self._stable_since

    # ── 設定 ──────────────────────────────────────────────────────────────────

    def set_thresholds(self, pos_mm: float, rot_deg: float):
        """調整靜止判斷門檻。"""
        self._pos_thresh = float(pos_mm)
        self._rot_thresh = float(rot_deg)

    def set_stable_duration(self, s: float):
        """調整需連續靜止的時間長度（秒）。"""
        self._stable_dur = float(s)
