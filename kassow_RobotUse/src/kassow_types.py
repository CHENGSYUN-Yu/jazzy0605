"""Kassow ros2_interface 共用常數 — 所有節點/Mixin 統一 import 這支。

避免 KR_NS / STATE_NAMES 散落各檔導致預設值不一致 (bug 歷史：曾經三檔寫 'kr/left'
一檔寫 'kr'，連線排查多花一天)。
"""
import os


# Namespace — 對齊 docker-compose KASSOW_NS。
# 預設 'kr' 是官方 orange-ros2 範例與單臂 CBun 設定。雙臂 teleop framework
# 用 'kr/left' / 'kr/right'，由環境變數覆寫。
KR_NS = os.environ.get('KASSOW_NS', 'kr').strip('/')


# /<KR_NS>/system/state 最大允許 staleness（秒）。
# Kassow CBun 預設 500Hz publish — 超過 2 秒沒收新訊息就視為連線中斷
# （motion_node 拒 confirmed_pose、calib 拒採樣、verify 顯示「TCP stale」）。
# task 分頁狀態列顯示用 3.0s，比這寬鬆是因為偶爾 DDS 丟一兩 frame 不需急著紅燈。
ARM_STATE_STALE_S = 2.0


# RobotState.msg val → 字串（與官方 kr_msgs/msg/RobotState.msg 的 SUPPORTED_STATES 對應）
STATE_NAMES = {
    1: 'INIT',
    2: 'STANDBY',
    3: 'MOVING',
    4: 'BACKDRIVE',
    5: 'SUSPENDED',
    6: 'ALARM',
}

# RobotMode.msg val → 字串
MODE_NAMES = {
    0: 'MANUAL',
    1: 'AUTONOMOUS',
}

# SafetyMode.msg val → 字串（含速度上限說明）
SAFETY_NAMES = {
    0: 'SAFE (0.25 m/s)',
    1: 'REDUCED (1.0 m/s)',
    2: 'NORMAL (unlimited)',
}


def state_name(val: int) -> str:
    """RobotState.val → 字串；未知值傳回 'UNKNOWN(<val>)'"""
    return STATE_NAMES.get(val, f'UNKNOWN({val})')


def mode_name(val: int) -> str:
    return MODE_NAMES.get(val, f'UNKNOWN({val})')


def safety_name(val: int) -> str:
    return SAFETY_NAMES.get(val, f'UNKNOWN({val})')
