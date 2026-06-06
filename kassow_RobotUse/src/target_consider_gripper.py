"""
target_consider_gripper.py — 考慮夾爪長度的法蘭目標位姿計算物件（獨立工廠）

職責：
    接收 Flange2Base 的輸出（器械在 base frame 的 pos + yaw），
    考慮夾爪裝上後的幾何，計算法蘭面真正需要到達的目標位姿。

考慮項目：
    1. Z 軸補償：
       夾爪有一定長度（gripper_length_mm）。
       法蘭面需在器械 Z 位置的基礎上往上移 gripper_length_mm，
       讓夾爪中心剛好對準器械。

    2. Rz 對齊：
       法蘭面的偏航角對齊器械長軸方向（yaw_deg），
       讓夾爪開合方向與器械方向匹配。

外部使用方式：
    tcg = TargetConsiderGripper(gripper_length_mm=80.0)
    target = tcg.compute(instrument)
    # target: {'x_mm', 'y_mm', 'z_mm', 'yaw_deg', 'source'}
"""

import copy


class TargetConsiderGripper:
    """
    考慮夾爪長度的法蘭目標位姿計算物件。
    """

    def __init__(self, gripper_length_mm: float = 120.0):
        self._length = float(gripper_length_mm)

    # ── 設定 ──────────────────────────────────────────────────────────────────

    def set_gripper_length(self, mm: float):
        """調整夾爪長度。"""
        self._length = float(mm)

    @property
    def gripper_length_mm(self) -> float:
        return self._length

    # ── 單一器械計算 ──────────────────────────────────────────────────────────

    def compute(self, instrument: dict):
        """
        計算裝上夾爪後法蘭面真正需要到達的目標位姿。

        輸入 instrument：dict，需包含：
            'pos_base_mm'  : [x, y, z]（mm），來自 Flange2Base
            'yaw_deg' : float（度），來自 Flange2Base

        回傳 dict（深拷貝）：
            'x_mm'    : float  — 法蘭目標 X（= 器械 X）
            'y_mm'    : float  — 法蘭目標 Y（= 器械 Y）
            'z_mm'    : float  — 法蘭目標 Z（= 器械 Z + gripper_length）
            'yaw_deg' : float  — 法蘭目標偏航角（= 器械 yaw_deg）
            'source'  : dict   — 原始器械資料（深拷貝）

        pos_base_mm 或 yaw_deg 為 None 時回傳 None。
        """
        pos = instrument.get('pos_base_mm')
        yaw = instrument.get('yaw_deg')

        if pos is None or yaw is None:
            return None

        return {
            'x_mm':    float(pos[0]),
            'y_mm':    float(pos[1]),
            'z_mm':    float(pos[2]) + self._length,
            'yaw_deg': float(yaw),
            'source':  copy.deepcopy(instrument),
        }

    # ── 批次多器械計算 ────────────────────────────────────────────────────────

    def compute_all(self, instruments: list) -> list:
        """
        批次計算多個器械的法蘭目標位姿。
        pos_base_mm 或 yaw_deg 為 None 的器械自動跳過。
        """
        results = []
        for inst in instruments:
            target = self.compute(inst)
            if target is not None:
                results.append(target)
        return results
