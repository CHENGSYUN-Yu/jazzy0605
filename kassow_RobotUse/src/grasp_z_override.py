"""
grasp_z_override.py — 夾取目標 Z 軸強制覆蓋物件（獨立工廠）

職責：
    在法蘭面目標位姿最終確定前，將 z_mm 替換成固定值。
    用於手腕相機深度不穩定時，以已知安全高度覆蓋計算結果。

外部使用方式：
    gz = GraspZOverride(z_mm=-394.5, enabled=True)
    target = gz.apply(target)   # z_mm 被替換成 -394.5
    gz.set_enabled(False)       # 停用，恢復原始計算結果
    gz.set_z(-390.0)            # 動態調整覆蓋值
"""

import copy


class GraspZOverride:
    """
    夾取目標 Z 軸強制覆蓋物件。

    啟用時將 target['z_mm'] 替換為固定值，其餘欄位不變。
    停用時原樣回傳，不影響上游計算結果。
    """

    def __init__(self, z_mm: float, enabled: bool = True):
        """
        z_mm    : 覆蓋後的法蘭目標 Z（mm，WORLD frame）
        enabled : True = 啟用覆蓋；False = 透通（不修改）
        """
        self._z_mm   = float(z_mm)
        self._enabled = bool(enabled)

    # ── 套用 ──────────────────────────────────────────────────────────────────

    def apply(self, target: dict) -> dict:
        """
        套用 Z 覆蓋。

        輸入 target：dict，需包含 'z_mm' 欄位（來自 TargetConsiderGripper）
        回傳：深拷貝 dict，z_mm 替換為固定值（停用時原樣回傳）
        target 為 None 時回傳 None。
        """
        if target is None:
            return None
        if not self._enabled:
            return target
        out = copy.copy(target)
        out['z_mm'] = self._z_mm
        return out

    # ── 設定 ──────────────────────────────────────────────────────────────────

    def set_z(self, z_mm: float) -> None:
        self._z_mm = float(z_mm)

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = bool(enabled)

    @property
    def z_mm(self) -> float:
        return self._z_mm

    @property
    def enabled(self) -> bool:
        return self._enabled
