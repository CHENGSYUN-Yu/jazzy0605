"""
pose_offset.py — 目標位姿硬性補償物件（獨立工廠）

職責：
    對目標位姿的 x/y/z 套用固定偏移量（硬性補償），
    用於修正系統性誤差（例如相機安裝偏移、TCP 校正偏差）。

外部使用方式：
    po = PoseOffset(dx=20.0)           # x +20mm
    po = PoseOffset(dx=20.0, dy=-5.0)  # x +20mm, y -5mm
    result = po.apply(target)          # 回傳補償後的位姿（深拷貝）
"""

import copy


class PoseOffset:
    """
    目標位姿硬性補償物件。

    對 {'x_mm', 'y_mm', 'z_mm', 'yaw_deg', ...} 格式的位姿 dict
    套用固定的 dx/dy/dz 偏移，其餘欄位原樣保留。
    輸出為深拷貝，不修改輸入。
    """

    def __init__(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0):
        """
        dx, dy, dz : 各軸補償量（mm），正值為正方向偏移。
        """
        self._dx = float(dx)
        self._dy = float(dy)
        self._dz = float(dz)

    # ── 設定 ──────────────────────────────────────────────────────────────────

    def set_offset(self, dx: float = None, dy: float = None, dz: float = None):
        """動態調整補償量。只傳入要改變的軸，其他軸維持原值。"""
        if dx is not None:
            self._dx = float(dx)
        if dy is not None:
            self._dy = float(dy)
        if dz is not None:
            self._dz = float(dz)

    @property
    def offset(self) -> dict:
        """回傳目前補償量 {'dx', 'dy', 'dz'}（mm）。"""
        return {'dx': self._dx, 'dy': self._dy, 'dz': self._dz}

    # ── 套用補償 ──────────────────────────────────────────────────────────────

    def apply(self, target: dict) -> 'dict | None':
        """
        對目標位姿套用硬性補償。

        參數：
            target : dict，需包含 'x_mm', 'y_mm', 'z_mm'；
                     其他欄位（yaw_deg, Rz, source 等）原樣保留。

        回傳：補償後的位姿 dict（深拷貝），輸入無效時回傳 None。
        """
        if target is None:
            return None

        result = copy.deepcopy(target)
        result['x_mm'] = float(target['x_mm']) + self._dx
        result['y_mm'] = float(target['y_mm']) + self._dy
        result['z_mm'] = float(target['z_mm']) + self._dz
        return result
