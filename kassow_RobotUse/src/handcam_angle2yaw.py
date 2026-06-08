"""
handcam_angle2yaw.py — 手腕相機 2D 傾角 → 夾爪目標 yaw（獨立工廠物件）

職責：
    將 YOLO 偵測到的器械 2D 傾角（angle_deg）透過固定 offset 補償，
    直接轉換為夾爪（法蘭面）需要到達的目標偏航角（yaw_deg）。

    不使用 T_matrix，適用於手腕相機已知幾何關係的簡化補償：
        公式：yaw = -angle_deg + offset（注意是負號，fitEllipse 座標與 Rz 方向相反）
        angle_deg=  0° → raw = -135°，選優解 -135° 或 45°
        angle_deg= 90° → raw =  135°，選優解 -45°  或 135°

    支援 180° 對稱選優解（夾爪雙向可夾），選與當前 TCP yaw 最近的解。

外部使用方式：
    h2y = HandcamAngle2Yaw(offset_deg=-135.0)
    yaw = h2y.convert(angle_deg=5.8,   current_yaw=-135.0)  # → -140.8°
    yaw = h2y.convert(angle_deg=134.5, current_yaw=-90.0)   # →  -89.5°
    dets = h2y.convert_all(dets, current_yaw=tcp_yaw)       # 批次，新增 yaw_deg 欄位
    h2y.set_offset(-135.0)
"""

import copy


def _normalize(deg: float) -> float:
    """正規化到 (-180, 180]。"""
    while deg >  180.0: deg -= 360.0
    while deg <= -180.0: deg += 360.0
    return deg


class HandcamAngle2Yaw:
    """
    手腕相機 2D 傾角 → 夾爪目標 yaw 轉換物件。

    輸出皆深拷貝，不修改輸入 dict。
    """

    def __init__(self, offset_deg: float = -45.0):
        """
        offset_deg : 2D 傾角加上此 offset 得到夾爪 yaw。預設 -45°。
        """
        self._offset = float(offset_deg)

    # ── 設定 ──────────────────────────────────────────────────────────────────

    def set_offset(self, offset_deg: float) -> None:
        """動態調整 offset（不影響外部介面）。"""
        self._offset = float(offset_deg)

    @property
    def offset_deg(self) -> float:
        return self._offset

    # ── 轉換 ──────────────────────────────────────────────────────────────────

    def convert(self, angle_deg: float,
                current_yaw: float = 0.0) -> float:
        """
        單一傾角轉換。

        參數：
            angle_deg   : YOLO 偵測的器械 2D 傾角（度）
            current_yaw : 手臂當前 TCP yaw（度），用於 180° 對稱選優解

        回傳：夾爪目標 yaw_deg（度）
        """
        raw = _normalize(-float(angle_deg) + self._offset)
        # 180° 對稱解：選與當前 TCP yaw 差距最小的
        alt = raw + 180.0 if raw <= 0.0 else raw - 180.0
        return raw if abs(raw - current_yaw) <= abs(alt - current_yaw) else alt

    def convert_all(self, instruments: list,
                    current_yaw: float = 0.0) -> list:
        """
        批次轉換，對每個 det dict 新增或覆蓋 'yaw_deg' 欄位。

        參數：
            instruments : list of dict，每個元素需含 'angle_deg'
            current_yaw : 手臂當前 TCP yaw（度）

        回傳：深拷貝後的 list，每個元素新增 'yaw_deg'。
        angle_deg 為 None 或不存在的器械，yaw_deg 設為 None。
        """
        result = []
        for inst in instruments:
            out = copy.deepcopy(inst)
            angle = inst.get('angle_deg')
            if angle is None:
                out['yaw_deg'] = None
            else:
                out['yaw_deg'] = self.convert(float(angle), current_yaw)
            result.append(out)
        return result
