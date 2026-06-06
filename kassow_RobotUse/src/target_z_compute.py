"""
target_z_compute.py — 目標位姿 Z 軸偏移計算物件（獨立工廠）

職責：
    接收器械在機器人 base frame 的位姿（pos_base_mm + yaw_deg/Rz），
    將 Z 值加上偏移量（預設 +300mm），產生手臂接近點位姿。

用途：
    讓手臂移動到器械正上方 300mm 處，頭部相機可以從正上方
    拍攝器械並進行後續精確位置補償。

    XY 方向對齊器械位置，Rz 保持器械長軸方向，Z 抬高至安全高度。

外部使用方式：
    tzc = TargetZCompute(z_offset_mm=300.0)

    # 單一器械
    target = tzc.compute(instrument)
    print(target['x_mm'], target['y_mm'], target['z_mm'])  # 器械上方 300mm
    print(target['yaw_deg'])                                # 保持器械方向

    # 批次
    targets = tzc.compute_all(instruments)
"""

import copy


class TargetZCompute:
    """
    目標位姿 Z 軸偏移計算物件。

    輸入：器械在 base frame 的位姿（pos_base_mm + yaw_deg/Rz）
    輸出：手臂接近點位姿（Z 加偏移，XY 及 Rz 不變）
    內外隔離：輸出皆深拷貝。
    """

    def __init__(self, z_offset_mm: float = 300.0):
        """
        z_offset_mm : 在器械 Z 座標基礎上往上抬的距離（mm），預設 300mm。
        """
        self._z_offset = float(z_offset_mm)

    # ── 設定 ──────────────────────────────────────────────────────────────────

    def set_z_offset(self, mm: float):
        """動態調整 Z 偏移量（不影響外部介面）。"""
        self._z_offset = float(mm)

    @property
    def z_offset_mm(self) -> float:
        return self._z_offset

    # ── 單一器械 ──────────────────────────────────────────────────────────────

    def compute(self, instrument: dict):
        """
        計算單一器械的手臂接近點位姿。

        輸入 instrument：dict，需包含：
            'pos_base_mm' : [x, y, z]（mm），來自 HeadCam2Base
            'yaw_deg'     : float，來自 Angle2Rz
            'Rz'          : list（3×3），來自 Angle2Rz

        回傳 dict（深拷貝）：
            'x_mm'    : float  — 目標 X（= 器械 X）
            'y_mm'    : float  — 目標 Y（= 器械 Y）
            'z_mm'    : float  — 目標 Z（= 器械 Z + z_offset_mm）
            'yaw_deg' : float  — 目標偏航角（= 器械長軸方向）
            'Rz'      : list   — 3×3 旋轉矩陣
            'source'  : dict   — 原始器械資料（深拷貝）

        pos_base_mm 為 None 時回傳 None。
        """
        pos = instrument.get('pos_base_mm')
        if pos is None:
            return None

        x_mm, y_mm, z_mm = float(pos[0]), float(pos[1]), float(pos[2])

        return {
            'x_mm':    x_mm,
            'y_mm':    y_mm,
            'z_mm':    z_mm + self._z_offset,
            'yaw_deg': instrument.get('yaw_deg'),
            'Rz':      copy.deepcopy(instrument.get('Rz')),
            'source':  copy.deepcopy(instrument),
        }

    # ── 批次多器械 ────────────────────────────────────────────────────────────

    def compute_all(self, instruments: list) -> list:
        """
        批次計算多個器械的手臂接近點位姿。

        輸入：HeadCam2Base + Angle2Rz 輸出的 list of dict
        回傳：list of dict，每個元素為 compute() 的輸出；
              pos_base_mm 為 None 的器械跳過（不加入結果）。
        """
        results = []
        for inst in instruments:
            target = self.compute(inst)
            if target is not None:
                results.append(target)
        return results
