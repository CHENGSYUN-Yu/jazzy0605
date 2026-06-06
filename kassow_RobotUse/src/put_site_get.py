"""
put_site_get.py — 器械放置位姿查找物件（獨立工廠）

職責：
    根據主程式的計數器數值（目前已夾取並放置過的器械數量），
    決定當前夾爪夾著的器械應該擺放到哪個位置（base frame）。

放置順序（base frame，mm）：
    第 1 個器械 → (513, -150, -355)
    第 2 個器械 → (513, -110, -355)
    第 3 個器械 → (513,  -70, -355)

    X 固定 513mm，Z 固定 -355mm，Y 軸每個間距 40mm。

輸入計數器來源：
    MemoryInstrumentPoint.total_recorded（已累計夾取次數）

外部使用方式：
    psg = PutSiteGet()
    target = psg.get(count=1)   # 第 1 個 → {'x_mm':513, 'y_mm':-150, 'z_mm':-355, 'yaw_deg':0}
    target = psg.get(count=2)   # 第 2 個 → y=-110
    target = psg.get(count=3)   # 第 3 個 → y=-70
    target = psg.get(count=4)   # 超出範圍 → None
"""

import copy


# 三個放置點（base frame，mm），index 0 = 第 1 個器械
_DEFAULT_SITES = [
    (513.0, -150.0, -355.0),   # 第 1 個
    (513.0, -110.0, -355.0),   # 第 2 個
    (513.0,  -70.0, -355.0),   # 第 3 個
]


class PutSiteGet:
    """
    器械放置位姿查找物件。

    根據計數器值（1-based）返回對應的放置目標位姿。
    放置點可在初始化後透過 set_site() 動態調整。
    """

    def __init__(self, yaw_deg: float = 0.0):
        """
        yaw_deg : 放置時的法蘭偏航角（度），預設 0.0。
        """
        self._sites   = copy.deepcopy(list(_DEFAULT_SITES))
        self._yaw_deg = float(yaw_deg)

    # ── 查找放置位姿 ──────────────────────────────────────────────────────────

    def get(self, count: int):
        """
        根據計數器值取得放置目標位姿。

        參數：
            count : int — 當前是第幾個器械（1-based），
                          對應 MemoryInstrumentPoint.total_recorded

        回傳 dict：
            'x_mm'    : float
            'y_mm'    : float
            'z_mm'    : float
            'yaw_deg' : float
            None — count 超出範圍（< 1 或 > 放置點數量）時
        """
        idx = int(count) - 1   # 轉 0-based
        if not (0 <= idx < len(self._sites)):
            return None

        x, y, z = self._sites[idx]
        return {
            'x_mm':    x,
            'y_mm':    y,
            'z_mm':    z,
            'yaw_deg': self._yaw_deg,
        }

    # ── 放置點設定 ────────────────────────────────────────────────────────────

    def set_site(self, count: int, x_mm: float, y_mm: float, z_mm: float):
        """
        動態調整指定器械的放置點座標。
        count: 1-based（1~3）
        """
        idx = int(count) - 1
        if 0 <= idx < len(self._sites):
            self._sites[idx] = [float(x_mm), float(y_mm), float(z_mm)]

    def set_yaw(self, yaw_deg: float):
        """調整所有放置點的偏航角。"""
        self._yaw_deg = float(yaw_deg)

    def get_all_sites(self) -> list:
        """取得所有放置點座標（深拷貝）。"""
        return copy.deepcopy(self._sites)

    @property
    def site_count(self) -> int:
        """放置點總數。"""
        return len(self._sites)
