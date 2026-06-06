"""
roi_filter.py — ROI 過濾獨立封裝物件

職責：
    管理一個矩形 ROI（Region of Interest）範圍，
    提供點是否在範圍內的判斷。
    可被任何需要 ROI 過濾功能的物件使用。

外部使用方式：
    roi = RoiFilter()
    roi.set(275, 153, 442, 342)
    roi.contains(300, 200)    # True
    roi.contains(100, 100)    # False
"""


class RoiFilter:

    def __init__(self):
        self._roi = None    # (x1, y1, x2, y2) 或 None

    # ── 設定 ──────────────────────────────────────────────────────────────────

    def set(self, x1: int, y1: int, x2: int, y2: int):
        """設定 ROI 範圍（像素座標，左上 → 右下）。"""
        self._roi = (int(x1), int(y1), int(x2), int(y2))

    def clear(self):
        """清除 ROI，所有點都視為在範圍內。"""
        self._roi = None

    # ── 查詢 ──────────────────────────────────────────────────────────────────

    def get(self):
        """取得目前 ROI；未設定則返回 None。"""
        return self._roi

    def is_active(self) -> bool:
        """是否已設定 ROI。"""
        return self._roi is not None

    def contains(self, cx: float, cy: float) -> bool:
        """
        判斷點 (cx, cy) 是否在 ROI 範圍內。
        未設定 ROI 時永遠返回 True（不過濾）。
        """
        if self._roi is None:
            return True
        x1, y1, x2, y2 = self._roi
        return x1 <= cx <= x2 and y1 <= cy <= y2
