"""
depth_reader.py — RealSense 深度圖像素深度讀取物件（獨立工廠）

職責：
    從 RealSense 深度圖（uint16，單位 mm）取得指定像素的深度值。
    內部使用 NxN patch 中位數過濾，對邊緣雜訊與無效像素（值為 0）有穩健性。
    內外隔離，外部只需傳入像素座標與深度圖即可取得深度。

外部使用方式：
    reader = DepthReader()
    depth_mm = reader.get_depth((cx, cy), depth_img)
    if depth_mm is not None:
        print(f'{depth_mm:.1f} mm')
"""

import numpy as np


class DepthReader:
    """
    RealSense 深度圖深度讀取物件。

    以 patch 中位數採樣提升穩定性，自動過濾無效像素（深度=0）。
    patch_size 為內部實作細節，外部不需感知。
    """

    def __init__(self, patch_size: int = 5):
        """
        patch_size：採樣區塊邊長（奇數），預設 5×5。
        越大越穩定但空間解析度越低；建議 3~7。
        """
        self._half = max(1, patch_size // 2)

    # ── 主要功能 ──────────────────────────────────────────────────────────────

    def get_depth(self, pixel, depth_img: np.ndarray):
        """
        取得指定像素位置的深度值（mm）。

        參數：
            pixel     : (cx, cy) 像素座標（float 或 int 皆可）
            depth_img : H×W uint16 numpy array，來自 RealSense aligned depth

        回傳：
            float  — 深度值（mm），patch 內有效像素的中位數
            None   — 越界、或 patch 內無有效深度（全為 0）時
        """
        if depth_img is None:
            return None

        H, W  = depth_img.shape[:2]
        cx, cy = float(pixel[0]), float(pixel[1])
        ui, vi = int(round(cx)), int(round(cy))

        if not (0 <= ui < W and 0 <= vi < H):
            return None

        h = self._half
        y0 = max(0, vi - h);  y1 = min(H, vi + h + 1)
        x0 = max(0, ui - h);  x1 = min(W, ui + h + 1)

        patch = depth_img[y0:y1, x0:x1]
        valid = patch[patch > 0]

        if len(valid) == 0:
            return None

        return float(np.median(valid))

    # ── 設定 ──────────────────────────────────────────────────────────────────

    def set_patch_size(self, patch_size: int):
        """動態調整採樣 patch 大小（內部實作，不影響外部呼叫方式）。"""
        self._half = max(1, patch_size // 2)

    @property
    def patch_size(self) -> int:
        return self._half * 2 + 1
