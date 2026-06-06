"""
pixel2handcam.py — 手腕相機（D405）像素座標轉相機座標系物件（獨立工廠）

職責：
    將手腕相機（RealSense D405）的器械重心像素 (u, v) 與深度值，
    利用 D405 相機內參反投影，轉換為手腕相機座標系下的 3D 位置（mm）。

    功能與 Pixel2HeadCam 相同，但完整獨立實作，相機內參對應 D405：
    - 相機內參由外部從 /cam1/color/camera_info 取得後設定
    - 可針對 D405 特性獨立調整，不影響頭部相機的座標轉換

座標系說明：
    手腕相機座標系（D405 camera frame）：
        X 軸：向右
        Y 軸：向下
        Z 軸：朝向物體（光軸方向）
    單位：mm

外部使用方式：
    p2h = Pixel2HandCam()
    p2h.set_intrinsics(fx, fy, cx, cy)   # 從 /cam1/color/camera_info 取得

    # 單一器械
    pos = p2h.project(u, v, depth_mm)    # → [x, y, z] mm 或 None

    # 批次
    results = p2h.project_all(instruments)  # → list（新增 pos_cam_mm）
"""

import copy


class Pixel2HandCam:
    """
    手腕相機（D405）像素座標轉相機座標系物件。

    相機內參（D405）由外部設定，轉換邏輯封裝在內部。
    完整獨立，不依賴 Pixel2HeadCam。
    """

    def __init__(self):
        self._fx = None
        self._fy = None
        self._cx = None
        self._cy = None

    # ── 內參設定 ──────────────────────────────────────────────────────────────

    def set_intrinsics(self, fx: float, fy: float,
                       cx: float, cy: float):
        """
        設定 D405 相機內參（來自 /cam1/color/camera_info）。

        參數：
            fx, fy : 焦距（像素單位）
            cx, cy : 主點（光軸在影像上的位置，像素單位）
        """
        self._fx = float(fx)
        self._fy = float(fy)
        self._cx = float(cx)
        self._cy = float(cy)

    @property
    def is_ready(self) -> bool:
        """D405 相機內參是否已設定。"""
        return None not in (self._fx, self._fy, self._cx, self._cy)

    # ── 單一器械轉換 ──────────────────────────────────────────────────────────

    def project(self, u: float, v: float, depth_mm: float):
        """
        將單一像素 (u, v) + 深度值轉換為手腕相機座標系 3D 位置。

        回傳：
            [x_cam, y_cam, z_cam]（mm）
            None — 內參未設定或深度無效（≤ 0）時
        """
        if not self.is_ready:
            return None
        if depth_mm is None or depth_mm <= 0:
            return None

        x_cam = (float(u) - self._cx) * depth_mm / self._fx
        y_cam = (float(v) - self._cy) * depth_mm / self._fy
        z_cam = float(depth_mm)

        return [x_cam, y_cam, z_cam]

    # ── 批次多器械轉換 ────────────────────────────────────────────────────────

    def project_all(self, instruments: list) -> list:
        """
        批次處理多個器械，轉換為手腕相機座標系。

        輸入 instruments：list of dict，每個元素需包含：
            'center'   : (u, v)  — 重心像素座標
            'depth_mm' : float   — 該像素的深度值（mm，來自 GetDepthHandcam）
            其他欄位原樣保留

        回傳：list of dict（深拷貝），每個元素新增：
            'pos_cam_mm' : [x, y, z]（mm）或 None（深度無效時）

        未設定內參時回傳空 list。
        """
        if not self.is_ready:
            return []

        results = []
        for inst in instruments:
            out = copy.deepcopy(inst)
            u, v     = inst['center']
            depth_mm = inst.get('depth_mm')
            out['pos_cam_mm'] = self.project(u, v, depth_mm)
            results.append(out)

        return results
