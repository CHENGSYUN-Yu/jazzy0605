"""
pixel2headcam.py — 像素座標轉頭部相機座標系物件（獨立工廠）

職責：
    將器械重心像素 (u, v) 與深度值，利用相機內參反投影，
    轉換為頭部相機座標系下的 3D 位置（mm）。
    支援同時處理多個器械 bbox 的位姿資訊。

座標系說明：
    相機座標系（camera frame）：
        X 軸：向右
        Y 軸：向下
        Z 軸：朝向物體（光軸方向）
    單位：mm

轉換公式：
    x_cam = (u - cx) * z / fx
    y_cam = (v - cy) * z / fy
    z_cam = z（深度）

外部使用方式：
    p2c = Pixel2HeadCam()
    p2c.set_intrinsics(fx, fy, cx, cy)

    # 單一器械
    result = p2c.project(u, v, depth_mm)

    # 多個器械（批次）
    instruments = [
        {'center': (u1, v1), 'depth_mm': 850.0, 'conf': 0.92},
        {'center': (u2, v2), 'depth_mm': 910.0, 'conf': 0.88},
    ]
    results = p2c.project_all(instruments)
"""

import copy


class Pixel2HeadCam:
    """
    像素座標轉頭部相機座標系物件。

    相機內參（fx, fy, cx, cy）由外部設定，轉換邏輯封裝在內部。
    支援單一及批次多器械處理，輸出深拷貝確保內外隔離。
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
        設定相機內參（來自 CameraInfo topic）。

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
        """相機內參是否已設定。"""
        return None not in (self._fx, self._fy, self._cx, self._cy)

    # ── 單一器械轉換 ──────────────────────────────────────────────────────────

    def project(self, u: float, v: float, depth_mm: float):
        """
        將單一像素 (u, v) + 深度值轉換為相機座標系 3D 位置。

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
        批次處理多個器械，同時將所有器械的像素位置轉換為相機座標系。

        輸入 instruments：list of dict，每個元素需包含：
            'center'   : (u, v)  — 重心像素座標
            'depth_mm' : float   — 該像素的深度值（mm）
            其他欄位（conf, cls_id 等）原樣保留

        回傳：list of dict（深拷貝），每個元素在原有欄位基礎上新增：
            'pos_cam_mm' : [x, y, z]（mm）或 None（深度無效時）

        未設定內參時回傳空 list。
        """
        if not self.is_ready:
            return []

        results = []
        for inst in instruments:
            out = copy.deepcopy(inst)
            u, v      = inst['center']
            depth_mm  = inst.get('depth_mm')
            pos       = self.project(u, v, depth_mm)
            out['pos_cam_mm'] = pos
            results.append(out)

        return results
