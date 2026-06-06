"""
headcam2base.py — 頭部相機座標系轉機器人基座座標系物件（獨立工廠）

職責：
    接收 pixel2headcam 的輸出（pos_cam_mm），
    將相機座標系 3D 位置轉換為機器人基座座標系（base_link）。

    轉換方式：
        1. pos_cam_mm → 4×1 齊次向量 [x, y, z, 1]ᵀ
        2. P_base = T_headcam2base @ P_cam（4×4 矩陣乘法）
        3. 取前三維 → pos_base_mm

    支援批次處理多個器械。
    T_matrix 可從 .npy 檔載入或由外部直接設定。

外部使用方式：
    h2b = HeadCam2Base()
    h2b.load_T('/root/kassow_ws/src/T_matrix.npy')

    # 承接 pixel2headcam 的輸出，批次轉換
    results = h2b.transform_all(instruments)
    for r in results:
        print(r['pos_base_mm'])   # [x, y, z] mm，base_link 座標系
"""

import copy
import os

import numpy as np


class HeadCam2Base:
    """
    頭部相機座標系 → 機器人基座座標系轉換物件。

    T_matrix（4×4 齊次矩陣）為內部狀態，對外不暴露矩陣細節。
    所有輸出皆深拷貝，內外互不影響。
    """

    def __init__(self, T: np.ndarray = None):
        """
        T : 4×4 np.ndarray（可選），T_headcam2base 齊次轉換矩陣。
            也可之後呼叫 load_T() 或 set_T() 設定。
        """
        self._T = np.array(T, dtype=np.float64) if T is not None else None

    # ── T_matrix 設定 ─────────────────────────────────────────────────────────

    def set_T(self, T: np.ndarray):
        """直接設定 4×4 T_matrix（不影響外部傳入的矩陣）。"""
        self._T = np.array(T, dtype=np.float64)

    def load_T(self, path: str):
        """從 .npy 檔載入 T_matrix。"""
        if not os.path.exists(path):
            raise FileNotFoundError(f'T_matrix 檔案不存在：{path}')
        self._T = np.load(path).astype(np.float64)

    @property
    def is_ready(self) -> bool:
        """T_matrix 是否已設定。"""
        return self._T is not None

    # ── 單一器械轉換 ──────────────────────────────────────────────────────────

    def transform(self, pos_cam_mm: list):
        """
        將單一相機座標系位置轉換為基座座標系。

        參數：
            pos_cam_mm : [x, y, z]（mm），來自 Pixel2HeadCam.project()

        回傳：
            [x_base, y_base, z_base]（mm）
            None — T_matrix 未設定或輸入無效時
        """
        if not self.is_ready or pos_cam_mm is None:
            return None

        P_cam  = np.array([pos_cam_mm[0], pos_cam_mm[1], pos_cam_mm[2], 1.0],
                          dtype=np.float64)
        P_base = self._T @ P_cam
        return [float(P_base[0]), float(P_base[1]), float(P_base[2])]

    # ── 批次多器械轉換 ────────────────────────────────────────────────────────

    def transform_all(self, instruments: list) -> list:
        """
        批次處理多個器械，承接 Pixel2HeadCam.project_all() 的輸出。

        輸入 instruments：list of dict，每個元素需包含：
            'pos_cam_mm' : [x, y, z]（mm）或 None

        回傳：list of dict（深拷貝），每個元素在原有欄位基礎上新增：
            'pos_base_mm' : [x, y, z]（mm）或 None（輸入無效時）

        T_matrix 未設定時回傳空 list。
        """
        results = []
        for inst in instruments:
            out = copy.deepcopy(inst)
            out['pos_base_mm'] = self.transform(inst.get('pos_cam_mm')) \
                                 if self.is_ready else None
            results.append(out)

        return results
