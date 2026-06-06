"""
angle2rz.py — 2D 器械傾角轉基座座標系 Rz 旋轉物件（獨立工廠）

職責：
    將 YOLO 偵測到的器械 2D 傾角（影像平面），
    透過 T_headcam2base 轉換為機器人基座座標系的 Rz（偏航角）旋轉。

原理：
    1. 2D 傾角 θ（來自 _mask_edge_orientation，範圍 (-90, 90]°）
       → 相機座標系方向向量：dir_cam = [cos(θ), sin(θ), 0]
    2. 取 T_headcam2base 的旋轉部分 R（3×3），旋轉方向向量：
       dir_base = R @ dir_cam
    3. 取 XY 平面投影的偏航角：
       yaw_deg = atan2(dir_base[1], dir_base[0])

    yaw_deg 即為夾爪在 base_link 座標系下應採用的 Rz 旋轉角度。

外部使用方式：
    a2rz = Angle2Rz()
    a2rz.load_T('/root/kassow_ws/src/T_matrix.npy')

    # 單一器械
    yaw_deg = a2rz.convert(angle_deg)

    # 批次多器械（承接上游物件輸出）
    results = a2rz.convert_all(instruments)
    for r in results:
        print(r['yaw_deg'], r['Rz'])   # 偏航角 + 3×3 旋轉矩陣
"""

import copy
import math
import os

import numpy as np


class Angle2Rz:
    """
    2D 器械傾角 → 基座座標系 Rz 旋轉物件。

    T_headcam2base 由外部設定，轉換邏輯封裝在內部。
    所有輸出皆深拷貝，內外互不影響。
    """

    def __init__(self, T: np.ndarray = None):
        self._T = np.array(T, dtype=np.float64) if T is not None else None

    # ── T_matrix 設定 ─────────────────────────────────────────────────────────

    def set_T(self, T: np.ndarray):
        """直接設定 4×4 T_headcam2base（內部複製，不影響外部矩陣）。"""
        self._T = np.array(T, dtype=np.float64)

    def load_T(self, path: str):
        """從 .npy 檔載入 T_headcam2base。"""
        if not os.path.exists(path):
            raise FileNotFoundError(f'T_matrix 檔案不存在：{path}')
        self._T = np.load(path).astype(np.float64)

    @property
    def is_ready(self) -> bool:
        return self._T is not None

    # ── 單一器械轉換 ──────────────────────────────────────────────────────────

    def convert(self, angle_deg: float):
        """
        將單一 2D 傾角轉換為基座座標系偏航角與 Rz 旋轉矩陣。

        參數：
            angle_deg : YOLO _mask_edge_orientation 輸出的長軸角度（度，範圍 (-90, 90]）

        回傳 dict：
            'yaw_deg' : float  — 基座座標系偏航角（度）
            'yaw_rad' : float  — 基座座標系偏航角（弧度）
            'Rz'      : list   — 3×3 Rz 旋轉矩陣（繞 Z 軸）（深拷貝）

        T_matrix 未設定時回傳 None。
        """
        if not self.is_ready:
            return None

        theta = math.radians(float(angle_deg))
        dir_cam = np.array([math.cos(theta), math.sin(theta), 0.0])

        R_cam2base = self._T[:3, :3]
        dir_base   = R_cam2base @ dir_cam

        yaw_rad = math.atan2(float(dir_base[1]), float(dir_base[0]))
        yaw_deg = math.degrees(yaw_rad)

        Rz = _make_Rz(yaw_rad)

        return {
            'yaw_deg': yaw_deg,
            'yaw_rad': yaw_rad,
            'Rz':      Rz.tolist(),   # list 避免外部改到內部 ndarray
        }

    # ── 批次多器械轉換 ────────────────────────────────────────────────────────

    def convert_all(self, instruments: list) -> list:
        """
        批次處理多個器械，承接上游物件（YoloDetector / HeadCam2Base）的輸出。

        輸入 instruments：list of dict，每個元素需包含：
            'angle_deg' : float — YOLO 偵測到的 2D 傾角

        回傳：list of dict（深拷貝），每個元素在原有欄位基礎上新增：
            'yaw_deg' : float  — 基座座標系偏航角（度）
            'yaw_rad' : float  — 基座座標系偏航角（弧度）
            'Rz'      : list   — 3×3 Rz 旋轉矩陣

        T_matrix 未設定時回傳空 list。
        """
        results = []
        for inst in instruments:
            out    = copy.deepcopy(inst)
            result = self.convert(inst.get('angle_deg', 0.0)) \
                     if self.is_ready else None
            out['yaw_deg'] = result['yaw_deg'] if result else None
            out['yaw_rad'] = result['yaw_rad'] if result else None
            out['Rz']      = result['Rz']      if result else None
            results.append(out)

        return results


# ── 模組級工具函式 ─────────────────────────────────────────────────────────────

def _make_Rz(yaw_rad: float) -> np.ndarray:
    """繞 Z 軸旋轉 yaw_rad 弧度的 3×3 旋轉矩陣。"""
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    return np.array([[ c, -s, 0.0],
                     [ s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)
