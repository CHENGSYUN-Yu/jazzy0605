"""
angle2rz_handcam.py — 手腕相機 2D 器械傾角轉法蘭座標系 Rz 物件（獨立工廠）

職責：
    將 YoloDetectHandcam 偵測到的器械 2D 傾角（手腕相機影像平面），
    透過 T_handcam2flange 轉換為法蘭（flange）座標系的 Rz 偏航角。

    功能與 Angle2Rz 相同，但完整獨立實作：
    - T_matrix 對應 T_handcam2flange（EIH 校正，T_cam2gripper.npy）
    - 目標座標系為法蘭，而非機器人 base_link
    - 內部修改不影響 Angle2Rz

轉換原理：
    1. 2D 傾角 θ → 手腕相機座標系方向向量：dir_cam = [cos(θ), sin(θ), 0]
    2. 取 T_handcam2flange 旋轉部分 R（3×3）：
       dir_flange = R @ dir_cam
    3. 法蘭座標系偏航角：
       yaw_deg = atan2(dir_flange[1], dir_flange[0])

外部使用方式：
    a2rz = Angle2RzHandcam()
    a2rz.load_T('/root/kassow_ws/src/T_cam2gripper.npy')

    result = a2rz.convert(angle_deg)
    results = a2rz.convert_all(instruments)
    # 每個元素新增 yaw_deg, yaw_rad, Rz
"""

import copy
import math
import os

import numpy as np


class Angle2RzHandcam:
    """
    手腕相機 2D 傾角 → 法蘭座標系 Rz 旋轉物件。

    完整獨立，不依賴 Angle2Rz。
    所有輸出皆深拷貝，內外互不影響。
    """

    def __init__(self, T: np.ndarray = None):
        self._T = np.array(T, dtype=np.float64) if T is not None else None

    # ── T_matrix 設定 ─────────────────────────────────────────────────────────

    def set_T(self, T: np.ndarray):
        """直接設定 4×4 T_handcam2flange（內部複製，不影響外部矩陣）。"""
        self._T = np.array(T, dtype=np.float64)

    def load_T(self, path: str):
        """從 .npy 檔載入 T_handcam2flange（T_cam2gripper.npy）。"""
        if not os.path.exists(path):
            raise FileNotFoundError(f'T_matrix 檔案不存在：{path}')
        self._T = np.load(path).astype(np.float64)

    @property
    def is_ready(self) -> bool:
        return self._T is not None

    # ── 單一器械轉換 ──────────────────────────────────────────────────────────

    def convert(self, angle_deg: float):
        """
        將單一 2D 傾角轉換為法蘭座標系偏航角與 Rz 旋轉矩陣。

        參數：
            angle_deg : YoloDetectHandcam 偵測到的長軸角度（度，範圍 (-90, 90]）

        回傳 dict：
            'yaw_deg' : float  — 法蘭座標系偏航角（度）
            'yaw_rad' : float  — 法蘭座標系偏航角（弧度）
            'Rz'      : list   — 3×3 Rz 旋轉矩陣

        T_matrix 未設定時回傳 None。
        """
        if not self.is_ready:
            return None

        theta   = math.radians(float(angle_deg))
        dir_cam = np.array([math.cos(theta), math.sin(theta), 0.0])

        R_cam2flange = self._T[:3, :3]
        dir_flange   = R_cam2flange @ dir_cam

        yaw_rad = math.atan2(float(dir_flange[1]), float(dir_flange[0]))
        yaw_deg = math.degrees(yaw_rad)

        c, s = math.cos(yaw_rad), math.sin(yaw_rad)
        Rz = [[ c, -s, 0.0],
              [ s,  c, 0.0],
              [0.0, 0.0, 1.0]]

        return {
            'yaw_deg': yaw_deg,
            'yaw_rad': yaw_rad,
            'Rz':      Rz,
        }

    # ── 批次多器械轉換 ────────────────────────────────────────────────────────

    def convert_all(self, instruments: list) -> list:
        """
        批次處理多個器械，承接 YoloDetectHandcam 的輸出。

        輸入 instruments：list of dict，每個元素需包含：
            'angle_deg' : float

        回傳：list of dict（深拷貝），每個元素新增：
            'yaw_deg' / 'yaw_rad' / 'Rz'（T未設定時為 None）
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
