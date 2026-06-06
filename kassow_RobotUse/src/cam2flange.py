"""
cam2flange.py — 手腕相機目標位置轉法蘭目標位置物件（獨立工廠）

職責：
    已知「希望手腕相機到達的目標位置」，
    利用 T_handcam2flange（含旋轉 R + 平移 t）計算「法蘭實際需要到達的位置」，
    使手腕相機（而非夾爪/法蘭面）抵達器械正上方。

使用情境：
    手腕相機裝在法蘭旁邊（非正中心），且有傾斜角度。
    頭部相機偵測出器械上方目標後，補償相機與法蘭之間的偏移，
    讓相機對準器械而非法蘭面。

轉換原理：
    相機中心在法蘭座標系中的位置：t_c = T_handcam2flange[:3, 3]
    法蘭在 base frame 的旋轉近似：
        R_f2b = R_z(yaw) @ T_handcam2flange[:3, :3]
        （結合法蘭偏航 + 相機在法蘭上的傾斜旋轉）

    法蘭目標位置 = 相機目標位置 - R_f2b @ t_c

外部使用方式：
    c2f = Cam2Flange()
    c2f.load_T('/root/kassow_ws/src/T_cam2gripper.npy')

    # 輸入：TargetZCompute 輸出（希望相機到達的位置）
    # 輸出：法蘭實際需要到達的位置
    flange_target = c2f.compute(cam_target)
"""

import copy
import math
import os

import numpy as np


class Cam2Flange:
    """
    手腕相機目標位置 → 法蘭目標位置轉換物件。

    使用 T_handcam2flange 的完整 R 和 t 計算補償偏移，
    考量相機安裝傾斜角度對偏移方向的影響。
    """

    def __init__(self, T: np.ndarray = None):
        self._T = None
        if T is not None:
            self.set_T(T)

    # ── T_matrix 設定 ─────────────────────────────────────────────────────────

    def set_T(self, T: np.ndarray):
        """直接設定 T_handcam2flange（4×4）。"""
        self._T = np.array(T, dtype=np.float64)

    def load_T(self, path: str):
        """從 .npy 檔載入 T_handcam2flange（T_cam2gripper.npy）。"""
        if not os.path.exists(path):
            raise FileNotFoundError(f'T_matrix 檔案不存在：{path}')
        self._T = np.load(path).astype(np.float64)

    @property
    def is_ready(self) -> bool:
        return self._T is not None

    # ── 轉換 ──────────────────────────────────────────────────────────────────

    def compute(self, cam_target: dict):
        """
        已知手腕相機應到達的目標位置，計算法蘭需要到達的位置。

        參數：
            cam_target : dict，需包含：
                'x_mm'    : float — 希望相機到達的 X（base frame）
                'y_mm'    : float — 希望相機到達的 Y（base frame）
                'z_mm'    : float — 希望相機到達的 Z（base frame）
                'yaw_deg' : float — 法蘭偏航角

        回傳 dict（深拷貝）：
            'x_mm'    : float — 法蘭目標 X（補償後）
            'y_mm'    : float — 法蘭目標 Y（補償後）
            'z_mm'    : float — 法蘭目標 Z（補償後）
            'yaw_deg' : float — 法蘭偏航角（不變）
            'source'  : dict  — 原始相機目標（深拷貝）

        T_matrix 未設定或輸入無效時回傳 None。
        """
        if not self.is_ready or cam_target is None:
            return None

        yaw_rad = math.radians(float(cam_target['yaw_deg']))
        c = math.cos(yaw_rad)
        s = math.sin(yaw_rad)

        # 法蘭繞 Z 軸旋轉矩陣（yaw）
        R_z = np.array([[ c, -s, 0.0],
                         [ s,  c, 0.0],
                         [0.0, 0.0, 1.0]])

        # T_flange2handcam = inv(T_handcam2flange)，包含完整 R 和 t
        # 其平移部分 = 法蘭原點在相機座標系的位置 = -R_C^T @ t_cam_in_flange
        T_inv = np.linalg.inv(self._T)
        t_flange_in_cam = T_inv[:3, 3]

        # 法蘭目標位置 = 相機目標位置 + R_z(yaw) @ t_flange_in_cam
        # 正號：inv 的平移已含負號，方向正確
        offset = R_z @ t_flange_in_cam

        return {
            'x_mm':    float(cam_target['x_mm'])    + float(offset[0]),
            'y_mm':    float(cam_target['y_mm'])    + float(offset[1]),
            'z_mm':    float(cam_target['z_mm'])    + float(offset[2]),
            'yaw_deg': float(cam_target['yaw_deg']),
            'source':  copy.deepcopy(cam_target),
        }
