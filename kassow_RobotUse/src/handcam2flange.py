"""
handcam2flange.py — 手腕相機座標系轉法蘭座標系物件（獨立工廠）

職責：
    接收 Pixel2HandCam 的輸出（pos_cam_mm），
    將手腕相機（D405）座標系下的 3D 位置，
    透過 T_matrix_handcam2flange 轉換為法蘭（flange）座標系（mm）。

    功能與 HeadCam2Base 相同，但完整獨立實作：
    - T_matrix 對應 EIH 校正結果（T_cam2gripper.npy）
    - 目標座標系為法蘭，而非機器人 base_link

轉換公式：
    P_flange = T_handcam2flange @ [x_cam, y_cam, z_cam, 1]ᵀ

外部使用方式：
    h2f = HandCam2Flange()
    h2f.load_T('/root/kassow_ws/src/T_cam2gripper.npy')

    # 批次（承接 Pixel2HandCam.project_all() 輸出）
    results = h2f.transform_all(instruments)
    for r in results:
        print(r['pos_flange_mm'])  # [x, y, z] mm，法蘭座標系
"""

import copy
import os

import numpy as np


class HandCam2Flange:
    """
    手腕相機座標系 → 法蘭座標系轉換物件。

    T_matrix（4×4 齊次矩陣，T_cam2gripper）由外部設定。
    完整獨立，不依賴 HeadCam2Base。
    所有輸出皆深拷貝，內外互不影響。
    """

    def __init__(self, T: np.ndarray = None):
        self._T = np.array(T, dtype=np.float64) if T is not None else None

    # ── T_matrix 設定 ─────────────────────────────────────────────────────────

    def set_T(self, T: np.ndarray):
        """直接設定 4×4 T_handcam2flange（內部複製，不影響外部矩陣）。"""
        self._T = np.array(T, dtype=np.float64)

    def load_T(self, path: str):
        """從 .npy 檔載入 T_handcam2flange（對應 EIH 校正的 T_cam2gripper）。"""
        if not os.path.exists(path):
            raise FileNotFoundError(f'T_matrix 檔案不存在：{path}')
        self._T = np.load(path).astype(np.float64)

    @property
    def is_ready(self) -> bool:
        return self._T is not None

    # ── 單一器械轉換 ──────────────────────────────────────────────────────────

    def transform(self, pos_cam_mm: list):
        """
        將單一手腕相機座標系位置轉換為法蘭座標系。

        回傳：
            [x_flange, y_flange, z_flange]（mm）
            None — T_matrix 未設定或輸入無效時
        """
        if not self.is_ready or pos_cam_mm is None:
            return None

        P_cam    = np.array([pos_cam_mm[0], pos_cam_mm[1], pos_cam_mm[2], 1.0],
                            dtype=np.float64)
        P_flange = self._T @ P_cam
        return [float(P_flange[0]), float(P_flange[1]), float(P_flange[2])]

    # ── 批次多器械轉換 ────────────────────────────────────────────────────────

    def transform_all(self, instruments: list) -> list:
        """
        批次處理多個器械，承接 Pixel2HandCam.project_all() 的輸出。

        輸入 instruments：list of dict，每個元素需包含：
            'pos_cam_mm' : [x, y, z]（mm）或 None

        回傳：list of dict（深拷貝），每個元素新增：
            'pos_flange_mm' : [x, y, z]（mm）或 None

        T_matrix 未設定時，pos_flange_mm 設為 None（pass through，不丟棄資料）。
        """
        results = []
        for inst in instruments:
            out = copy.deepcopy(inst)
            out['pos_flange_mm'] = self.transform(inst.get('pos_cam_mm')) \
                                   if self.is_ready else None
            results.append(out)
        return results
