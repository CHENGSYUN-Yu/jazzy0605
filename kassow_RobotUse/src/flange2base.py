"""
flange2base.py — 法蘭座標系轉機器人基座座標系物件（獨立工廠）

職責：
    接收 HandCam2Flange 的輸出（pos_flange_mm），
    利用手臂當前 TCP 位姿（位置 + RPY），
    將法蘭座標系下的 3D 位置轉換為機器人基座座標系（base_link，mm）。

與其他座標轉換物件的差異：
    HeadCam2Base / HandCam2Flange 使用「固定矩陣」（T_matrix，校正一次後不變）。
    Flange2Base 使用「即時位姿」：手臂每幀都在動，
    必須持續更新 update_arm_pose() 才能得到正確的轉換結果。

轉換原理：
    1. RPY（Kassow 'xyz' intrinsic 慣例）→ 旋轉矩陣 R_flange2base
    2. P_base = R_flange2base @ P_flange + t_tcp
       t_tcp = 當前 TCP 位置（mm，来自 SystemState.pos）

外部使用方式：
    f2b = Flange2Base()

    # 每幀手臂狀態更新時呼叫（在 _on_state callback 裡）
    f2b.update_arm_pose(current_pos, current_rot)

    # 批次轉換（承接 HandCam2Flange.transform_all() 輸出）
    results = f2b.transform_all(instruments)
    for r in results:
        print(r['pos_base_mm'])   # [x, y, z] mm，base_link 座標系
"""

import copy
import math
import os

import numpy as np
from scipy.spatial.transform import Rotation

# Kassow SystemState.rot 的 RPY 慣例（內旋 xyz = Rx·Ry·Rz）
_RPY_CONVENTION = 'xyz'


def _rpy_to_rot(rx, ry, rz):
    """RPY（度）→ 3×3 旋轉矩陣（Kassow 慣例 'xyz' intrinsic）。"""
    r = Rotation.from_euler(_RPY_CONVENTION, [rx, ry, rz], degrees=True)
    return r.as_dcm() if hasattr(r, 'as_dcm') else r.as_matrix()


class Flange2Base:
    """
    法蘭座標系 → 機器人基座座標系轉換物件。

    需搭配即時手臂位姿更新（update_arm_pose），才能正確轉換。
    所有輸出皆深拷貝，內外互不影響。
    """

    def __init__(self):
        self._current_pos = None   # [x, y, z] mm，來自 SystemState.pos
        self._current_rot = None   # [roll, pitch, yaw] deg，來自 SystemState.rot

    # ── 手臂位姿更新 ──────────────────────────────────────────────────────────

    def update_arm_pose(self, pos: list, rot: list):
        """
        更新手臂當前 TCP 位姿（每幀從 SystemState 取得後呼叫）。

        參數：
            pos : [x, y, z]（mm），來自 Kassow SystemState.pos
            rot : [roll, pitch, yaw]（deg），來自 SystemState.rot
        """
        self._current_pos = list(pos)
        self._current_rot = list(rot)

    @property
    def is_ready(self) -> bool:
        """手臂位姿是否已更新。"""
        return self._current_pos is not None and self._current_rot is not None

    # ── 單一器械轉換 ──────────────────────────────────────────────────────────

    def transform(self, pos_flange_mm: list, Rz_flange=None):
        """
        將法蘭座標系下的位置與旋轉一起轉換為基座座標系。

        轉換公式：
            R_f2b  = rpy_to_rot(roll, pitch, yaw)   ← 當前 TCP 旋轉矩陣
            P_base = R_f2b @ P_flange + t_tcp        ← 位置轉換
            R_obj_base = R_f2b @ Rz_flange           ← 旋轉轉換
            yaw_base = atan2(R_obj_base[1,0], R_obj_base[0,0])

        回傳 dict：
            'pos_base_mm'  : [x, y, z]（mm）
            'yaw_base_deg' : float（度）
            'Rz_base'      : list（3×3 旋轉矩陣）
            None — 手臂位姿未更新或位置輸入無效時
        """
        if not self.is_ready or pos_flange_mm is None:
            return None

        R_f2b  = _rpy_to_rot(*self._current_rot)
        t      = np.array(self._current_pos, dtype=np.float64)
        p      = np.array(pos_flange_mm,     dtype=np.float64)
        P_base = R_f2b @ p + t

        # 旋轉轉換：R_obj_base = R_flange2base @ Rz_flange
        if Rz_flange is not None:
            Rz_f = np.array(Rz_flange, dtype=np.float64)
            R_obj_base = R_f2b @ Rz_f
            yaw_base_rad = math.atan2(float(R_obj_base[1, 0]),
                                      float(R_obj_base[0, 0]))
            yaw_base_deg = math.degrees(yaw_base_rad)
            Rz_base = R_obj_base.tolist()
        else:
            yaw_base_deg = None
            Rz_base      = None

        return {
            'pos_base_mm':  [float(P_base[0]), float(P_base[1]), float(P_base[2])],
            'yaw_base_deg': yaw_base_deg,
            'Rz_base':      Rz_base,
        }

    # ── 批次多器械轉換 ────────────────────────────────────────────────────────

    def transform_all(self, instruments: list) -> list:
        """
        批次處理多個器械，承接 Angle2RzHandcam.convert_all() 的輸出。

        輸入 instruments：list of dict，每個元素需包含：
            'pos_flange_mm' : [x, y, z]（mm）或 None
            'Rz'            : list（3×3，來自 Angle2RzHandcam）或 None

        回傳：list of dict（深拷貝），每個元素新增：
            'pos_base_mm'  : [x, y, z]（mm）或 None
            'yaw_base_deg' : float（度）或 None
            'Rz_base'      : list（3×3）或 None

        手臂位姿未更新時，以上三個欄位均為 None（pass through）。
        """
        results = []
        for inst in instruments:
            out    = copy.deepcopy(inst)
            result = self.transform(inst.get('pos_flange_mm'),
                                    inst.get('Rz')) \
                     if self.is_ready else None
            out['pos_base_mm']  = result['pos_base_mm']  if result else None
            out['yaw_base_deg'] = result['yaw_base_deg'] if result else None
            out['Rz_base']      = result['Rz_base']      if result else None
            results.append(out)
        return results
