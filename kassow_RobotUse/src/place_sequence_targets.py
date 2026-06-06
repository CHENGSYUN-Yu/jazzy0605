"""
place_sequence_targets.py — 放置動作三段目標位姿計算物件（獨立工廠）

職責：
    根據手臂當前位姿與 PutSiteGet 的放置點，
    計算放置動作的三段連續目標位姿，供 TrajectoryPlan 依序執行。

三段移動：
    1. Lift（Z +200mm）：
       原地抬升，避免器械在橫移時碰撞障礙物
       XY Rz 不動，Z 上升 lift_z_mm

    2. Approach（XY + Rz 移動到放置位置上方）：
       Z 保持抬升高度不變，橫移到放置點正上方並對齊 Rz

    3. Place（Z 下降到放置位置）：
       XY Rz 不動，Z 下降到 PutSiteGet 指定的放置高度

移動執行：
    三段目標位姿各自送入 TrajectoryPlan.plan() → ExecuteMotion → CheckArrive。
    移動物件已寫好，此物件只負責計算傳入值。

外部使用方式：
    pst = PlaceSequenceTargets(lift_z_mm=200.0)
    targets = pst.compute(current_pos, current_rot, put_site)

    # targets['lift']     → TrajectoryPlan 第 1 段
    # targets['approach'] → TrajectoryPlan 第 2 段
    # targets['place']    → TrajectoryPlan 第 3 段
"""

import copy


class PlaceSequenceTargets:
    """
    放置動作三段目標位姿計算物件。

    lift_z_mm : 第一段抬升距離（mm），預設 200mm。
    """

    def __init__(self, lift_z_mm: float = 200.0):
        self._lift_z = float(lift_z_mm)

    # ── 計算三段目標位姿 ──────────────────────────────────────────────────────

    def compute(self, current_pos: list, current_rot: list,
                put_site: dict) -> dict:
        """
        計算放置動作三段目標位姿。

        參數：
            current_pos : [x, y, z]（mm），當前手臂 TCP 位置
            current_rot : [roll, pitch, yaw]（deg），當前手臂 TCP 姿態
            put_site    : PutSiteGet.get() 的輸出
                          {'x_mm', 'y_mm', 'z_mm', 'yaw_deg'}

        回傳 dict（各段皆可直接傳入 TrajectoryPlan.plan() 的 target）：
            'lift'     : {'x_mm', 'y_mm', 'z_mm', 'yaw_deg'} — 抬升
            'approach' : {'x_mm', 'y_mm', 'z_mm', 'yaw_deg'} — 橫移到上方
            'place'    : {'x_mm', 'y_mm', 'z_mm', 'yaw_deg'} — 下降放置

        put_site 為 None 時回傳 None。
        """
        if put_site is None:
            return None

        cx   = float(current_pos[0])
        cy   = float(current_pos[1])
        cz   = float(current_pos[2])
        cyaw = float(current_rot[2])

        lifted_z = cz + self._lift_z

        px   = float(put_site['x_mm'])
        py   = float(put_site['y_mm'])
        pz   = float(put_site['z_mm'])
        pyaw = float(put_site['yaw_deg'])

        return {
            'lift': {
                'x_mm':    cx,
                'y_mm':    cy,
                'z_mm':    lifted_z,
                'yaw_deg': cyaw,
            },
            'approach': {
                'x_mm':    px,
                'y_mm':    py,
                'z_mm':    lifted_z,   # Z 保持抬升高度
                'yaw_deg': pyaw,
            },
            'place': {
                'x_mm':    px,
                'y_mm':    py,
                'z_mm':    pz,         # Z 下降到放置高度
                'yaw_deg': pyaw,
            },
        }

    # ── 設定 ──────────────────────────────────────────────────────────────────

    def set_lift_z(self, mm: float):
        """動態調整抬升距離。"""
        self._lift_z = float(mm)

    @property
    def lift_z_mm(self) -> float:
        return self._lift_z
