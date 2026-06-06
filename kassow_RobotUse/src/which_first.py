"""
which_first.py — 夾取優先順序判斷物件（獨立工廠）

職責：
    接收單一幀內所有器械在 base frame 的位姿（pos_base_mm + yaw_deg），
    依據以下規則判斷夾取先後順序：

    主要規則：
        pos_base_mm[2]（Z）越大 → 位置越高 → 越優先夾取

    次要規則（Z 完全相同，浮點容差 0.5mm）：
        pos_base_mm[1]（Y）越大 → 越優先夾取

    which_first 結束後，最優先的器械才會送進 TargetZCompute 計算
    手臂實際要移動過去的位姿。

流程位置：
    HeadCam2Base.transform_all() + Angle2Rz.convert_all()
        ↓
    WhichFirst.rank()                ← 此物件
        ↓ 排序後最優先的器械
    TargetZCompute.compute()
        ↓
    TrajectoryPlan.plan()

外部使用方式：
    wf = WhichFirst()

    ranked = wf.rank(instruments)       # 回傳排序後的 list（最優先在前）
    first  = wf.get_first(instruments)  # 直接取第一優先器械
"""

import copy



class WhichFirst:
    """
    夾取優先順序判斷物件。

    規則：Z 越大越優先（Z 高 = 位置上層）；
    Z 差距 ≤ 0.5mm（視為完全相同）時才以 Y 越大決定。
    輸出皆深拷貝，內外互不影響。
    """

    # ── 排序 ──────────────────────────────────────────────────────────────────

    def rank(self, instruments: list) -> list:
        """
        對所有器械依優先順序排序。

        輸入 instruments：list of dict，每個元素需包含：
            'pos_base_mm' : [x, y, z]（mm）— 來自 HeadCam2Base

        排序規則：
            1. pos_base_mm[2]（Z）越大越優先（主要）
            2. Z 差距 ≤ 0.5mm 時，pos_base_mm[1]（Y）越大越優先（次要）

        回傳：排序後的 list（深拷貝），每個元素新增：
            'priority_rank' : int   — 1 = 最優先，2 = 次優先，依此類推
            'priority_z_mm' : float — 排序依據的 Z 值
            'priority_y_mm' : float — 排序依據的 Y 值

        pos_base_mm 為 None 的器械排在最後。
        """
        if not instruments:
            return []

        valid   = [i for i in instruments if i.get('pos_base_mm') is not None]
        invalid = [i for i in instruments if i.get('pos_base_mm') is None]

        if not valid:
            result = copy.deepcopy(invalid)
            for k, inst in enumerate(result):
                inst['priority_rank'] = k + 1
                inst['priority_z_mm'] = None
                inst['priority_y_mm'] = None
            return result

        def _key(inst):
            z = float(inst['pos_base_mm'][2])
            y = float(inst['pos_base_mm'][1])
            return (-z, -y)   # Z 越大越前；Z 相同時 Y 越大越前

        sorted_valid = sorted(valid, key=_key)

        result = []
        for k, inst in enumerate(sorted_valid):
            out = copy.deepcopy(inst)
            out['priority_rank']  = k + 1
            out['priority_z_mm']  = round(float(inst['pos_base_mm'][2]), 1)
            out['priority_y_mm']  = round(float(inst['pos_base_mm'][1]), 1)
            result.append(out)

        # pos_base_mm=None 的排在最後
        for k, inst in enumerate(invalid):
            out = copy.deepcopy(inst)
            out['priority_rank']  = len(sorted_valid) + k + 1
            out['priority_z_mm']  = None
            out['priority_y_mm']  = None
            result.append(out)

        return result

    def get_first(self, instruments: list):
        """
        取第一優先器械（priority_rank=1）。

        回傳：dict（深拷貝）或 None（instruments 為空時）
        """
        ranked = self.rank(instruments)
        return ranked[0] if ranked else None
