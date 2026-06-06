"""
which_first_handcam.py — 手腕相機夾取優先順序判斷物件（獨立工廠）

職責：
    接收手腕相機偵測到的多個器械（經過完整座標轉換），
    依據以下規則判斷夾取先後順序：

    主要規則：
        depth_mm 越小 → 器械距手腕相機越近（越淺）→ 越優先夾取

    次要規則（深度相近時）：
        center[0]（重心像素 u）越小 → 器械越靠左 → 越優先夾取

    與 WhichFirst 的差異：
        WhichFirst（頭部相機）：以 base frame Z 高度排序
        WhichFirstHandcam（手腕相機）：以相機深度排序，靠近且靠左優先

流程位置：
    Flange2Base.transform_all()（含 pos_base_mm + yaw_base_deg）
        ↓
    WhichFirstHandcam.get_first()       ← 此物件
        ↓ 最優先器械

外部使用方式：
    wfh = WhichFirstHandcam(depth_similar_thresh_mm=15.0)
    ranked = wfh.rank(instruments)
    first  = wfh.get_first(instruments)
"""

import copy


class WhichFirstHandcam:
    """
    手腕相機夾取優先順序判斷物件。

    深度淺（closer）優先；深度相近時以重心靠左（u 小）優先。
    完整獨立，不依賴 WhichFirst。
    輸出皆深拷貝，內外互不影響。
    """

    def __init__(self, depth_similar_thresh_mm: float = 15.0):
        """
        depth_similar_thresh_mm : 兩器械深度差值在此範圍內視為相近，
                                  改以重心 u 座標決定優先順序。預設 15mm。
        """
        self._depth_thresh = float(depth_similar_thresh_mm)

    # ── 設定 ──────────────────────────────────────────────────────────────────

    def set_depth_thresh(self, mm: float):
        self._depth_thresh = float(mm)

    @property
    def depth_similar_thresh_mm(self) -> float:
        return self._depth_thresh

    # ── 排序 ──────────────────────────────────────────────────────────────────

    def rank(self, instruments: list) -> list:
        """
        對所有器械依優先順序排序。

        輸入 instruments：list of dict，每個元素需包含：
            'depth_mm' : float — 器械距手腕相機的深度
            'center'   : (u, v) — 重心像素座標

        排序規則：
            1. depth_mm 越小越優先（越近越先夾）
            2. depth_mm 差值 ≤ depth_similar_thresh_mm 時，
               改以 center[0]（u）越小越優先（越靠左越先夾）

        回傳：排序後的 list（深拷貝），每個元素新增：
            'priority_rank'      : int   — 1 = 最優先
            'priority_depth_mm'  : float — 排序依據的深度值
            'priority_center_u'  : float — 排序依據的重心 u 座標

        depth_mm 為 None 的器械排在最後。
        """
        if not instruments:
            return []

        valid   = [i for i in instruments if i.get('depth_mm') is not None]
        invalid = [i for i in instruments if i.get('depth_mm') is None]

        if not valid:
            result = copy.deepcopy(invalid)
            for k, inst in enumerate(result):
                inst['priority_rank']     = k + 1
                inst['priority_depth_mm'] = None
                inst['priority_center_u'] = None
            return result

        # 最淺深度作為相近判斷基準
        min_depth = min(float(i['depth_mm']) for i in valid)

        def _key(inst):
            depth = float(inst['depth_mm'])
            u     = float(inst['center'][0]) if inst.get('center') else float('inf')
            # 深度在 thresh 範圍內視為相近 → 改以 u 排序
            depth_eff = min_depth if abs(depth - min_depth) <= self._depth_thresh \
                        else depth
            return (depth_eff, u)   # 兩者都越小越優先

        sorted_valid = sorted(valid, key=_key)

        result = []
        for k, inst in enumerate(sorted_valid):
            out = copy.deepcopy(inst)
            out['priority_rank']     = k + 1
            out['priority_depth_mm'] = round(float(inst['depth_mm']), 1)
            out['priority_center_u'] = round(float(inst['center'][0]), 1) \
                                       if inst.get('center') else None
            result.append(out)

        for k, inst in enumerate(invalid):
            out = copy.deepcopy(inst)
            out['priority_rank']     = len(sorted_valid) + k + 1
            out['priority_depth_mm'] = None
            out['priority_center_u'] = None
            result.append(out)

        return result

    def get_first(self, instruments: list):
        """取第一優先器械（priority_rank=1）；空時返回 None。"""
        ranked = self.rank(instruments)
        return ranked[0] if ranked else None
