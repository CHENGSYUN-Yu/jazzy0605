"""
restore_instrument_sequence.py — 器械還原動作序列計算物件（獨立工廠）

職責：
    三支器械夾出並放置完成後，將所有器械依序夾回原先在托盤的位姿。
    由最後一個放置的器械開始，逆序執行，完全不需要視覺介入。

每支器械的動作序列（8 步）：
    1. MOVE → 放置台上方（z+300mm，yaw=-135°）
    2. MOVE → 放置台夾取高度（yaw=-135°）
    3. CLOSE_GRIPPER
    4. MOVE → 放置台上方（z+300mm，抬升）
    5. MOVE → 托盤原始位置上方（z+300mm，原始 yaw）
    6. MOVE → 托盤原始夾取高度（return_z_mm，原始 yaw）
    7. OPEN_GRIPPER
    8. MOVE → 托盤原始位置上方（z+300mm，離開）

最後一步：
    9. MOVE → Home 位姿

動作 dict 格式：
    {'type': 'move',          'target': {'x_mm', 'y_mm', 'z_mm', 'yaw_deg'}}
    {'type': 'close_gripper'}
    {'type': 'open_gripper'}

外部使用方式：
    rseq = RestoreInstrumentSequence(approach_offset_mm=300.0)
    actions = rseq.compute(put_sites, original_records, return_z_mm, home_target)
    # actions 是按執行順序排列的 list，逆序（最後放的先夾回）
"""

import copy


class RestoreInstrumentSequence:
    """
    器械還原動作序列計算物件。

    不含任何執行邏輯，只負責計算所有動作目標，
    實際執行由 AutoGrasp 負責。
    """

    def __init__(self, approach_offset_mm: float = 300.0):
        """
        approach_offset_mm : 接近和抬升時 z 軸偏移量（mm），預設 300mm。
        """
        self._offset = float(approach_offset_mm)

    # ── 序列計算 ──────────────────────────────────────────────────────────────

    def compute(self,
                put_sites:        list,
                original_records: list,
                return_z_mm:      float,
                home_target:      dict) -> list:
        """
        計算器械還原的完整動作序列。

        參數：
            put_sites        : list of dict {x_mm, y_mm, z_mm, yaw_deg}
                               長度 3，index 0=第1個放置點
            original_records : list of dict（來自 MemoryInstrumentPoint.get(i)）
                               每個 dict 需含 pos_base_mm[x,y,z] 和 yaw_deg
                               長度 3，index 0=第1個夾取記錄
            return_z_mm      : 法蘭面放回托盤時的目標 z（mm），
                               與原始夾取相同高度（GraspZOverride.z_mm 或計算值）
            home_target      : dict {x_mm, y_mm, z_mm, yaw_deg}（ReturnHomeTargets.home_pose）

        回傳：
            list of action dict，按執行順序排列（最後放的器械最先夾回）
            空 list 表示無有效資料
        """
        actions = []
        n = min(len(put_sites), len(original_records))

        # 逆序：最後夾出（放置）的器械先還原
        for i in range(n - 1, -1, -1):
            put  = put_sites[i]
            orig = original_records[i]
            if put is None or orig is None:
                continue

            orig_pos = orig.get('pos_base_mm', [0.0, 0.0, 0.0])
            orig_yaw = float(orig.get('yaw_deg') or -135.0)
            px, py, pz = float(put['x_mm']), float(put['y_mm']), float(put['z_mm'])
            ox, oy, oz = float(orig_pos[0]),  float(orig_pos[1]),  float(orig_pos[2])


            approach_put  = pz + self._offset
            approach_tray = oz + self._offset

            # ① 移動到放置台正上方
            actions.append({'type': 'move', 'target': {
                'x_mm': px, 'y_mm': py, 'z_mm': approach_put, 'yaw_deg': -135.0}})

            # ② 伸下去到夾取高度
            actions.append({'type': 'move', 'target': {
                'x_mm': px, 'y_mm': py, 'z_mm': pz, 'yaw_deg': -135.0}})

            # ③ 閉合夾爪
            actions.append({'type': 'close_gripper'})

            # ④ 抬升離開放置台
            actions.append({'type': 'move', 'target': {
                'x_mm': px, 'y_mm': py, 'z_mm': approach_put, 'yaw_deg': -135.0}})

            # ⑤ 移動到托盤原始位置正上方
            actions.append({'type': 'move', 'target': {
                'x_mm': ox, 'y_mm': oy, 'z_mm': approach_tray, 'yaw_deg': orig_yaw}})

            # ⑥ 伸下去到放回高度
            actions.append({'type': 'move', 'target': {
                'x_mm': ox, 'y_mm': oy, 'z_mm': return_z_mm, 'yaw_deg': orig_yaw}})

            # ⑦ 張開夾爪，器械還原
            actions.append({'type': 'open_gripper'})

            # ⑧ 抬升離開托盤
            actions.append({'type': 'move', 'target': {
                'x_mm': ox, 'y_mm': oy, 'z_mm': approach_tray, 'yaw_deg': orig_yaw}})

        # ⑨ 全部還原後回 Home
        if home_target is not None:
            actions.append({'type': 'move', 'target': copy.copy(home_target)})

        return actions

    # ── 設定 ──────────────────────────────────────────────────────────────────

    def set_approach_offset(self, mm: float) -> None:
        self._offset = float(mm)

    @property
    def approach_offset_mm(self) -> float:
        return self._offset
