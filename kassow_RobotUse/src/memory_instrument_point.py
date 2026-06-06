"""
memory_instrument_point.py — 器械目標位姿記憶物件（獨立工廠）

職責：
    紀錄準備夾取的器械在 base frame 的目標位姿（來自 TargetConsiderGripper），
    最多同時紀錄 3 個器械的位姿，供後續流程使用。

設計：
    固定 3 個槽位（slot 0~2）。
    record() 可指定槽位，或自動填入第一個空槽。
    所有輸出皆深拷貝，內外互不影響。

外部使用方式：
    mem = MemoryInstrumentPoint()

    # 自動填入
    slot = mem.record(target)          # 回傳使用的槽位 index (0~2)，滿了回傳 -1

    # 指定槽位
    mem.record(target, slot=1)

    # 讀取
    point = mem.get(0)                 # 取槽位 0 的資料（None 表示空）
    all_points = mem.get_all()         # 取全部（含 None）

    # 清除
    mem.clear(slot=0)                  # 清除槽位 0
    mem.clear()                        # 清除全部

    # 狀態
    mem.count()                        # 當前已佔用的槽位數（0~3）
    mem.total_recorded                 # 累計成功記錄次數（不因 clear 歸零）
    mem.is_full()                      # 是否 3 個都有資料
"""

import copy


class MemoryInstrumentPoint:
    """
    器械目標位姿記憶物件，最多記錄 3 個器械。

    每個槽位儲存一筆 TargetConsiderGripper 輸出：
        {'x_mm', 'y_mm', 'z_mm', 'yaw_deg', 'source'}

    total_recorded：累計成功記錄次數，每呼叫一次 record() 成功就 +1，
                    不因 clear() 歸零，供主程式判斷流程進度。
    """

    MAX_SLOTS = 3

    def __init__(self):
        self._slots          = [None] * self.MAX_SLOTS
        self._total_recorded = 0   # 累計計數器

    # ── 記錄 ──────────────────────────────────────────────────────────────────

    def record(self, instrument_point: dict, slot: int = None) -> int:
        """
        記錄器械目標位姿。

        參數：
            instrument_point : TargetConsiderGripper.compute() 的輸出 dict
            slot             : 指定槽位（0~2）；None 表示自動填入第一個空槽

        回傳：
            int  — 實際使用的槽位 index（0~2）
            -1   — 未指定槽位且已全滿；或指定槽位超出範圍
        """
        if slot is not None:
            if not (0 <= slot < self.MAX_SLOTS):
                return -1
            self._slots[slot] = copy.deepcopy(instrument_point)
            self._total_recorded += 1
            return slot

        # 自動填入第一個空槽
        for i, s in enumerate(self._slots):
            if s is None:
                self._slots[i] = copy.deepcopy(instrument_point)
                self._total_recorded += 1
                return i
        return -1   # 全滿（不計入）

    # ── 讀取 ──────────────────────────────────────────────────────────────────

    def get(self, slot: int):
        """
        取得指定槽位的器械位姿（深拷貝）。
        槽位超出範圍或為空時返回 None。
        """
        if not (0 <= slot < self.MAX_SLOTS):
            return None
        data = self._slots[slot]
        return copy.deepcopy(data) if data is not None else None

    def get_all(self) -> list:
        """
        取得全部 3 個槽位的資料（深拷貝）。
        空槽位以 None 表示，list 長度固定為 3。
        """
        return [copy.deepcopy(s) for s in self._slots]

    def get_recorded(self) -> list:
        """取得所有非空槽位的資料（深拷貝），跳過 None。"""
        return [copy.deepcopy(s) for s in self._slots if s is not None]

    # ── 清除 ──────────────────────────────────────────────────────────────────

    def clear(self, slot: int = None):
        """
        清除槽位。
        slot=None：清除全部；slot=0~2：清除指定槽位。
        """
        if slot is None:
            self._slots = [None] * self.MAX_SLOTS
        elif 0 <= slot < self.MAX_SLOTS:
            self._slots[slot] = None

    # ── 狀態查詢 ──────────────────────────────────────────────────────────────

    def count(self) -> int:
        """當前已佔用的槽位數量（0~3）。"""
        return sum(1 for s in self._slots if s is not None)

    @property
    def total_recorded(self) -> int:
        """累計成功記錄次數（不因 clear() 歸零）。"""
        return self._total_recorded

    def reset_counter(self):
        """重置累計計數器（需要時手動呼叫）。"""
        self._total_recorded = 0

    def is_full(self) -> bool:
        """是否 3 個槽位都已有資料。"""
        return all(s is not None for s in self._slots)

    def is_empty(self) -> bool:
        """是否全部槽位都是空的。"""
        return all(s is None for s in self._slots)

    def is_slot_occupied(self, slot: int) -> bool:
        """指定槽位是否有資料。"""
        if not (0 <= slot < self.MAX_SLOTS):
            return False
        return self._slots[slot] is not None
