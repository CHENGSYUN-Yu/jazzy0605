"""
gpu_inference_switch.py — GPU 推論使用權切換物件（獨立工廠）

職責：
    管理頭部相機（YoloDetector）與手腕相機（YoloDetectHandcam）
    的 GPU 推論使用權，確保任何時刻只有一台相機進行 GPU 推論，
    避免 VRAM 競爭導致的 Conv ORT 錯誤。

機制：
    維護一個 owner 狀態（'head' / 'hand' / 'none'）。
    呼叫方在執行 infer() 前先查詢 can_head_infer() / can_hand_infer()，
    若不是使用權持有者則改呼叫 get_all()（回傳上次結果，不動 GPU）。

使用流程：
    switch = GpuInferenceSwitch()

    # 第一階段：頭部偵測
    switch.grant_to_head()
    if switch.can_head_infer():
        dets = head_detector.infer(frame)   # GPU
    else:
        dets = head_detector.get_all()      # 不動 GPU

    # 第二階段：手腕偵測（頭部讓出使用權）
    switch.grant_to_hand()
    if switch.can_hand_infer():
        dets = hand_detector.infer(frame)   # GPU
"""

import threading


class GpuInferenceSwitch:
    """
    GPU 推論使用權切換物件。

    線程安全，狀態切換使用 Lock 保護。
    不持有偵測器的引用，純粹管理「誰有權推論」的狀態。
    """

    HEAD = 'head'
    HAND = 'hand'
    NONE = 'none'

    def __init__(self, initial: str = 'none'):
        """
        initial : 初始使用權持有者，預設 'none'。
                  可傳入 'head' 或 'hand' 直接啟用。
        """
        self._owner = initial if initial in (self.HEAD, self.HAND, self.NONE) \
                      else self.NONE
        self._lock  = threading.Lock()

    # ── 使用權切換 ────────────────────────────────────────────────────────────

    def grant_to_head(self):
        """將 GPU 推論使用權交給頭部相機。"""
        with self._lock:
            self._owner = self.HEAD

    def grant_to_hand(self):
        """將 GPU 推論使用權交給手腕相機。"""
        with self._lock:
            self._owner = self.HAND

    def release(self):
        """釋放使用權（兩邊都不做 GPU 推論）。"""
        with self._lock:
            self._owner = self.NONE

    # ── 查詢 ──────────────────────────────────────────────────────────────────

    def can_head_infer(self) -> bool:
        """頭部相機目前是否可以進行 GPU 推論。"""
        with self._lock:
            return self._owner == self.HEAD

    def can_hand_infer(self) -> bool:
        """手腕相機目前是否可以進行 GPU 推論。"""
        with self._lock:
            return self._owner == self.HAND

    @property
    def owner(self) -> str:
        """目前持有推論使用權的相機：'head' / 'hand' / 'none'。"""
        with self._lock:
            return self._owner

    @property
    def is_idle(self) -> bool:
        """是否目前沒有任何相機持有使用權。"""
        with self._lock:
            return self._owner == self.NONE
