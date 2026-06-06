"""
gripper_control.py — 夾爪控制獨立工廠物件

職責：
    封裝夾爪開合邏輯（DO01，Kassow IOBoard）。
    透過 service_callback 呼叫 /kr/iob/set_digital_output，
    不直接依賴 ROS，保持物件獨立可打包。

硬體資訊（來自 gripper_logic.md）：
    - IO 識別：DO01（index=1）
    - 閉合：index=1, value=1
    - 張開：index=1, value=0
    - Service：/kr/iob/set_digital_output（kr_msgs/srv/SetDiscreteOutput）
    - RMW：rmw_fastrtps_cpp（FastDDS，非 CycloneDDS）

延遲特性：
    首次 call 50–200ms（DDS 建連線），後續 1–10ms。
    建議啟動時呼叫 warmup() 預熱連線。

外部使用方式：
    def call_io(index, value):
        # ros2 service call /kr/iob/set_digital_output
        req = SetDiscreteOutput.Request()
        req.index = index; req.value = value
        return ros_node.call_service(req).success

    gc = GripperControl(service_callback=call_io)
    gc.warmup()         # 預熱（首次 DDS 連線）
    gc.close()          # 閉合 → True/False
    gc.open()           # 張開 → True/False
"""

import time


# 硬體常數（對應 gripper_logic.md）
_IO_INDEX    = 1   # DO01
_VALUE_CLOSE = 1   # ON  → 夾緊
_VALUE_OPEN  = 0   # OFF → 釋放


class GripperControl:
    """
    夾爪控制獨立工廠物件。

    透過 service_callback(index, value) → bool 呼叫 IO service，
    不直接依賴 ROS，可在任何地方使用。
    """

    def __init__(self, service_callback=None):
        """
        service_callback : callable(index: int, value: int) → bool
            呼叫 /kr/iob/set_digital_output 的函式。
            None 表示 stub 模式（只印 log，不實際動作）。
        """
        self._cb          = service_callback
        self._last_result = None

    # ── 預熱 ──────────────────────────────────────────────────────────────────

    def warmup(self) -> bool:
        """
        預熱連線（首次 DDS 連線需 50–200ms）。
        發送一次閉合再立即張開，讓 DDS 建立連線。
        """
        ok = self._send(_VALUE_CLOSE)
        time.sleep(0.05)
        self._send(_VALUE_OPEN)
        return ok

    # ── 開合 ──────────────────────────────────────────────────────────────────

    def close(self) -> bool:
        """
        夾爪閉合（index=1, value=1）。
        回傳：True = service 回傳 success；False = 失敗或無 callback。
        """
        result = self._send(_VALUE_CLOSE)
        self._last_result = {'action': 'close', 'success': result}
        return result

    def open(self) -> bool:
        """
        夾爪張開（index=1, value=0）。
        回傳：True = service 回傳 success；False = 失敗或無 callback。
        """
        result = self._send(_VALUE_OPEN)
        self._last_result = {'action': 'open', 'success': result}
        return result

    # ── 內部 ──────────────────────────────────────────────────────────────────

    def _send(self, value: int) -> bool:
        if self._cb is None:
            print(f'[GripperControl stub] index={_IO_INDEX} value={value}')
            return True
        try:
            return bool(self._cb(_IO_INDEX, value))
        except Exception as e:
            print(f'[GripperControl] service call failed: {e}')
            return False

    # ── 狀態查詢 ──────────────────────────────────────────────────────────────

    @property
    def last_result(self):
        return dict(self._last_result) if self._last_result else None
