"""
ros2_node.py — 直接 rclpy 發布節點（供 500Hz 自動夾取使用）

不透過 subprocess，直接以 rclpy publisher 發布 JogLinear，
延遲 < 1ms，適合 500Hz 控制迴圈。

使用方式：
    from src.ros2_node import get_ros2_node
    node = get_ros2_node(domain_id=1)
    node.publish_jog([vx, vy, vz], [0, 0, rz])
    node.publish_stop()
    ok = node.call_gripper_io(index=1, value=1)
"""
import os
import threading
import time

_node = None
_node_lock = threading.Lock()


def get_ros2_node(domain_id: int = 1):
    """取得（或初始化）singleton rclpy 節點。thread-safe。"""
    global _node
    with _node_lock:
        if _node is not None:
            return _node
        try:
            os.environ['ROS_DOMAIN_ID'] = str(domain_id)
            import rclpy
            from rclpy.node import Node
            if not rclpy.ok():
                rclpy.init()
            _node = _KassowNode()
            threading.Thread(
                target=lambda: rclpy.spin(_node),
                daemon=True, name='rclpy_spin'
            ).start()
            print(f'[ros2_node] rclpy node started (domain_id={domain_id})')
        except Exception as e:
            print(f'[ros2_node] init failed: {e}')
            _node = _StubNode()
        return _node


# ── 正式 rclpy 節點 ───────────────────────────────────────────────────────────

class _KassowNode:
    """封裝 rclpy Node，提供 publish_jog / publish_stop / call_gripper_io。"""

    def __init__(self):
        from rclpy.node import Node
        from kr_msgs.msg import JogLinear
        from kr_msgs.srv import SetDiscreteOutput, MoveLinear

        self._node = Node('kassow_auto_grasp_node')
        self._pub  = self._node.create_publisher(
            JogLinear, '/kr/motion/jog_linear', 10)
        from kr_msgs.srv import SetInteractivity
        self._cli_gripper = self._node.create_client(
            SetDiscreteOutput, '/kr/iob/set_digital_output')
        self._cli_move = self._node.create_client(
            MoveLinear, '/kr/motion/move_linear')
        self._cli_interactivity = self._node.create_client(
            SetInteractivity, '/kr/motion/set_interactivity')
        self._JogLinear = JogLinear
        self._SetDiscreteOutput = SetDiscreteOutput
        self._MoveLinear = MoveLinear
        self._SetInteractivity = SetInteractivity
        self._lock = threading.Lock()

    # rclpy.spin 需要存取底層 node
    def __getattr__(self, name):
        return getattr(self._node, name)

    def publish_jog(self, vel: list, rot: list) -> None:
        msg = self._JogLinear()
        msg.vel = [float(v) for v in vel[:3]]
        msg.rot = [float(r) for r in rot[:3]]
        self._pub.publish(msg)

    def publish_stop(self) -> None:
        self.publish_jog([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])

    def call_move_linear(self,
                         pos: list,
                         rot: list,
                         speed_mm_s: float = 50.0,
                         ref: int = 0,
                         sync: float = 78.0,
                         timeout_sec: float = 30.0) -> bool:
        """
        呼叫 /kr/motion/move_linear 服務，等待手臂到達目標後回傳結果。

        pos       : [x, y, z] mm（base frame）
        rot       : [roll, pitch, yaw] deg
        speed_mm_s: TCP 移動速度（mm/s），使用 TT_VEL 模式
        ref       : 0=WORLD, 1=BASE, 2=TCP
        timeout_sec: 最長等待秒數（超時回傳 False）
        回傳 True = 到達目標；False = 失敗或超時

        注意：不使用 spin_until_future_complete，避免與 rclpy_spin 執行緒衝突。
              改用 threading.Event 等待 future 完成（由 spin 執行緒處理回調）。
        """
        try:
            if not self._cli_move.wait_for_service(timeout_sec=3.0):
                print('[ros2_node] move_linear service not available')
                return False
            req = self._MoveLinear.Request()
            req.pos    = [float(v) for v in pos[:3]]
            req.rot    = [float(v) for v in rot[:3]]
            req.ref    = int(ref)
            req.ttype  = 0           # TT_VEL = 0
            req.tvalue = float(speed_mm_s)
            print(f'[ros2_node] move_linear request: pos={[round(v,1) for v in pos]} '
                  f'rot={[round(v,1) for v in rot]} ref={ref} speed={speed_mm_s}mm/s')
            req.bpoint = 0           # BP_STOP = 0
            req.btype  = 0           # BT_ACC = 0
            req.bvalue = 1000.0
            req.sync   = float(sync)
            req.chaining = 0         # CH_INT = 0
            done_event = threading.Event()
            future = self._cli_move.call_async(req)
            future.add_done_callback(lambda _f: done_event.set())
            completed = done_event.wait(timeout=timeout_sec)
            if not completed:
                print('[ros2_node] move_linear timeout')
                return False
            result = future.result()
            print(f'[ros2_node] move_linear response: success={result.success}  '
                  f'(all fields: {vars(result) if hasattr(result, "__dict__") else result})')
            return bool(result.success)
        except Exception as e:
            print(f'[ros2_node] call_move_linear failed: {e}')
            return False

    def call_set_interactivity(self, enable: bool) -> bool:
        """
        切換機器人模式。
        enable=False → AUTONOMOUS（可執行 MoveLinear）
        enable=True  → MANUAL（可執行 JogLinear，恢復手動）
        """
        try:
            if not self._cli_interactivity.wait_for_service(timeout_sec=3.0):
                print('[ros2_node] set_interactivity service not available')
                return False
            req = self._SetInteractivity.Request()
            req.enable = bool(enable)
            done_event = threading.Event()
            future = self._cli_interactivity.call_async(req)
            future.add_done_callback(lambda _f: done_event.set())
            if not done_event.wait(timeout=5.0):
                print('[ros2_node] set_interactivity timeout')
                return False
            result = future.result()
            mode = 'MANUAL' if enable else 'AUTONOMOUS'
            print(f'[ros2_node] set_interactivity({mode}) → success={result.success}')
            return bool(result.success)
        except Exception as e:
            print(f'[ros2_node] call_set_interactivity failed: {e}')
            return False

    def call_gripper_io(self, index: int, value: int) -> bool:
        try:
            if not self._cli_gripper.wait_for_service(timeout_sec=2.0):
                print('[ros2_node] gripper service not available')
                return False
            req = self._SetDiscreteOutput.Request()
            req.index = int(index)
            req.value = int(value)
            print(f'[ros2_node] gripper IO index={index} value={value}')
            done_event = threading.Event()
            future = self._cli_gripper.call_async(req)
            future.add_done_callback(lambda _f: done_event.set())
            completed = done_event.wait(timeout=3.0)
            if not completed:
                return False
            return bool(future.result().success)
        except Exception as e:
            print(f'[ros2_node] call_gripper_io failed: {e}')
            return False


# ── Stub（rclpy 不可用時）────────────────────────────────────────────────────

class _StubNode:
    """rclpy 初始化失敗時的替代物件，印 log 但不報錯。"""

    def publish_jog(self, vel: list, rot: list) -> None:
        pass

    def publish_stop(self) -> None:
        pass

    def call_gripper_io(self, index: int, value: int) -> bool:
        print(f'[ros2_node stub] gripper IO index={index} value={value}')
        return True
