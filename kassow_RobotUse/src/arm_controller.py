"""
Per-arm ROS2 connection, joint tracking, and state application logic.
Each ArmController instance owns one robot arm (arm_id="L" or "R").
"""
import datetime
import math
import os
import subprocess
import threading

import dearpygui.dearpygui as dpg
import yaml

from src.logger import Logger

# Left arm: ROS joint name → URDF slider name
_ROS_TO_SLIDER: dict[str, str] = {
    "RevoluteJointA1L": "Left_joint_1",
    "RevoluteJointA2L": "Left_joint_2",
    "RevoluteJointA3L": "Left_joint_3",
    "RevoluteJointA4L": "Left_jont_4",
    "RevoluteJointA5L": "Left_joint_5",
    "RevoluteJointA6L": "Left_joint_6",
    "RevoluteJointA7L": "Left_joint_7",
}

_LEFT_JOINT_LABELS  = ["J1", "J2", "J3", "J4", "J5", "J6", "J7"]
_RIGHT_JOINT_NAMES  = [
    "Right_joint_1", "Right_joint_2", "Right_joint_3",
    "Right_joint_4", "Right_joint_5", "Right_joint_6", "Right_link_7",
]
_RIGHT_JOINT_LABELS = ["J1", "J2", "J3", "J4", "J5", "J6", "J7"]

# URDF axis opposite to physical robot; negate before sending to viewer
_NEGATE_FOR_VIEWER: set[str] = {"Left_jont_4", "Right_joint_4"}

# cart jog buttons are shared (arm selected via radio button in UI)
_CONTROLS_CART: list[str] = (
    [f"cart_{ax}_{d}"
     for ax in ("X", "Y", "Z", "Rx", "Ry", "Rz")
     for d in ("neg", "pos")]
    + ["cart_arm_select"]
)

# dpg item tags that require ROS2 connection to be enabled
_CONTROLS_LEFT: list[str] = (
    ["btn_sync_joints", "chk_joint_track"]
    + [f"jog_{j}" for j in _ROS_TO_SLIDER.values()]
    + _CONTROLS_CART
)
_CONTROLS_RIGHT: list[str] = (
    ["btn_sync_joints_R", "chk_joint_track_R",
     "btn_ros2_sync_R", "chk_ros2_track_R"]
    + [f"jog_{j}" for j in _RIGHT_JOINT_NAMES]
    + _CONTROLS_CART
)


class ArmController:
    """Encapsulates all ROS2 connection and state logic for one robot arm."""

    def __init__(self, arm_id: str, logger: Logger, robot_viewer=None) -> None:
        assert arm_id in ("L", "R")
        self.arm_id = arm_id
        self._logger = logger
        self._rv = robot_viewer  # RobotViewer | None (only left arm updates 3D view)

        self._monitoring = False
        self._was_connected = False
        self._tracking = False
        self._track_proc: subprocess.Popen | None = None
        self._last_state_log = 0.0

        # 最新 TCP 位姿（供 auto_grasp 讀取）
        self.current_pos: list[float] | None = None   # [x, y, z] mm
        self.current_rot: list[float] | None = None   # [roll, pitch, yaw] deg
        self.current_joints: list[float] = []          # 7 關節角度 deg

    # ── tag shortcuts ─────────────────────────────────────────────────────────

    @property
    def _arm_name(self) -> str:
        return "左臂" if self.arm_id == "L" else "右臂"

    @property
    def _status_tag(self) -> str:
        return f"ros2_status_{self.arm_id}"

    @property
    def _domain_tag(self) -> str:
        return f"ros2_domain_{self.arm_id}"

    @property
    def _sync_status_tag(self) -> str:
        return f"joint_sync_status_{self.arm_id}"

    @property
    def _controls(self) -> list[str]:
        return _CONTROLS_LEFT if self.arm_id == "L" else _CONTROLS_RIGHT

    # ── public API ────────────────────────────────────────────────────────────

    def start_monitoring(self) -> None:
        if self._monitoring:
            return
        self._monitoring = True
        self._was_connected = False
        dpg.set_value(self._status_tag, "連線中...")
        dpg.configure_item(self._status_tag, color=(220, 200, 80))
        self._logger.log(f"[INFO] 開始監控 {self._arm_name} ROS2 連線（每 3 秒檢查一次）")
        threading.Thread(target=self._monitor_loop, daemon=True).start()

    def stop_monitoring(self) -> None:
        self._monitoring = False
        self._was_connected = False
        dpg.set_value(self._status_tag, "未連線")
        dpg.configure_item(self._status_tag, color=(150, 150, 150))
        self.set_controls_enabled(False)
        self._logger.log(f"[INFO] {self._arm_name} 停止監控")

    def sync_once(self) -> None:
        """One-shot joint state read (non-blocking)."""
        threading.Thread(target=self._sync_once_thread, daemon=True).start()

    def toggle_tracking(self, enabled: bool) -> None:
        if enabled:
            self._tracking = True
            dpg.set_value(self._sync_status_tag, "即時追蹤中")
            dpg.configure_item(self._sync_status_tag, color=(220, 200, 80))
            threading.Thread(target=self._track_loop, daemon=True).start()
        else:
            self._tracking = False
            if self._track_proc:
                self._track_proc.terminate()
                self._track_proc = None
            dpg.set_value(self._sync_status_tag, "")

    def apply_state(self, data: dict) -> None:
        """Update GUI sliders and 3D viewer from a kr_msgs/SystemState dict."""
        if not data:
            return

        sensed = data.get("sensed_pos", [])
        pos    = data.get("pos", [])
        rot    = data.get("rot", [])

        # 同步更新 current_pos / current_rot（供 auto_grasp 直接讀取）
        if len(pos) >= 3:
            self.current_pos = [float(pos[0]), float(pos[1]), float(pos[2])]
        if len(rot) >= 3:
            self.current_rot = [float(rot[0]), float(rot[1]), float(rot[2])]
        if len(sensed) >= 7:
            self.current_joints = [float(sensed[i]) for i in range(7)]

        if self.arm_id == "L":
            angles: dict[str, float] = {}
            for i, (slider_name, label) in enumerate(
                zip(_ROS_TO_SLIDER.values(), _LEFT_JOINT_LABELS)
            ):
                if i >= len(sensed):
                    break
                deg = float(sensed[i])
                dpg.set_value(f"jog_{slider_name}", deg)
                dpg.set_value(f"deg_{label}", f"{deg:.2f}°")
                rad = math.radians(deg)
                angles[slider_name] = -rad if slider_name in _NEGATE_FOR_VIEWER else rad
            if angles and self._rv:
                self._rv.set_angles(angles)
            for ax, val in zip(("X", "Y", "Z"), pos):
                dpg.set_value(f"tcp_{ax}", f"{float(val):.2f}")
            for ax, val in zip(("A", "B", "C"), rot):
                dpg.set_value(f"tcp_{ax}", f"{float(val):.2f}")
        else:
            angles: dict[str, float] = {}
            for i, (jname, label) in enumerate(
                zip(_RIGHT_JOINT_NAMES, _RIGHT_JOINT_LABELS)
            ):
                if i >= len(sensed):
                    break
                deg = float(sensed[i])
                dpg.set_value(f"jog_{jname}", deg)
                dpg.set_value(f"deg_R_{label}", f"{deg:.2f}°")
                rad = math.radians(deg)
                angles[jname] = -rad if jname in _NEGATE_FOR_VIEWER else rad
            if angles and self._rv:
                self._rv.set_angles(angles)
            for ax, val in zip(("X", "Y", "Z"), pos):
                dpg.set_value(f"tcp_R_{ax}", f"{float(val):.2f}")
            for ax, val in zip(("A", "B", "C"), rot):
                dpg.set_value(f"tcp_R_{ax}", f"{float(val):.2f}")

        now = datetime.datetime.now().timestamp()
        if now - self._last_state_log >= 5.0:
            self._last_state_log = now
            j_str = "  ".join(
                f"J{i+1}:{float(sensed[i]):7.2f}°" for i in range(min(7, len(sensed)))
            )
            p_str = (f"X:{float(pos[0]):.1f} Y:{float(pos[1]):.1f} Z:{float(pos[2]):.1f}"
                     if len(pos) >= 3 else "")
            r_str = (f"A:{float(rot[0]):.1f} B:{float(rot[1]):.1f} C:{float(rot[2]):.1f}"
                     if len(rot) >= 3 else "")
            self._logger.log(
                f"[STATE/{self._arm_name}] joints: {j_str} | TCP: {p_str} | rot: {r_str}"
            )

    def set_controls_enabled(self, enabled: bool) -> None:
        for tag in self._controls:
            if dpg.does_item_exist(tag):
                dpg.configure_item(tag, enabled=enabled)

    def cleanup(self) -> None:
        """Stop all background threads/processes owned by this arm."""
        self._monitoring = False
        self._was_connected = False
        self._tracking = False
        if self._track_proc and self._track_proc.poll() is None:
            self._track_proc.terminate()
            try:
                self._track_proc.wait(timeout=2)
            except Exception:
                self._track_proc.kill()
            self._track_proc = None

    # ── private ───────────────────────────────────────────────────────────────

    def _monitor_loop(self) -> None:
        domain_id = dpg.get_value(self._domain_tag)
        env = os.environ.copy()
        env["ROS_DOMAIN_ID"] = str(domain_id)

        while self._monitoring:
            connected = self._check_topics(env)

            if connected and not self._was_connected:
                self._was_connected = True
                dpg.set_value(self._status_tag, "已連線 ✓")
                dpg.configure_item(self._status_tag, color=(80, 220, 80))
                self.set_controls_enabled(True)
                self._logger.log(f"[INFO] {self._arm_name} 連線成功！偵測到 /kr/ topics")
                self._sync_master_speed(env)

            elif not connected and self._was_connected:
                self._was_connected = False
                dpg.set_value(self._status_tag, "連線中斷")
                dpg.configure_item(self._status_tag, color=(220, 80, 80))
                self.set_controls_enabled(False)
                self._logger.log(f"[WARN] {self._arm_name} /kr/ topics 消失，連線可能已中斷")

            threading.Event().wait(3.0)

    def _sync_master_speed(self, env: dict) -> None:
        """Read teach pendant master speed and update GUI speed slider."""
        try:
            result = subprocess.run(
                ["bash", "-c",
                 "source /opt/ros/jazzy/setup.bash && "
                 "source $HOME/ros2_ws/install/setup.bash 2>/dev/null && "
                 "ros2 service call /kr/system/get_master_speed "
                 "kr_msgs/srv/GetMasterSpeed 2>/dev/null"],
                capture_output=True, text=True, timeout=6, env=env,
            )
            for line in result.stdout.splitlines():
                if "speed=" in line:
                    val = float(line.split("speed=")[1].split(",")[0].split(")")[0])
                    pct = round(val * 100.0, 1)
                    if dpg.does_item_exist("jog_speed"):
                        dpg.set_value("jog_speed", pct)
                    self._logger.log(f"[INFO] {self._arm_name} 教導盒速度：{pct}%")
                    break
        except Exception:
            pass

    @staticmethod
    def _check_topics(env: dict) -> bool:
        try:
            result = subprocess.run(
                ["bash", "-c",
                 "source /opt/ros/jazzy/setup.bash && ros2 topic list 2>/dev/null"],
                capture_output=True, text=True, timeout=5, env=env,
            )
            return any(line.startswith("/kr/") for line in result.stdout.splitlines())
        except Exception:
            return False

    def _sync_once_thread(self) -> None:
        status_tag = self._sync_status_tag
        domain_id  = dpg.get_value(self._domain_tag)
        dpg.set_value(status_tag, "讀取中...")
        env = os.environ.copy()
        env["ROS_DOMAIN_ID"] = str(domain_id)
        try:
            result = subprocess.run(
                ["bash", "-c",
                 "source /opt/ros/jazzy/setup.bash && "
                 "source $HOME/ros2_ws/install/setup.bash 2>/dev/null && "
                 "ros2 topic echo /kr/system/state kr_msgs/msg/SystemState --once 2>/dev/null"],
                capture_output=True, text=True, timeout=8, env=env,
            )
            data = next(yaml.safe_load_all(result.stdout))
            self.apply_state(data)
            dpg.set_value(status_tag, "已同步 ✓")
            dpg.configure_item(status_tag, color=(80, 220, 80))
        except Exception as e:
            dpg.set_value(status_tag, f"失敗：{e}")
            dpg.configure_item(status_tag, color=(220, 80, 80))

    def _track_loop(self) -> None:
        domain_id = dpg.get_value(self._domain_tag)
        env = os.environ.copy()
        env["ROS_DOMAIN_ID"] = str(domain_id)
        cmd = (
            "source /opt/ros/jazzy/setup.bash && "
            "source $HOME/ros2_ws/install/setup.bash 2>/dev/null && "
            "ros2 topic echo /kr/system/state kr_msgs/msg/SystemState 2>/dev/null"
        )
        proc = subprocess.Popen(
            ["bash", "-c", cmd], env=env,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
        self._track_proc = proc

        block: list[str] = []
        for line in proc.stdout:
            if not self._tracking:
                break
            stripped = line.rstrip()
            if stripped == "---":
                if block:
                    try:
                        data = yaml.safe_load("\n".join(block))
                        self.apply_state(data)
                    except Exception:
                        pass
                    block = []
            else:
                block.append(stripped)

        proc.terminate()
