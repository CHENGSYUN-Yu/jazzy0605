import dearpygui.dearpygui as dpg
import math
import numpy as np
import os
import subprocess
import threading

from src.arm_controller import (ArmController, _NEGATE_FOR_VIEWER,
                               _LEFT_JOINT_LABELS, _RIGHT_JOINT_LABELS)
from src.auto_grasp import AutoGrasp
from src.yolo_engine import YoloEngine
from src.ros2_node import get_ros2_node
from src.logger import Logger
from src.realsense import RealSense
from src.robot_viewer import RobotViewer

_CAM_LABELS = ["Cam 1", "Cam 2", "Cam 3"]

# 各軸度數範圍（依 URDF limit 換算，±360° 軸限縮至 ±180° 方便操作）
_JOINT_DEG_LIMITS: dict[str, tuple[float, float]] = {
    "Left_joint_1":  (-180, 180),
    "Left_joint_2":  (-70,  180),
    "Left_joint_3":  (-180, 180),
    "Left_jont_4":   (-70,  180),
    "Left_joint_5":  (-180, 180),
    "Left_joint_6":  (-70,  180),
    "Left_joint_7":  (-180, 180),
    "Right_joint_1": (-180, 180),
    "Right_joint_2": (-70,  180),
    "Right_joint_3": (-180, 180),
    "Right_joint_4": (-70,  180),
    "Right_joint_5": (-180, 180),
    "Right_joint_6": (-70,  180),
    "Right_link_7":  (-180, 180),
}

_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"

_RES_OPTIONS = {
    "640 × 480":   (640,  480),
    "848 × 480":   (848,  480),
    "1280 × 720":  (1280, 720),
    "424 × 240":   (424,  240),
}
_RES_LABELS = list(_RES_OPTIONS.keys())


class App:
    def __init__(self, screen_w: int = 1920, screen_h: int = 1080):
        # Layout：相機面板佔 33%，其餘給控制面板
        panel_h          = screen_h - 60   # 留給 OS 工具列
        cam_panel_w      = int(screen_w * 0.33)
        ctrl_panel_w     = screen_w - cam_panel_w - 30

        # 相機影像顯示大小（維持 4:3）
        img_w = cam_panel_w - 25
        img_h = int(img_w * 3 / 4)

        self._layout = {
            "win_w":        screen_w,
            "win_h":        screen_h,
            "panel_h":      panel_h,
            "cam_panel_w":  cam_panel_w,
            "ctrl_panel_w": ctrl_panel_w,
            "img_w":        img_w,
            "img_h":        img_h,
        }

        # 目前擷取解析度（材質尺寸）
        self._tex_w, self._tex_h = 1280, 720

        self._rs = RealSense(width=self._tex_w, height=self._tex_h, fps=30)

        self._rv = RobotViewer()   # render thread starts automatically

        self._cam_az   = RobotViewer.CAM_AZ_DEFAULT
        self._cam_el   = RobotViewer.CAM_EL_DEFAULT
        self._cam_dist = RobotViewer.CAM_DIST_DEFAULT
        self._last_mouse_pos: list[float] = [0.0, 0.0]

        _log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
        self._logger = Logger(log_dir=_log_dir)

        self._arms: dict[str, ArmController] = {
            "L": ArmController("L", self._logger, self._rv),
            "R": ArmController("R", self._logger, self._rv),
        }

        self._cart_stepping = False

        # ROI 拖移狀態
        self._roi_mode      = False   # 目前是否在 ROI 拖移模式
        self._roi_dragging  = False   # 滑鼠正在拖移中
        self._roi_press_tex = None    # 按下時的材質座標 (x, y)
        self._roi_drag_tex  = None    # 目前拖移位置的材質座標 (x, y)

        # 雙相機 YOLO 推論引擎（cam0=D435I, cam1=D405）
        from src.auto_grasp import MODEL_PATH
        self._yolo = YoloEngine(MODEL_PATH, cam_ids=[0, 1], fps=10.0)
        self._yolo.start(self._rs)

        # 自動夾取（右臂，domain_id=1）
        self._auto_grasp = AutoGrasp(
            arm_ctrl=self._arms["R"],
            domain_id=1,
            yolo_engine=self._yolo,
        )
        self._auto_grasp.set_realsense(self._rs)

    # =========================================================================
    # 入口
    # =========================================================================

    def setup(self):
        self._setup_texture()

        with dpg.window(label="Kassow RobotUse", tag="main_window",
                        width=self._layout["win_w"], height=self._layout["win_h"],
                        no_close=True, no_move=True,
                        no_resize=True, no_title_bar=True):
            self._build_ui()

        dpg.set_primary_window("main_window", True)
        self._logger.bind_gui("ros2_log")
        self._set_jog_controls_enabled(False)

        with dpg.handler_registry():
            dpg.add_mouse_move_handler(callback=self._on_viewer_mouse_move)

    def cleanup(self) -> None:
        """關閉視窗時釋放所有資源。"""
        self._logger.log("[INFO] 程式關閉，釋放所有資源...")
        self._yolo.stop()
        self._auto_grasp.cleanup()
        for arm in self._arms.values():
            arm.cleanup()
        self._rv.stop()
        try:
            self._rs.disconnect()
        except Exception:
            pass
        self._logger.log("[INFO] 資源釋放完成，再見！")

    def _set_jog_controls_enabled(self, enabled: bool, arm: str = "both") -> None:
        if arm == "both":
            for a in self._arms.values():
                a.set_controls_enabled(enabled)
        else:
            self._arms[arm].set_controls_enabled(enabled)

    # =========================================================================
    # 材質
    # =========================================================================

    def _setup_texture(self) -> None:
        w, h = self._tex_w, self._tex_h
        blank = np.zeros(w * h * 4, dtype=np.float32)
        blank[0::4] = 0.08
        blank[1::4] = 0.08
        blank[2::4] = 0.14
        blank[3::4] = 1.0
        rw, rh = RobotViewer.VIEW_W, RobotViewer.VIEW_H   # 640 × 480
        robot_blank = np.zeros(rw * rh * 4, dtype=np.float32)
        robot_blank[3::4] = 1.0
        with dpg.texture_registry():
            dpg.add_dynamic_texture(
                width=w, height=h,
                default_value=blank.tolist(),
                tag="cam_texture",
            )
            dpg.add_dynamic_texture(
                width=rw, height=rh,
                default_value=robot_blank.tolist(),
                tag="robot_texture",
            )

    def _recreate_texture(self, new_w: int, new_h: int) -> None:
        """解析度變更時重建材質並更新影像顯示元件。"""
        self._tex_w, self._tex_h = new_w, new_h
        if dpg.does_item_exist("cam_texture"):
            dpg.delete_item("cam_texture")
        blank = np.zeros(new_w * new_h * 4, dtype=np.float32)
        blank[0::4] = 0.08
        blank[1::4] = 0.08
        blank[2::4] = 0.14
        blank[3::4] = 1.0
        with dpg.texture_registry():
            dpg.add_dynamic_texture(
                width=new_w, height=new_h,
                default_value=blank.tolist(),
                tag="cam_texture",
            )
        # 更新影像元件綁定的材質（重新指定 texture_tag）
        dpg.configure_item("cam_image", texture_tag="cam_texture")

    # =========================================================================
    # 視窗佈局
    # =========================================================================

    def _build_ui(self) -> None:
        with dpg.menu_bar():
            with dpg.menu(label="File"):
                dpg.add_menu_item(label="Exit", callback=lambda: dpg.stop_dearpygui())
            with dpg.menu(label="Help"):
                dpg.add_menu_item(label="About", callback=self._show_about)

        with dpg.child_window(width=-1, height=-1, horizontal_scrollbar=True, border=False):
            with dpg.group(horizontal=True):
                self._build_camera_panel()
                dpg.add_spacer(width=8)
                self._build_control_panel()

    # =========================================================================
    # 左側：相機畫面面板
    # =========================================================================

    def _build_camera_panel(self) -> None:
        L = self._layout
        with dpg.child_window(label="Camera View",
                              width=L["cam_panel_w"], height=L["panel_h"], border=True):
            dpg.add_text("Camera View", color=(100, 200, 255))
            dpg.add_separator()
            dpg.add_spacer(height=4)

            dpg.add_text("選擇相機", color=(200, 200, 100))
            dpg.add_radio_button(
                items=_CAM_LABELS,
                default_value=_CAM_LABELS[0],
                horizontal=True,
                tag="cam_selector",
            )

            dpg.add_spacer(height=4)
            dpg.add_text("串流類型", color=(200, 200, 100))
            dpg.add_radio_button(
                items=["RGB", "IR"],
                default_value="RGB",
                horizontal=True,
                tag="stream_selector",
            )

            dpg.add_separator()

            dpg.add_image("cam_texture",
                          width=L["img_w"], height=L["img_h"],
                          tag="cam_image")

            dpg.add_separator()
            dpg.add_spacer(height=4)

            _CAM_ROLE = ["Cam 1 (頭部 D435I)", "Cam 2 (手部 D405)", "Cam 3"]
            dpg.add_text("裝置狀態", color=(200, 200, 100))
            for i in range(3):
                with dpg.group(horizontal=True):
                    dpg.add_text(f"{_CAM_ROLE[i]}:", indent=8)
                    dpg.add_spacer(width=4)
                    dpg.add_text("尚未偵測", color=(150, 150, 150), tag=f"cam_status_{i}")
                    dpg.add_spacer(width=8)
                    dpg.add_button(
                        label="連線",
                        tag=f"btn_cam_{i}",
                        width=55,
                        callback=self._on_toggle_cam,
                        user_data=i,
                    )

            dpg.add_spacer(height=8)

            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="連接所有相機",
                    callback=self._on_connect_all,
                    tag="btn_connect",
                    width=140,
                )
                dpg.add_spacer(width=6)
                dpg.add_button(
                    label="全部中斷",
                    callback=self._on_disconnect_all,
                    tag="btn_disconnect",
                    width=100,
                )

            dpg.add_spacer(height=6)
            dpg.add_separator()
            dpg.add_spacer(height=4)
            dpg.add_text("ROI 偵測範圍 (Cam 1 頭部相機)", color=(200, 200, 100))
            dpg.add_spacer(height=4)
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="✏ 拖移設定 ROI",
                    tag="btn_roi_set",
                    callback=self._on_roi_toggle,
                    width=140,
                )
                dpg.add_spacer(width=6)
                dpg.add_button(
                    label="✕ 清除 ROI",
                    callback=self._on_roi_clear,
                    width=100,
                )
            dpg.add_spacer(height=4)
            dpg.add_text("未設定", tag="roi_coord_lbl",
                         color=(150, 150, 150))
            self._refresh_roi_label()

    # =========================================================================
    # 右側：機器人控制面板
    # =========================================================================

    def _build_control_panel(self) -> None:
        L = self._layout
        with dpg.child_window(label="Controls",
                              width=L["ctrl_panel_w"], height=L["panel_h"], border=True):
            dpg.add_text("Control Panel", color=(100, 200, 255))
            dpg.add_separator()
            dpg.add_spacer(height=5)

            with dpg.tab_bar():
                with dpg.tab(label="Jog Control"):
                    with dpg.child_window(width=-1, height=-1, border=False):
                        self._build_jog_tab()
                with dpg.tab(label="自動夾取"):
                    with dpg.child_window(width=-1, height=-1, border=False):
                        self._auto_grasp.build_ui()
                with dpg.tab(label="Program"):
                    self._build_program_tab()
                with dpg.tab(label="手眼校正"):
                    self._build_calibration_tab()
                with dpg.tab(label="ROS2 操控"):
                    self._build_ros2_tab()
                with dpg.tab(label="Settings"):
                    self._build_settings_tab()

    def _build_jog_tab(self) -> None:
        L = self._layout
        joint_panel_w = (L["ctrl_panel_w"] - 60) // 2

        left_joints = [
            ("Left_joint_1", "L-J1"), ("Left_joint_2", "L-J2"),
            ("Left_joint_3", "L-J3"), ("Left_jont_4",  "L-J4"),
            ("Left_joint_5", "L-J5"), ("Left_joint_6", "L-J6"),
            ("Left_joint_7", "L-J7"),
        ]
        right_joints = [
            ("Right_joint_1", "R-J1"), ("Right_joint_2", "R-J2"),
            ("Right_joint_3", "R-J3"), ("Right_joint_4", "R-J4"),
            ("Right_joint_5", "R-J5"), ("Right_joint_6", "R-J6"),
            ("Right_link_7",  "R-J7"),
        ]

        ctrl_col_w = L["ctrl_panel_w"] - RobotViewer.VIEW_W - 32
        cam_w      = (ctrl_col_w - 60) // 3

        # ── 上方：3D 預覽（左）＋ 相機視角 / 位置同步 / 角度（右）並排 ────────
        with dpg.group(horizontal=True):

            # 左欄：3D 預覽
            with dpg.group():
                dpg.add_text("3D 預覽", color=(200, 200, 100))
                dpg.add_text("載入中...", tag="robot_status", color=(150, 150, 150))
                dpg.add_image("robot_texture",
                              width=RobotViewer.VIEW_W, height=RobotViewer.VIEW_H,
                              tag="robot_image")

            dpg.add_spacer(width=16)

            # 右欄：觀看視角 ＋ 左右臂並排
            with dpg.group():
                dpg.add_text("3D 視角控制", color=(200, 200, 100))
                dpg.add_text("左鍵：旋轉　右鍵：仰角　左右同按：縮放",
                             color=(150, 150, 150))

                dpg.add_spacer(height=6)
                dpg.add_separator()
                dpg.add_spacer(height=8)

                arm_col_w = (ctrl_col_w - 24) // 2
                with dpg.group(horizontal=True):
                    # ── 左臂資訊 ──────────────────────────────────────────
                    with dpg.group():
                        dpg.add_text("左臂 (Left)", color=(80, 160, 255))
                        dpg.add_button(label="同步當前位置",
                                       tag="btn_sync_joints",
                                       width=arm_col_w - 10,
                                       callback=self._on_sync_joints,
                                       user_data="L", enabled=False)
                        dpg.add_spacer(height=4)
                        with dpg.group(horizontal=True):
                            dpg.add_checkbox(label="即時追蹤",
                                             tag="chk_joint_track",
                                             callback=self._on_toggle_joint_track,
                                             user_data="L")
                            dpg.add_spacer(width=8)
                            dpg.add_text("", tag="joint_sync_status_L",
                                         color=(150, 150, 150))
                        dpg.add_spacer(height=6)
                        dpg.add_text("關節角度（度）", color=(180, 180, 180))
                        with dpg.group(horizontal=True):
                            for lbl in _LEFT_JOINT_LABELS:
                                with dpg.group():
                                    dpg.add_text(lbl, indent=2)
                                    dpg.add_text("---°", tag=f"deg_{lbl}",
                                                 color=(100, 220, 255))
                                    dpg.add_spacer(width=8)
                        dpg.add_spacer(height=6)
                        dpg.add_text("TCP 位置 (mm)", color=(180, 180, 180))
                        for ax in ("X", "Y", "Z"):
                            with dpg.group(horizontal=True):
                                dpg.add_text(f"  {ax} :", color=(180,180,180))
                                dpg.add_text("------", tag=f"tcp_{ax}",
                                             color=(100, 255, 180))
                        dpg.add_spacer(height=4)
                        dpg.add_text("TCP 姿態 (°)", color=(180, 180, 180))
                        for ax in ("A", "B", "C"):
                            with dpg.group(horizontal=True):
                                dpg.add_text(f"  {ax} :", color=(180,180,180))
                                dpg.add_text("------", tag=f"tcp_{ax}",
                                             color=(255, 200, 100))

                    dpg.add_spacer(width=16)

                    # ── 右臂資訊 ──────────────────────────────────────────
                    with dpg.group():
                        dpg.add_text("右臂 (Right)", color=(255, 160, 80))
                        dpg.add_button(label="同步當前位置",
                                       tag="btn_sync_joints_R",
                                       width=arm_col_w - 10,
                                       callback=self._on_sync_joints,
                                       user_data="R", enabled=False)
                        dpg.add_spacer(height=4)
                        with dpg.group(horizontal=True):
                            dpg.add_checkbox(label="即時追蹤",
                                             tag="chk_joint_track_R",
                                             callback=self._on_toggle_joint_track,
                                             user_data="R")
                            dpg.add_spacer(width=8)
                            dpg.add_text("", tag="joint_sync_status_R",
                                         color=(150, 150, 150))
                        dpg.add_spacer(height=6)
                        dpg.add_text("關節角度（度）", color=(180, 180, 180))
                        with dpg.group(horizontal=True):
                            for lbl in _RIGHT_JOINT_LABELS:
                                with dpg.group():
                                    dpg.add_text(lbl, indent=2)
                                    dpg.add_text("---°", tag=f"deg_R_{lbl}",
                                                 color=(255, 180, 80))
                                    dpg.add_spacer(width=8)
                        dpg.add_spacer(height=6)
                        dpg.add_text("TCP 位置 (mm)", color=(180, 180, 180))
                        for ax in ("X", "Y", "Z"):
                            with dpg.group(horizontal=True):
                                dpg.add_text(f"  {ax} :", color=(180,180,180))
                                dpg.add_text("------", tag=f"tcp_R_{ax}",
                                             color=(100, 255, 180))
                        dpg.add_spacer(height=4)
                        dpg.add_text("TCP 姿態 (°)", color=(180, 180, 180))
                        for ax in ("A", "B", "C"):
                            with dpg.group(horizontal=True):
                                dpg.add_text(f"  {ax} :", color=(180,180,180))
                                dpg.add_text("------", tag=f"tcp_R_{ax}",
                                             color=(255, 200, 100))

        # ── Speed ─────────────────────────────────────────────────────────────
        dpg.add_spacer(height=8)
        dpg.add_slider_float(label="Speed (%)", default_value=10.0,
                             min_value=1.0, max_value=100.0,
                             width=400, tag="jog_speed")

        # ── Joint 滑桿（左右並排）────────────────────────────────────────────
        dpg.add_spacer(height=8)
        dpg.add_separator()
        dpg.add_spacer(height=4)
        dpg.add_text("Joint Control", color=(200, 200, 100))
        dpg.add_spacer(height=4)

        with dpg.group(horizontal=True):
            with dpg.child_window(width=joint_panel_w, height=420, border=True):
                dpg.add_text("Left Arm (°)", color=(80, 160, 255))
                dpg.add_separator()
                for jname, label in left_joints:
                    lo, hi = _JOINT_DEG_LIMITS.get(jname, (-180, 180))
                    dpg.add_slider_float(
                        label=label, tag=f"jog_{jname}",
                        default_value=0.0, min_value=lo, max_value=hi,
                        width=joint_panel_w - 80, callback=self._on_joint_slider,
                    )

            dpg.add_spacer(width=8)

            with dpg.child_window(width=joint_panel_w, height=420, border=True):
                dpg.add_text("Right Arm (°)", color=(255, 160, 80))
                dpg.add_separator()
                for jname, label in right_joints:
                    lo, hi = _JOINT_DEG_LIMITS.get(jname, (-180, 180))
                    dpg.add_slider_float(
                        label=label, tag=f"jog_{jname}",
                        default_value=0.0, min_value=lo, max_value=hi,
                        width=joint_panel_w - 80, callback=self._on_joint_slider,
                    )

        # ── Cartesian Jog ─────────────────────────────────────────────────────
        dpg.add_spacer(height=8)
        dpg.add_separator()
        dpg.add_spacer(height=6)
        dpg.add_text("直線點動（Cartesian Jog）", color=(200, 200, 100))
        dpg.add_spacer(height=4)

        with dpg.group(horizontal=True):
            dpg.add_text("控制手臂：", color=(180, 180, 180))
            dpg.add_radio_button(
                items=["左臂 (L)", "右臂 (R)"],
                default_value="左臂 (L)", horizontal=True,
                tag="cart_arm_select",
            )
        dpg.add_spacer(height=6)

        # ── 步距選擇
        with dpg.group(horizontal=True):
            dpg.add_text("步距（位移）：", color=(180, 180, 180))
            dpg.add_radio_button(
                items=["10mm", "1mm", "0.1mm"],
                default_value="1mm", horizontal=True,
                tag="cart_lin_step",
            )
            dpg.add_spacer(width=30)
            dpg.add_text("步距（轉動）：", color=(180, 180, 180))
            dpg.add_radio_button(
                items=["10°", "1°", "0.1°"],
                default_value="1°", horizontal=True,
                tag="cart_rot_step",
            )
        dpg.add_spacer(height=8)

        cart_btn_w = 110
        # ── 位移（XYZ）
        dpg.add_text("位移", color=(180, 220, 255))
        with dpg.group(horizontal=True):
            for axis in ("X", "Y", "Z"):
                with dpg.group():
                    dpg.add_text(f"   {axis}", color=(180, 220, 255))
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="  −  ", width=cart_btn_w,
                                       tag=f"cart_{axis}_neg",
                                       callback=self._on_cart_step,
                                       user_data=(axis, -1))
                        dpg.add_spacer(width=4)
                        dpg.add_button(label="  +  ", width=cart_btn_w,
                                       tag=f"cart_{axis}_pos",
                                       callback=self._on_cart_step,
                                       user_data=(axis, +1))
                dpg.add_spacer(width=28)

        dpg.add_spacer(height=12)

        # ── 轉動（RxRyRz）
        dpg.add_text("轉動", color=(255, 200, 100))
        with dpg.group(horizontal=True):
            for axis in ("Rx", "Ry", "Rz"):
                with dpg.group():
                    dpg.add_text(f"   {axis}", color=(255, 200, 100))
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="  −  ", width=cart_btn_w,
                                       tag=f"cart_{axis}_neg",
                                       callback=self._on_cart_step,
                                       user_data=(axis, -1))
                        dpg.add_spacer(width=4)
                        dpg.add_button(label="  +  ", width=cart_btn_w,
                                       tag=f"cart_{axis}_pos",
                                       callback=self._on_cart_step,
                                       user_data=(axis, +1))
                dpg.add_spacer(width=28)
        dpg.add_spacer(height=16)

    def _build_program_tab(self) -> None:
        L = self._layout
        editor_w = L["ctrl_panel_w"] - 30
        dpg.add_spacer(height=5)
        dpg.add_text("Program")
        dpg.add_input_text(label="##program", multiline=True,
                           width=editor_w, height=400,
                           default_value="# Write your robot program here\n# Example:\n# move_j([0, 0, 0, 0, 0, 0, 0])\n",
                           tag="program_editor")
        dpg.add_spacer(height=5)
        with dpg.group(horizontal=True):
            dpg.add_button(label="Run",       width=100, callback=lambda: None)
            dpg.add_button(label="Stop",      width=100, callback=lambda: None)
            dpg.add_button(label="Clear Log", width=100, callback=self._clear_log)
        dpg.add_spacer(height=5)
        dpg.add_text("Log Output", color=(200, 200, 100))
        dpg.add_input_text(label="##log", multiline=True,
                           width=editor_w, height=120,
                           readonly=True, tag="log_output",
                           default_value="[INFO] Kassow RobotUse started.\n")

    def _build_calibration_tab(self) -> None:
        L = self._layout
        w = L["ctrl_panel_w"] - 30

        dpg.add_spacer(height=5)

        # ---- 校正設定 ----
        dpg.add_text("校正設定", color=(200, 200, 100))
        dpg.add_separator()
        dpg.add_spacer(height=4)

        with dpg.group(horizontal=True):
            dpg.add_text("使用相機：", indent=8)
            dpg.add_combo(items=_CAM_LABELS, default_value=_CAM_LABELS[0],
                          width=120, tag="calib_cam")
            dpg.add_spacer(width=20)
            dpg.add_text("串流：")
            dpg.add_combo(items=["RGB", "IR"], default_value="RGB",
                          width=80, tag="calib_stream")

        dpg.add_spacer(height=6)
        dpg.add_text("棋盤格設定", color=(180, 180, 180), indent=8)
        with dpg.group(horizontal=True):
            dpg.add_text("角點 X：", indent=8)
            dpg.add_input_int(label="##cx", default_value=9, width=80, tag="calib_cx")
            dpg.add_spacer(width=10)
            dpg.add_text("角點 Y：")
            dpg.add_input_int(label="##cy", default_value=6, width=80, tag="calib_cy")
            dpg.add_spacer(width=10)
            dpg.add_text("格子大小 (mm)：")
            dpg.add_input_float(label="##sq", default_value=25.0, width=80, tag="calib_sq")

        dpg.add_spacer(height=10)

        # ---- 校正流程 ----
        dpg.add_text("校正流程", color=(200, 200, 100))
        dpg.add_separator()
        dpg.add_spacer(height=4)

        with dpg.group(horizontal=True):
            dpg.add_button(label="擷取目前姿態", width=140,
                           callback=lambda: self._append_calib_log("[INFO] 姿態擷取（尚未實作）"))
            dpg.add_spacer(width=10)
            dpg.add_text("已擷取：", tag="calib_count_label")
            dpg.add_text("0 筆", tag="calib_count", color=(100, 220, 100))
            dpg.add_spacer(width=10)
            dpg.add_button(label="清除全部", width=90,
                           callback=self._clear_calib)

        dpg.add_spacer(height=5)
        dpg.add_input_text(label="##calib_poses", multiline=True, readonly=True,
                           width=w, height=120, tag="calib_poses",
                           default_value="")

        dpg.add_spacer(height=6)
        dpg.add_button(label="執行手眼校正", width=160,
                       callback=lambda: self._append_calib_log("[INFO] 校正執行（尚未實作）"))

        dpg.add_spacer(height=10)

        # ---- 校正結果 ----
        dpg.add_text("校正結果", color=(200, 200, 100))
        dpg.add_separator()
        dpg.add_spacer(height=4)

        dpg.add_input_text(label="##calib_result", multiline=True, readonly=True,
                           width=w, height=100, tag="calib_result",
                           default_value="（尚未校正）")
        dpg.add_spacer(height=5)
        with dpg.group(horizontal=True):
            dpg.add_button(label="儲存校正結果", width=130, callback=lambda: None)
            dpg.add_spacer(width=10)
            dpg.add_button(label="載入校正結果", width=130, callback=lambda: None)

        dpg.add_spacer(height=8)
        dpg.add_text("校正記錄", color=(200, 200, 100))
        dpg.add_input_text(label="##calib_log", multiline=True, readonly=True,
                           width=w, height=80, tag="calib_log", default_value="")

    def _build_ros2_tab(self) -> None:
        L = self._layout
        w = L["ctrl_panel_w"] - 30

        dpg.add_spacer(height=5)

        # ---- 連線設定（左臂 / 右臂） ----
        dpg.add_text("連線設定", color=(200, 200, 100))
        dpg.add_separator()
        dpg.add_spacer(height=6)

        col_w = (w - 40) // 2
        with dpg.group(horizontal=True):
            # ── 左臂 ──────────────────────────────────────────
            with dpg.child_window(width=col_w, height=210, border=True):
                dpg.add_text("左臂 (Left)", color=(80, 160, 255))
                dpg.add_separator()
                dpg.add_spacer(height=4)
                with dpg.group(horizontal=True):
                    dpg.add_text("Domain ID：", indent=4)
                    dpg.add_input_int(label="##domain_L", default_value=0,
                                      width=120, step=0, tag="ros2_domain_L")
                dpg.add_spacer(height=6)
                dpg.add_button(label="啟動連線", width=180,
                               callback=self._on_ros2_start, user_data="L",
                               tag="btn_ros2_start_L")
                dpg.add_spacer(height=4)
                dpg.add_button(label="停止連線", width=180,
                               callback=self._on_ros2_stop, user_data="L",
                               tag="btn_ros2_stop_L")
                dpg.add_spacer(height=8)
                with dpg.group(horizontal=True):
                    dpg.add_text("狀態：", indent=4)
                    dpg.add_text("未連線", color=(150, 150, 150),
                                 tag="ros2_status_L")

            dpg.add_spacer(width=16)

            # ── 右臂 ──────────────────────────────────────────
            with dpg.child_window(width=col_w, height=310, border=True):
                dpg.add_text("右臂 (Right)", color=(255, 160, 80))
                dpg.add_separator()
                dpg.add_spacer(height=4)
                with dpg.group(horizontal=True):
                    dpg.add_text("Domain ID：", indent=4)
                    dpg.add_input_int(label="##domain_R", default_value=1,
                                      width=120, step=0, tag="ros2_domain_R")
                dpg.add_spacer(height=6)
                dpg.add_button(label="啟動連線", width=180,
                               callback=self._on_ros2_start, user_data="R",
                               tag="btn_ros2_start_R")
                dpg.add_spacer(height=4)
                dpg.add_button(label="停止連線", width=180,
                               callback=self._on_ros2_stop, user_data="R",
                               tag="btn_ros2_stop_R")
                dpg.add_spacer(height=8)
                with dpg.group(horizontal=True):
                    dpg.add_text("狀態：", indent=4)
                    dpg.add_text("未連線", color=(150, 150, 150),
                                 tag="ros2_status_R")
                dpg.add_spacer(height=10)
                dpg.add_separator()
                dpg.add_spacer(height=6)
                dpg.add_button(label="同步當前位置", width=180,
                               callback=self._on_sync_joints, user_data="R",
                               tag="btn_ros2_sync_R", enabled=False)
                dpg.add_spacer(height=4)
                with dpg.group(horizontal=True):
                    dpg.add_checkbox(label="即時追蹤",
                                     tag="chk_ros2_track_R",
                                     callback=self._on_toggle_joint_track,
                                     user_data="R")
                    dpg.add_spacer(width=8)
                    dpg.add_text("", tag="joint_sync_status_ros2_R",
                                 color=(150, 150, 150))

        dpg.add_spacer(height=10)

        # ---- 話題訂閱 ----
        dpg.add_text("話題訂閱", color=(200, 200, 100))
        dpg.add_separator()
        dpg.add_spacer(height=4)

        with dpg.group(horizontal=True):
            dpg.add_text("Topic：", indent=8)
            dpg.add_input_text(label="##sub_topic", default_value="/joint_states",
                               width=250, tag="ros2_sub_topic")
            dpg.add_spacer(width=8)
            dpg.add_button(label="訂閱", width=70,
                           callback=lambda: self._logger.log(f"[INFO] 訂閱 {dpg.get_value('ros2_sub_topic')}（尚未實作）"))

        dpg.add_spacer(height=4)
        dpg.add_input_text(label="##sub_data", multiline=True, readonly=True,
                           width=w, height=80, tag="ros2_sub_data",
                           default_value="")

        dpg.add_spacer(height=10)

        # ---- 話題發布 ----
        dpg.add_text("話題發布", color=(200, 200, 100))
        dpg.add_separator()
        dpg.add_spacer(height=4)

        with dpg.group(horizontal=True):
            dpg.add_text("Topic：", indent=8)
            dpg.add_input_text(label="##pub_topic", default_value="/cmd_vel",
                               width=250, tag="ros2_pub_topic")

        dpg.add_spacer(height=4)
        dpg.add_input_text(label="##pub_msg", multiline=True,
                           width=w, height=60, tag="ros2_pub_msg",
                           default_value='{"linear": {"x": 0.0}, "angular": {"z": 0.0}}')
        dpg.add_spacer(height=4)
        dpg.add_button(label="發布", width=80,
                       callback=lambda: self._logger.log(f"[INFO] 發布到 {dpg.get_value('ros2_pub_topic')}（尚未實作）"))

        dpg.add_spacer(height=8)
        dpg.add_text("ROS2 記錄", color=(200, 200, 100))
        dpg.add_input_text(label="##ros2_log", multiline=True, readonly=True,
                           width=w, height=300, tag="ros2_log", default_value="")

    def _build_settings_tab(self) -> None:
        dpg.add_spacer(height=5)
        dpg.add_text("Camera Resolution", color=(200, 200, 100))
        dpg.add_combo(
            items=_RES_LABELS,
            default_value="1280 × 720",
            width=200,
            tag="cam_resolution",
        )
        dpg.add_text("（重新連接後生效）", color=(150, 150, 150))

        dpg.add_spacer(height=10)
        dpg.add_text("Display Settings", color=(200, 200, 100))
        dpg.add_checkbox(label="Dark Theme", default_value=True, tag="dark_theme")

        dpg.add_spacer(height=10)
        dpg.add_text("字型大小", color=(200, 200, 100))
        with dpg.group(horizontal=True):
            dpg.add_slider_int(
                label="##font_size",
                default_value=36, min_value=10, max_value=48,
                width=200, tag="font_size",
            )
            dpg.add_button(label="套用", width=60, callback=self._apply_font_size)

        dpg.add_spacer(height=10)
        dpg.add_button(label="Save Settings", callback=lambda: None)

    # =========================================================================
    # 相機連線回呼
    # =========================================================================

    def _on_connect_all(self) -> None:
        dpg.configure_item("btn_connect", enabled=False, label="連接中...")
        threading.Thread(target=self._connect_thread, daemon=True).start()

    def _connect_thread(self) -> None:
        # 讀取 Settings 選擇的解析度
        res_label = dpg.get_value("cam_resolution")
        new_w, new_h = _RES_OPTIONS.get(res_label, (640, 480))

        # 解析度有變更時重建材質
        if new_w != self._tex_w or new_h != self._tex_h:
            self._recreate_texture(new_w, new_h)

        self._rs.width  = new_w
        self._rs.height = new_h
        self._rs.connect()
        self._refresh_cam_status()
        dpg.configure_item("btn_connect", enabled=True, label="連接所有相機")

    def _on_toggle_cam(self, sender, app_data, user_data: int) -> None:
        """單台相機連線／中斷按鈕。"""
        idx = user_data
        dpg.configure_item(f"btn_cam_{idx}", enabled=False)
        threading.Thread(target=self._toggle_cam_thread, args=(idx,), daemon=True).start()

    def _toggle_cam_thread(self, idx: int) -> None:
        if self._rs.is_connected(idx):
            self._rs.disconnect_one(idx)
        else:
            self._rs.connect_one(idx)
        self._refresh_cam_status()
        dpg.configure_item(f"btn_cam_{idx}", enabled=True)

    def _on_disconnect_all(self) -> None:
        self._rs.disconnect()
        self._refresh_cam_status()
        self._reset_texture()

    # =========================================================================
    # 狀態列更新
    # =========================================================================

    def _refresh_cam_status(self) -> None:
        devices = self._rs.devices

        for i in range(3):
            if i >= len(devices):
                dpg.set_value(f"cam_status_{i}", "尚未偵測")
                dpg.configure_item(f"cam_status_{i}", color=(150, 150, 150))
                dpg.configure_item(f"btn_cam_{i}", label="連線")
                continue

            info       = devices[i]
            short_name = info.name.replace("Intel RealSense ", "").replace("Depth Camera ", "")

            if self._rs.is_connected(i):
                dpg.set_value(f"cam_status_{i}", f"{short_name}  ✓ 已連線")
                dpg.configure_item(f"cam_status_{i}", color=(80, 220, 80))
                dpg.configure_item(f"btn_cam_{i}", label="中斷")
            else:
                dpg.set_value(f"cam_status_{i}", f"{short_name}  ✗ 未連線")
                dpg.configure_item(f"cam_status_{i}", color=(220, 80, 80))
                dpg.configure_item(f"btn_cam_{i}", label="連線")

    # =========================================================================
    # 材質更新（render loop 呼叫）
    # =========================================================================

    def update_camera(self) -> None:
        self._auto_grasp.tick()
        self._refresh_roi_label()

        selected_label = dpg.get_value("cam_selector")
        stream         = dpg.get_value("stream_selector")

        try:
            cam_idx = _CAM_LABELS.index(selected_label)
        except ValueError:
            cam_idx = 0

        # ROI 拖移處理（只對 cam0）
        if cam_idx == 0 and stream == "RGB":
            self._handle_roi_drag()

        # RGB 模式：優先顯示 YOLO overlay（cam0 和 cam1 都支援）
        if stream == "RGB" and self._yolo.is_loaded:
            overlay = self._yolo.get_overlay_tex(cam_idx)
            if overlay is not None:
                dpg.set_value("cam_texture", overlay)
                return
        tex_data = self._rs.get_texture(cam_idx, stream)
        if tex_data is not None:
            dpg.set_value("cam_texture", tex_data)

        # 更新機器人 3D 預覽
        if self._rv.ready:
            if dpg.does_item_exist("robot_status"):
                dpg.set_value("robot_status", "")
            tex = self._rv.get_texture()
            if tex is not None:
                dpg.set_value("robot_texture", tex)
        elif self._rv.error:
            if dpg.does_item_exist("robot_status"):
                dpg.set_value("robot_status", f"[錯誤] {self._rv.error}")
        else:
            if dpg.does_item_exist("robot_status"):
                dpg.set_value("robot_status", "載入中...")

    # =========================================================================
    # ROI 設定
    # =========================================================================

    def _on_roi_toggle(self) -> None:
        """切換 ROI 拖移模式。"""
        self._roi_mode = not self._roi_mode
        self._roi_dragging  = False
        self._roi_press_tex = None
        self._roi_drag_tex  = None
        if self._roi_mode:
            dpg.configure_item("btn_roi_set",
                               label="⬛ 取消設定（拖移中...）")
            self._yolo.clear_preview_roi(0)
        else:
            dpg.configure_item("btn_roi_set", label="✏ 拖移設定 ROI")
            self._yolo.clear_preview_roi(0)

    def _on_roi_clear(self) -> None:
        self._yolo.clear_roi(0)
        self._roi_mode      = False
        self._roi_dragging  = False
        self._roi_press_tex = None
        dpg.configure_item("btn_roi_set", label="✏ 拖移設定 ROI")
        self._refresh_roi_label()

    def _refresh_roi_label(self) -> None:
        roi = self._yolo.get_roi(0) if self._yolo else None
        if dpg.does_item_exist("roi_coord_lbl"):
            if roi:
                x1, y1, x2, y2 = roi
                dpg.set_value("roi_coord_lbl",
                              f"x: {x1}–{x2}  y: {y1}–{y2}")
                dpg.configure_item("roi_coord_lbl", color=(80, 220, 80))
            else:
                dpg.set_value("roi_coord_lbl", "未設定")
                dpg.configure_item("roi_coord_lbl", color=(150, 150, 150))

    def _handle_roi_drag(self) -> None:
        """在 render loop 裡處理 ROI 滑鼠拖移（僅 cam0 RGB 模式）。"""
        if not self._roi_mode:
            return
        if not dpg.does_item_exist("cam_image"):
            return

        hovered = dpg.is_item_hovered("cam_image")
        lmb_down = dpg.is_mouse_button_down(0)

        if hovered and lmb_down and not self._roi_dragging:
            # 開始拖移
            self._roi_dragging  = True
            self._roi_press_tex = self._mouse_to_tex()
            self._roi_drag_tex  = self._roi_press_tex

        elif self._roi_dragging and lmb_down:
            # 拖移中：更新預覽
            self._roi_drag_tex = self._mouse_to_tex()
            if self._roi_press_tex and self._roi_drag_tex:
                x1, y1 = self._roi_press_tex
                x2, y2 = self._roi_drag_tex
                self._yolo.set_preview_roi(0, x1, y1, x2, y2)

        elif self._roi_dragging and not lmb_down:
            # 放開：提交 ROI
            self._roi_dragging = False
            if self._roi_press_tex and self._roi_drag_tex:
                x1, y1 = self._roi_press_tex
                x2, y2 = self._roi_drag_tex
                if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
                    self._yolo.set_roi(0, x1, y1, x2, y2)
                    self._refresh_roi_label()
            self._yolo.clear_preview_roi(0)
            self._roi_press_tex = None
            self._roi_drag_tex  = None
            self._roi_mode = False
            dpg.configure_item("btn_roi_set", label="✏ 拖移設定 ROI")

    def _mouse_to_tex(self) -> 'tuple[int,int] | None':
        """
        將當前滑鼠位置轉換為 cam_image 的材質座標 (pixel_x, pixel_y)。
        考慮 cam_image 顯示尺寸 vs 材質尺寸的縮放。
        """
        if not dpg.does_item_exist("cam_image"):
            return None
        mx, my = dpg.get_mouse_pos()
        rect_min = dpg.get_item_rect_min("cam_image")
        rect_max = dpg.get_item_rect_max("cam_image")
        disp_w = rect_max[0] - rect_min[0]
        disp_h = rect_max[1] - rect_min[1]
        if disp_w <= 0 or disp_h <= 0:
            return None
        # 材質尺寸（cam0 的原始解析度）
        tex_w = self._tex_w
        tex_h = self._tex_h
        tx = int((mx - rect_min[0]) / disp_w * tex_w)
        ty = int((my - rect_min[1]) / disp_h * tex_h)
        tx = max(0, min(tex_w - 1, tx))
        ty = max(0, min(tex_h - 1, ty))
        return tx, ty

    # =========================================================================
    # 關節位置同步（/joint_states）
    # =========================================================================

    def _on_sync_joints(self, sender=None, app_data=None, user_data="L") -> None:
        self._arms[user_data].sync_once()

    def _on_toggle_joint_track(self, sender, app_data, user_data="L") -> None:
        self._arms[user_data].toggle_tracking(app_data)

    def _on_joint_slider(self, sender, app_data, user_data) -> None:
        jname = sender[len("jog_"):]
        rad = math.radians(float(app_data))
        if jname in _NEGATE_FOR_VIEWER:
            rad = -rad
        self._rv.set_angles({jname: rad})

    def _on_viewer_mouse_move(self, sender, app_data) -> None:
        mx, my = float(app_data[0]), float(app_data[1])
        dx = mx - self._last_mouse_pos[0]
        dy = my - self._last_mouse_pos[1]
        self._last_mouse_pos = [mx, my]

        if not dpg.is_item_hovered("robot_image"):
            return

        left  = dpg.is_mouse_button_down(0)
        right = dpg.is_mouse_button_down(1)
        changed = False

        if left and right:                   # 兩鍵同按 → 距離（前後移動）
            self._cam_dist = max(0.5, min(8.0, self._cam_dist - dy * 0.01))
            changed = True
        elif left:                           # 左鍵 → 方位角
            self._cam_az = max(-3.14, min(3.14, self._cam_az + dx * 0.01))
            changed = True
        elif right:                          # 右鍵 → 仰角
            self._cam_el = max(-1.5,  min(1.5,  self._cam_el - dy * 0.007))
            changed = True

        if changed:
            self._rv.set_camera(az=self._cam_az, el=self._cam_el, dist=self._cam_dist)

    # =========================================================================
    # 直線步進點動（Cartesian Step Jog）
    # =========================================================================

    _LIN_STEPS = {"10mm": 10.0, "1mm": 1.0, "0.1mm": 0.1}
    _ROT_STEPS  = {"10°":  10.0, "1°":  1.0, "0.1°":  0.1}
    _AXIS_IDX   = {"X": 0, "Y": 1, "Z": 2, "Rx": 0, "Ry": 1, "Rz": 2}
    _JOG_SPEED_MAX = 50.0    # mm/s at 100% for jog_linear
    _JOG_ROT_MAX   = 20.0    # deg/s at 100% for jog_linear rotation
    _JOG_HZ        = 20      # required publish rate for jog_linear

    def _on_cart_step(self, sender, app_data, user_data) -> None:
        axis, direction = user_data
        arm = "R" if dpg.get_value("cart_arm_select") == "右臂 (R)" else "L"
        if axis.startswith("R"):
            step = self._ROT_STEPS.get(dpg.get_value("cart_rot_step"), 1.0) * direction
        else:
            step = self._LIN_STEPS.get(dpg.get_value("cart_lin_step"), 1.0) * direction
        threading.Thread(target=self._execute_cart_step,
                         args=(arm, axis, step), daemon=True).start()

    def _execute_cart_step(self, arm: str, axis: str, delta: float) -> None:
        if self._cart_stepping:
            return
        self._cart_stepping = True
        try:
            import time as _time

            speed_pct = dpg.get_value("jog_speed") / 100.0
            is_rot    = axis.startswith("R")
            speed     = max(1.0, (self._JOG_ROT_MAX if is_rot
                                  else self._JOG_SPEED_MAX) * speed_pct)
            duration  = abs(delta) / speed          # 秒
            interval  = 1.0 / self._JOG_HZ          # 50ms

            idx = self._AXIS_IDX[axis]
            vel = [0.0, 0.0, 0.0]
            rot = [0.0, 0.0, 0.0]
            if is_rot:
                rot[idx] = speed if delta > 0 else -speed
            else:
                vel[idx] = speed if delta > 0 else -speed

            # 直接用 rclpy publisher（<1ms 延遲，比 subprocess 準確）
            domain_id = int(dpg.get_value(f"ros2_domain_{arm}"))
            node = get_ros2_node(domain_id)

            t_start = _time.perf_counter()
            while _time.perf_counter() - t_start < duration:
                t0 = _time.perf_counter()
                node.publish_jog(vel, rot)
                sleep_t = interval - (_time.perf_counter() - t0)
                if sleep_t > 0:
                    _time.sleep(sleep_t)

            # 停止：多送幾次確保手臂收到
            for _ in range(3):
                node.publish_stop()
                _time.sleep(0.01)

            unit = 'deg/s' if is_rot else 'mm/s'
            self._logger.log(
                f"[JOG] {arm} {axis}{delta:+.3f}  "
                f"@ {speed:.1f}{unit}  {duration*1000:.0f}ms"
            )
        except Exception as e:
            self._logger.log(f"[ERR] {arm} 步進失敗：{e}")
        finally:
            self._cart_stepping = False

    def _reset_texture(self) -> None:
        blank = np.zeros(self._tex_w * self._tex_h * 4, dtype=np.float32)
        blank[0::4] = 0.08
        blank[1::4] = 0.08
        blank[2::4] = 0.14
        blank[3::4] = 1.0
        dpg.set_value("cam_texture", blank)

    # =========================================================================
    # 通用回呼
    # =========================================================================

    def _append_calib_log(self, msg: str) -> None:
        dpg.set_value("calib_log", dpg.get_value("calib_log") + msg + "\n")

    def _clear_calib(self) -> None:
        dpg.set_value("calib_poses", "")
        dpg.set_value("calib_count", "0 筆")
        dpg.set_value("calib_result", "（尚未校正）")
        dpg.set_value("calib_log", "")

    # =========================================================================
    # ROS2 連線監控
    # =========================================================================

    def _on_ros2_start(self, sender=None, app_data=None, user_data="L") -> None:
        self._arms[user_data].start_monitoring()

    def _on_ros2_stop(self, sender=None, app_data=None, user_data="L") -> None:
        self._arms[user_data].stop_monitoring()

    def _apply_font_size(self) -> None:
        size = int(dpg.get_value("font_size"))
        tag  = f"font_{size}"
        if not dpg.does_item_exist(tag):
            with dpg.font(_FONT_PATH, size, tag=tag, parent="font_registry"):
                dpg.add_font_range_hint(dpg.mvFontRangeHint_Chinese_Full)
        dpg.bind_font(tag)

    def _clear_log(self) -> None:
        dpg.set_value("log_output", "")

    def _append_log(self, message: str) -> None:
        current = dpg.get_value("log_output")
        dpg.set_value("log_output", current + message + "\n")

    def _show_about(self) -> None:
        if dpg.does_item_exist("about_modal"):
            dpg.configure_item("about_modal", show=True)
            return
        with dpg.window(label="About", modal=True, tag="about_modal",
                        width=320, height=160, pos=[460, 300]):
            dpg.add_text("Kassow RobotUse")
            dpg.add_text("GUI tool for Kassow collaborative robots")
            dpg.add_text("with Intel RealSense camera support.")
            dpg.add_spacer(height=10)
            dpg.add_button(label="Close",
                           callback=lambda: dpg.configure_item("about_modal", show=False))
