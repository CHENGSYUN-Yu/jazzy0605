"""
auto_grasp.py — 自動夾取模組（DearPyGui 版）

架構：
  - AutoGrasp 持有狀態機 + 所有計算物件
  - 500Hz 控制迴圈跑在 daemon thread
  - build_ui() 建立 dpg items（在 app.py 的分頁內呼叫）
  - tick() 由主迴圈呼叫，把 _pending_ui 佇列刷新到 dpg

路徑常數：
  MODEL_PATH     : YOLO 模型
  T_MATRIX_PATH  : 頭部相機手眼校正矩陣
  EIH_T_PATH     : 手腕相機手眼校正矩陣
"""
import os
import queue
import threading
import time

import cv2
import numpy as np
import dearpygui.dearpygui as dpg

from src.trajectory_plan     import TrajectoryPlan
from src.execute_motion      import ExecuteMotion
from src.check_arrive        import CheckArrive
from src.which_first         import WhichFirst
from src.which_first_handcam import WhichFirstHandcam
from src.target_z_compute    import TargetZCompute
from src.target_consider_gripper import TargetConsiderGripper
from src.cam2flange          import Cam2Flange
from src.gripper_control     import GripperControl
from src.memory_instrument_point import MemoryInstrumentPoint
from src.put_site_get        import PutSiteGet
from src.place_sequence_targets import PlaceSequenceTargets
from src.return_home_targets import ReturnHomeTargets
from src.depth_reader        import DepthReader
from src.pixel2headcam       import Pixel2HeadCam
from src.headcam2base        import HeadCam2Base
from src.angle2rz            import Angle2Rz
from src.pose_offset         import PoseOffset
from src.handcam_angle2yaw   import HandcamAngle2Yaw
from src.yolo_engine         import YoloEngine
from src.yolo_detect_handcam import YoloDetectHandcam   # 備用
from src.get_depth_handcam   import GetDepthHandcam
from src.pixel2handcam       import Pixel2HandCam
from src.handcam2flange      import HandCam2Flange
from src.angle2rz_handcam    import Angle2RzHandcam
from src.flange2base         import Flange2Base
from src.ros2_node           import get_ros2_node

# ── 路徑 ──────────────────────────────────────────────────────────────────────
_BASE = os.path.join(os.path.dirname(__file__), '..')
MODEL_PATH    = os.path.join(_BASE, 'models', 'best20260603.pt')
T_MATRIX_PATH = os.path.join(_BASE, 'T_matrix_20260603_head30.npy')   # 頭部相機→base
EIH_T_PATH    = os.path.join(_BASE, 'T_cam2gripper_20260526_more14z.npy')  # 手部相機→法蘭

# ── 狀態標籤 ──────────────────────────────────────────────────────────────────
_PHASE_LABELS = {
    'idle':                   '● 待機',
    'detecting':              '🔍 偵測中...',
    'confirm_selection':      '⏸ 確認：偵測結果',
    'confirm_target':         '⏸ 確認：接近目標',
    'moving_approach':        '🚀 移動中（接近器械上方）',
    'confirm_arrived':        '⏸ 確認：已到達上方',
    'handcam_detecting':      '🔍 手腕相機偵測中...',
    'confirm_handcam':        '⏸ 確認：手腕相機結果',
    'confirm_grasp_target':   '⏸ 確認：夾取位姿',
    'moving_grasp':           '🚀 移動中（移向夾取位置）',
    'confirm_grasp_arrived':  '⏸ 確認：已到夾取位置',
    'closing_gripper':        '✊ 夾爪閉合中...',
    'confirm_gripper_closed': '⏸ 確認：夾取完成',
    'confirm_recording':      '⏸ 確認：放置目標',
    'moving_sequence':        '🚀 移動中（放置 / 回 Home）',
    'confirm_place_arrived':  '⏸ 確認：已到放置位置',
    'opening_gripper':        '✋ 夾爪張開中...',
    'complete':               '✅ 完成',
    'stopped':                '■ 已停止',
}

_PHASE_COLORS = {
    'idle':            (150, 150, 150),
    'detecting':       (255, 180, 0),
    'complete':        (80, 220, 80),
    'stopped':         (220, 80, 80),
    'closing_gripper': (255, 180, 0),
    'opening_gripper': (255, 180, 0),
}

_TABLE_Z_ABS_MM = 395.0   # 桌面距機器人基座的絕對距離（mm）
_Z_BUFFER_MM    = 2.0     # 安全緩衝（mm）


class AutoGrasp:
    """
    自動夾取狀態機 + DearPyGui UI。

    使用方式：
        ag = AutoGrasp(arm_ctrl=right_arm_controller, domain_id=1)
        ag.build_ui()          # 在 dpg tab 裡呼叫
        # 主迴圈每幀：
        ag.tick()
    """

    def __init__(self, arm_ctrl=None, domain_id: int = 1):
        self._arm        = arm_ctrl
        self._domain_id  = domain_id

        # ── YOLO 偵測物件（各自獨立）─────────────────────────────────────────
        self._head_detector = YoloEngine(MODEL_PATH, cam_id=0, fps=10.0)  # D435I 頭部
        self._hand_detector = YoloEngine(MODEL_PATH, cam_id=1, fps=10.0)  # D405 手部

        # ── 目標位姿硬性補償（套用在 Cam2Flange 之前）────────────────────────
        self._pose_offset = PoseOffset(dx=25.0)   # 手部相機接近點 x +25mm

        # ── 手腕相機 yaw 轉換（2D 傾角 → 法蘭目標 yaw）─────────────────────
        self._handcam_a2y   = HandcamAngle2Yaw(offset_deg=-135.0)

        # ── 手腕相機物件鏈 ────────────────────────────────────────────────────
        self._handcam_det   = YoloDetectHandcam(model_path=MODEL_PATH)
        self._handcam_depth = GetDepthHandcam()
        self._p2h           = Pixel2HandCam()
        self._h2f           = HandCam2Flange()
        self._a2rz_h        = Angle2RzHandcam()
        self._f2b           = Flange2Base()
        self._which_first_h = WhichFirstHandcam()
        self._handcam_selected = None
        self._handcam_dets: list = []
        self._last_handcam_t = 0.0
        try:
            self._h2f.load_T(EIH_T_PATH)
            self._a2rz_h.load_T(EIH_T_PATH)
            print(f'[AutoGrasp] EIH T_matrix 載入成功（手腕相機鏈）')
        except Exception as e:
            print(f'[AutoGrasp] EIH T_matrix 載入失敗：{e}')

        # ── 頭部相機座標轉換鏈 ────────────────────────────────────────────────
        self._depth_reader = DepthReader()
        self._p2c          = Pixel2HeadCam()
        self._h2b          = HeadCam2Base()
        self._a2rz         = Angle2Rz()
        try:
            self._h2b.load_T(T_MATRIX_PATH)
            self._a2rz.load_T(T_MATRIX_PATH)
            print(f'[AutoGrasp] T_matrix 載入成功：{T_MATRIX_PATH}')
        except Exception as e:
            print(f'[AutoGrasp] T_matrix 載入失敗：{e}')

        # ── 計算物件 ──────────────────────────────────────────────────────────
        self._which_first   = WhichFirst()
        self._tz_compute    = TargetZCompute(z_offset_mm=300.0)
        self._c2f           = Cam2Flange()
        self._tcg           = TargetConsiderGripper(gripper_length_mm=77.0)
        self._mem           = MemoryInstrumentPoint()
        self._put_site      = PutSiteGet()
        self._place_seq     = PlaceSequenceTargets(lift_z_mm=200.0)
        self._home_seq      = ReturnHomeTargets(lift_z_mm=200.0)
        self._traj_plan     = TrajectoryPlan()
        self._exec_motion   = ExecuteMotion()
        self._check_arrive  = CheckArrive(stable_duration_s=0.3)

        try:
            self._c2f.load_T(EIH_T_PATH)
        except Exception:
            pass

        # ── 夾爪（服務呼叫在連線後設定）──────────────────────────────────────
        self._gripper = GripperControl(service_callback=None)

        # ── 狀態機 ────────────────────────────────────────────────────────────
        self._phase            = 'idle'
        self._auto_selected    = None
        self._auto_target      = None
        self._auto_put_site    = None
        self._auto_t0          = 0.0
        self._move_queue:     list = []
        self._move_on_arrived     = None
        self._sequence_done_cb    = None   # 序列全部完成後的 callback（獨立於每步 _move_on_arrived）
        self._current_dets:list = []

        # ── 手腕相機（Phase 2，暫存 stub）─────────────────────────────────────
        self._handcam_selected = None

        # ── 500Hz 控制執行緒 ──────────────────────────────────────────────────
        self._motion_thread: threading.Thread | None = None
        self._motion_running = False

        # ── UI 更新佇列（background thread → main thread）────────────────────
        self._ui_queue: queue.SimpleQueue = queue.SimpleQueue()

        # ── dpg item tags ─────────────────────────────────────────────────────
        self._tag_phase  = 'ag_phase_label'
        self._tag_log    = 'ag_log_text'
        self._tag_confirm = 'ag_confirm_btn'
        self._tag_start   = 'ag_start_btn'
        self._tag_stop    = 'ag_stop_btn'

        self._last_infer_t = 0.0
        self._rs = None
        self._yaw_offset_deg = -45.0
        self._use_move_linear = True    # True = MoveLinear 服務（預設）；False = jog P-control

    @property
    def _grasp_z_limit(self) -> float:
        """夾取 Z 安全下限（動態，隨夾爪補償長度變化）。"""
        return -(_TABLE_Z_ABS_MM + self._tcg.gripper_length_mm + _Z_BUFFER_MM)

    def _on_gripper_length_change(self, sender, app_data) -> None:
        self._tcg.set_gripper_length(float(app_data))

    def set_realsense(self, rs) -> None:
        """由 app 注入 RealSense 物件，同時啟動兩個 YOLO 偵測器。"""
        self._rs = rs
        self._head_detector.start(rs)
        self._hand_detector.start(rs)

    def _on_conf_thresh_cam0(self, sender, app_data) -> None:
        self._head_detector.set_conf_threshold(float(app_data))

    def _on_conf_thresh_cam1(self, sender, app_data) -> None:
        self._hand_detector.set_conf_threshold(float(app_data))

    def _on_yaw_offset_change(self, sender, app_data) -> None:
        self._yaw_offset_deg = float(app_data)

    def _on_handcam_yaw_offset_change(self, sender, app_data) -> None:
        self._handcam_a2y.set_offset(float(app_data))

    # ═════════════════════════════════════════════════════════════════════════
    # YOLO 結果讀取（從 YoloEngine）
    # ═════════════════════════════════════════════════════════════════════════

    def _get_cam0_dets(self) -> list:
        """讀取頭部相機（D435I）最新偵測結果。"""
        return self._head_detector.get_dets()

    def _try_inject_detection(self) -> None:
        """
        若在偵測階段且 cam0 有結果，執行完整座標轉換鏈後進入 confirm_selection。
        轉換鏈：pixel → headcam 3D → base frame → Rz/yaw
        tick() 每幀呼叫，內部用 _last_inject_t 限速避免重複觸發。
        """
        if self._phase != 'detecting':
            return
        if self._rs is None:
            return

        # 限速：每 0.3 秒最多嘗試一次
        now = time.monotonic()
        if now - self._last_infer_t < 0.3:
            return
        self._last_infer_t = now

        dets = self._get_cam0_dets()
        if not dets:
            return

        # ── Step 1：更新相機內參（每次都從 SDK 讀，保證最新）────────────────
        intr = self._rs.get_intrinsics(0)
        if intr is None:
            return
        self._p2c.set_intrinsics(
            intr['fx'], intr['fy'], intr['cx'], intr['cy'])

        # ── Step 2：讀取深度幀（uint16 mm）──────────────────────────────────
        depth_frame = self._rs.get_depth_frame(0)
        if depth_frame is None:
            return

        # ── Step 3：為每個偵測結果加上深度值（5×5 patch 中位數）────────────
        for det in dets:
            det['depth_mm'] = self._depth_reader.get_depth(
                det['center'], depth_frame)

        # ── Step 4：pixel + depth → headcam 座標系 (mm)─────────────────────
        dets = self._p2c.project_all(dets)

        # ── Step 5：headcam → base frame（T_matrix 齊次轉換）────────────────
        dets = self._h2b.transform_all(dets)

        # ── Step 6：2D 傾角 → base frame Rz/yaw ────────────────────────────
        dets = self._a2rz.convert_all(dets)

        # ── Step 7：套用 yaw offset，並選與當前 TCP 最近的 180° 對稱解 ────
        current_yaw = (self._arm.current_rot[2]
                       if self._arm and self._arm.current_rot else 0.0)
        for det in dets:
            if det.get('yaw_deg') is None:
                continue
            raw = det['yaw_deg'] + self._yaw_offset_deg
            # 正規化到 (-180, 180]
            raw = (raw + 180.0) % 360.0 - 180.0
            # 夾爪 180° 對稱：選擇與當前 TCP yaw 差距最小的解
            alt = raw + 180.0 if raw < 0 else raw - 180.0
            det['yaw_deg'] = raw if abs(raw - current_yaw) <= abs(alt - current_yaw) else alt

        # ── Step 8：過濾掉座標轉換失敗的偵測────────────────────────────────
        dets = [d for d in dets if d.get('pos_base_mm')]
        if not dets:
            return

        # ── Step 9：WhichFirst 選出最佳目標────────────────────────────────
        first = self._which_first.get_first(dets)
        if first is None:
            return

        self._auto_selected = first
        pos = first['pos_base_mm']
        raw_angle = first.get('angle_deg', 0.0)
        txt = (
            f'YOLO 偵測到目標（完整座標轉換）：\n'
            f'  conf       = {first["conf"]:.2f}\n'
            f'  depth      = {first.get("depth_mm", 0):.0f} mm\n'
            f'  pos_base   = {[round(v, 1) for v in pos]}\n'
            f'  cam angle  = {round(raw_angle, 1)}°\n'
            f'  yaw offset = {self._yaw_offset_deg:+.1f}°\n'
            f'  法蘭 Rz   = {round(first.get("yaw_deg", 0), 1)}°\n\n'
            f'按「確認繼續」計算接近目標'
        )
        self._set_phase('confirm_selection', txt)

    # ═════════════════════════════════════════════════════════════════════════
    # UI 建構
    # ═════════════════════════════════════════════════════════════════════════

    def build_ui(self) -> None:
        """在當前 dpg 父容器內建立自動夾取 UI。"""
        iw = -1  # input_text 全寬

        with dpg.group():
            dpg.add_text('自動夾取流程', color=(200, 200, 100))
            dpg.add_separator()
            dpg.add_spacer(height=4)

            # ── 狀態（可選取）──────────────────────────────────────────────────
            dpg.add_text('狀態：', color=(180, 180, 180))
            dpg.add_input_text(
                tag=self._tag_phase,
                default_value=_PHASE_LABELS['idle'],
                readonly=True, width=iw,
            )
            dpg.add_spacer(height=6)

            # ── 流程 log（可選取）─────────────────────────────────────────────
            dpg.add_text('流程記錄：', color=(180, 180, 180))
            dpg.add_input_text(
                tag=self._tag_log,
                default_value='',
                multiline=True, readonly=True,
                width=iw, height=200,
                hint='流程結果將顯示在此...',
            )
            dpg.add_spacer(height=6)

            # ── 按鈕列 ────────────────────────────────────────────────────────
            with dpg.group(horizontal=True):
                dpg.add_button(label='▶ 開始',
                               tag=self._tag_start,
                               callback=self._on_start, width=120)
                dpg.add_button(label='✔ 確認繼續',
                               tag=self._tag_confirm,
                               callback=self._on_confirm, width=130)
                dpg.add_button(label='■ 停止',
                               tag=self._tag_stop,
                               callback=self._on_stop, width=100)
            dpg.add_spacer(height=6)
            dpg.add_checkbox(
                label='改用 Jog P-control（取消勾選 = MoveLinear 預設）',
                tag='ag_use_move_linear',
                default_value=False,
                callback=lambda s, v: setattr(self, '_use_move_linear', not v),
            )
            dpg.add_spacer(height=4)
            dpg.add_text('MoveLinear 速度 (mm/s)：', color=(180, 180, 180))
            dpg.add_slider_float(
                tag='ag_move_speed',
                default_value=130.0,
                min_value=5.0, max_value=200.0,
                width=iw, format='%.0f mm/s',
            )
            dpg.add_spacer(height=10)
            dpg.add_separator()
            dpg.add_spacer(height=6)

            # ── TCP 位置（可選取）─────────────────────────────────────────────
            dpg.add_text('手臂位置 TCP (mm)：', color=(180, 180, 180))
            dpg.add_input_text(
                tag='ag_tcp_line',
                default_value='X: ---  Y: ---  Z: ---',
                readonly=True, width=iw,
            )
            dpg.add_text('手臂姿態 RPY (deg)：', color=(180, 180, 180))
            dpg.add_input_text(
                tag='ag_rot_line',
                default_value='R: ---  P: ---  Yaw: ---',
                readonly=True, width=iw,
            )
            dpg.add_spacer(height=6)
            dpg.add_separator()
            dpg.add_spacer(height=4)

            # ── 信心分門檻滑桿（兩台相機各自可調）──────────────────────────
            dpg.add_text('信心分門檻 — Cam 1 頭部相機：', color=(180, 180, 180))
            dpg.add_slider_float(
                tag='ag_conf_thresh_cam0',
                default_value=0.15,
                min_value=0.05, max_value=0.99,
                width=iw, format='%.2f',
                callback=self._on_conf_thresh_cam0,
            )
            dpg.add_spacer(height=4)
            dpg.add_text('信心分門檻 — Cam 2 手部相機：', color=(180, 180, 180))
            dpg.add_slider_float(
                tag='ag_conf_thresh_cam1',
                default_value=0.50,
                min_value=0.05, max_value=0.99,
                width=iw, format='%.2f',
                callback=self._on_conf_thresh_cam1,
            )
            dpg.add_spacer(height=6)

            # ── Yaw offset 滑桿（頭部相機）───────────────────────────────────
            dpg.add_text('Yaw Offset 頭部相機（2D角→法蘭 Rz）：',
                         color=(180, 180, 180))
            dpg.add_slider_float(
                tag='ag_yaw_offset',
                default_value=self._yaw_offset_deg,
                min_value=-180.0, max_value=180.0,
                width=iw, format='%.1f°',
                callback=self._on_yaw_offset_change,
            )
            dpg.add_spacer(height=4)

            # ── Yaw offset 滑桿（手腕相機）───────────────────────────────────
            dpg.add_text('Yaw Offset 手腕相機（2D角→法蘭 Rz）：',
                         color=(180, 180, 180))
            dpg.add_slider_float(
                tag='ag_handcam_yaw_offset',
                default_value=-135.0,
                min_value=-180.0, max_value=180.0,
                width=iw, format='%.1f°',
                callback=self._on_handcam_yaw_offset_change,
            )
            dpg.add_spacer(height=4)

            # ── 夾爪補償長度 ──────────────────────────────────────────────────
            dpg.add_text('夾爪補償長度（法蘭到指尖 mm）：', color=(180, 180, 180))
            dpg.add_slider_float(
                tag='ag_gripper_length',
                default_value=self._tcg.gripper_length_mm,
                min_value=50.0, max_value=150.0,
                width=iw, format='%.0f mm',
                callback=self._on_gripper_length_change,
            )
            dpg.add_spacer(height=4)

            # ── 夾取計數（可選取）─────────────────────────────────────────────
            dpg.add_text('夾取計數 / YOLO 狀態：', color=(180, 180, 180))
            dpg.add_input_text(
                tag='ag_status_line',
                default_value='已夾取: 0/3  |  YOLO: 載入中...',
                readonly=True, width=iw,
            )

        self._refresh_btn_state()

    # ═════════════════════════════════════════════════════════════════════════
    # Main-thread tick（每幀由 app 呼叫）
    # ═════════════════════════════════════════════════════════════════════════

    def tick(self) -> None:
        """
        刷新 UI：把 background thread 推入的更新套用到 dpg。
        必須在主渲染執行緒呼叫。
        """
        # 更新 TCP 顯示（從 arm_controller 讀）
        if self._arm is not None:
            pos = self._arm.current_pos
            rot = self._arm.current_rot
            if pos and dpg.does_item_exist('ag_tcp_line'):
                dpg.set_value('ag_tcp_line',
                              f'X: {pos[0]:.2f}  Y: {pos[1]:.2f}  Z: {pos[2]:.2f}')
            if rot and dpg.does_item_exist('ag_rot_line'):
                dpg.set_value('ag_rot_line',
                              f'R: {rot[0]:.2f}  P: {rot[1]:.2f}  Yaw: {rot[2]:.2f}')

        # 手腕相機偵測階段：限速 10fps
        if self._phase == 'handcam_detecting':
            now = time.monotonic()
            if now - self._last_handcam_t >= 0.1:
                self._last_handcam_t = now
                self._process_handcam()

        # YOLO + 夾取計數狀態
        if dpg.does_item_exist('ag_status_line'):
            count = self._mem.total_recorded if self._mem else 0
            h_err = self._head_detector.load_error
            h_ok  = self._head_detector.is_loaded
            a_err = self._hand_detector.load_error
            a_ok  = self._hand_detector.is_loaded
            if h_err or a_err:
                yolo_txt = f'YOLO: ❌ {(h_err or a_err)[:40]}'
            elif not (h_ok and a_ok):
                yolo_txt = 'YOLO: ⏳ 載入中...'
            else:
                n0 = self._head_detector.get_det_count()
                n1 = self._hand_detector.get_det_count()
                t0 = self._head_detector.conf_threshold
                t1 = self._hand_detector.conf_threshold
                yolo_txt = f'YOLO: ✅ 頭:{n0}[≥{t0:.2f}]  手:{n1}[≥{t1:.2f}]'
                self._try_inject_detection()
            dpg.set_value('ag_status_line',
                          f'已夾取: {count}/3  |  {yolo_txt}')

        # 處理 UI 更新佇列
        while not self._ui_queue.empty():
            try:
                fn = self._ui_queue.get_nowait()
                fn()
            except Exception:
                pass

    # ═════════════════════════════════════════════════════════════════════════
    # 狀態機核心
    # ═════════════════════════════════════════════════════════════════════════

    def _set_phase(self, phase: str, log_text: str = '') -> None:
        """切換狀態並排入 UI 更新（thread-safe）。"""
        self._phase = phase
        label    = _PHASE_LABELS.get(phase, phase)
        log_text = log_text

        def _update():
            if dpg.does_item_exist(self._tag_phase):
                dpg.set_value(self._tag_phase, label)
            if log_text and dpg.does_item_exist(self._tag_log):
                dpg.set_value(self._tag_log, log_text)
            self._refresh_btn_state()
        self._ui_queue.put(_update)

    def _log(self, text: str) -> None:
        """附加一行 log（thread-safe）。"""
        def _update():
            if dpg.does_item_exist(self._tag_log):
                cur = dpg.get_value(self._tag_log) or ''
                dpg.set_value(self._tag_log, (cur + '\n' + text).strip())
        self._ui_queue.put(_update)

    def _refresh_btn_state(self) -> None:
        can_start   = self._phase in ('idle', 'stopped', 'complete')
        can_confirm = self._phase.startswith('confirm_')
        if dpg.does_item_exist(self._tag_start):
            dpg.configure_item(self._tag_start,   enabled=can_start)
        if dpg.does_item_exist(self._tag_confirm):
            dpg.configure_item(self._tag_confirm, enabled=can_confirm)

    # ═════════════════════════════════════════════════════════════════════════
    # 按鈕 callbacks（主執行緒）
    # ═════════════════════════════════════════════════════════════════════════

    def _on_start(self) -> None:
        if dpg.does_item_exist(self._tag_log):
            dpg.set_value(self._tag_log, '')
        if hasattr(self._mem, 'reset'):
            self._mem.reset()
        self._log('[INFO] 自動夾取流程啟動')

        yolo_ready = (self._head_detector.is_loaded and self._hand_detector.is_loaded)
        rs_ready   = (self._rs is not None)

        if yolo_ready and rs_ready:
            # 真實偵測模式：tick() 會呼叫 _try_inject_detection() 觸發
            self._set_phase('detecting',
                '🔍 YOLO 偵測中，等待穩定目標...\n'
                '（偵測到目標後會自動進入確認階段）')
        else:
            # Fallback stub（YOLO 未就緒或相機未連線）
            reason = []
            if not yolo_ready:
                reason.append('YOLO 未就緒')
            if not rs_ready:
                reason.append('相機未連線')
            self._set_phase('detecting',
                f'⚠ Stub 模式（{", ".join(reason)}）\n等待手臂位置...')
            threading.Thread(target=self._stub_detecting, daemon=True).start()

    def _on_stop(self) -> None:
        self._motion_running = False
        self._move_queue.clear()
        self._move_on_arrived = None
        node = get_ros2_node(self._domain_id)
        node.publish_stop()
        self._traj_plan.reset()
        self._exec_motion.reset()
        self._check_arrive.reset()
        self._set_phase('stopped', '流程已停止')

    def _on_confirm(self) -> None:
        p = self._phase
        if p == 'confirm_selection':
            self._step_compute_target()
        elif p == 'confirm_target':
            self._step_start_moving('moving_approach', self._step_after_approach)
        elif p == 'confirm_arrived':
            self._step_start_handcam_detect()
        elif p == 'confirm_handcam':
            self._step_compute_grasp()
        elif p == 'confirm_grasp_target':
            self._step_start_moving('moving_grasp', self._step_after_grasp)
        elif p == 'confirm_grasp_arrived':
            self._step_close_gripper()
        elif p == 'confirm_gripper_closed':
            self._step_record_and_compute_place()
        elif p == 'confirm_recording':
            self._step_start_place_sequence()
        elif p == 'confirm_place_arrived':
            self._step_open_gripper()

    # ═════════════════════════════════════════════════════════════════════════
    # Phase 1 stub：手動提供目標（無相機）
    # ═════════════════════════════════════════════════════════════════════════

    def _stub_detecting(self) -> None:
        """
        Phase 1：沒有相機，使用手臂當前 TCP 位置作為偵測目標（測試用）。
        等待手臂有已知位置後，自動進入 confirm_selection。
        """
        for _ in range(50):   # 最多等 5 秒
            time.sleep(0.1)
            if self._arm and self._arm.current_pos:
                break

        pos = self._arm.current_pos if self._arm else None
        if pos is None:
            self._set_phase('stopped', '❌ 手臂位置未知，請先同步位置')
            return

        # 用當前 TCP 建立虛擬偵測結果
        self._auto_selected = {
            'pos_base_mm': list(pos),
            'yaw_deg':     self._arm.current_rot[2] if self._arm.current_rot else 0.0,
            'conf':        1.0,
            'center':      (0, 0),
            'priority_rank': 0,
        }
        txt = (
            f'[Phase 1 stub] 使用當前 TCP 作為偵測目標：\n'
            f'  pos = {[round(v,1) for v in pos]}\n'
            f'  yaw = {round(self._auto_selected["yaw_deg"],1)}°\n\n'
            f'按「確認繼續」計算接近目標'
        )
        self._set_phase('confirm_selection', txt)

    def inject_detection(self, instrument: dict) -> None:
        """
        由相機模組注入偵測結果（Phase 2 用）。
        instrument 需包含 pos_base_mm, yaw_deg 等欄位。
        """
        if self._phase != 'detecting':
            return
        self._auto_selected = instrument
        txt = (
            f'偵測結果注入：\n'
            f'  pos_base = {[round(v,1) for v in instrument.get("pos_base_mm", [])]}\n'
            f'  yaw = {round(instrument.get("yaw_deg", 0),1)}°\n\n'
            f'按「確認繼續」計算接近目標'
        )
        self._set_phase('confirm_selection', txt)

    # ═════════════════════════════════════════════════════════════════════════
    # 流程步驟
    # ═════════════════════════════════════════════════════════════════════════

    def _step_compute_target(self) -> None:
        # Step 1：TargetZCompute → 手部相機應到達的位置（器械上方 300mm，base frame）
        cam_target = self._tz_compute.compute(self._auto_selected)
        if cam_target is None:
            self._set_phase('stopped', '❌ TargetZCompute 失敗')
            return

        # Step 2：PoseOffset → 硬性補償相機目標位置（x+20mm）
        cam_target = self._pose_offset.apply(cam_target)

        # Step 3：Cam2Flange → 補償手部相機與法蘭的偏移
        # 確保手部相機（而非法蘭面）對準器械正上方
        if self._c2f.is_ready:
            flange_target = self._c2f.compute(cam_target)
            if flange_target is None:
                self._set_phase('stopped', '❌ Cam2Flange 計算失敗')
                return
        else:
            flange_target = cam_target   # T_matrix 未載入時直接用（不補償）

        self._auto_target = flange_target

        det_pos = self._auto_selected.get('pos_base_mm', [0, 0, 0])
        off = self._pose_offset.offset
        raw_cam = cam_target['source'] if 'source' in cam_target else cam_target
        txt = (
            f'接近目標（手部相機對準器械上方 {self._tz_compute._z_offset:.0f}mm）：\n'
            f'  器械 base  = [{det_pos[0]:.1f}, {det_pos[1]:.1f}, {det_pos[2]:.1f}]\n'
            f'  相機目標   = [{cam_target["x_mm"]:.1f}, {cam_target["y_mm"]:.1f}, {cam_target["z_mm"]:.1f}]'
            f'  (offset dx={off["dx"]:+.1f} dy={off["dy"]:+.1f} dz={off["dz"]:+.1f})\n'
            f'  法蘭目標   = [{flange_target["x_mm"]:.1f}, {flange_target["y_mm"]:.1f}, {flange_target["z_mm"]:.1f}]'
            f'{"" if self._c2f.is_ready else "  ⚠ T_cam2gripper 未載入"}\n'
            f'  yaw = {flange_target["yaw_deg"]:.1f}°\n\n'
            f'按「確認繼續」開始移動'
        )
        self._set_phase('confirm_target', txt)

    def _step_start_moving(self, phase: str, on_arrived_cb) -> None:
        pos = self._arm.current_pos if self._arm else None
        rot = self._arm.current_rot if self._arm else None
        if pos is None:
            self._set_phase('stopped', '❌ 手臂位置未知，無法規劃軌跡')
            return

        self._move_on_arrived = on_arrived_cb
        self._set_phase(phase)

        if self._use_move_linear:
            # ── MoveLinear 服務模式 ───────────────────────────────────────────
            target = self._auto_target
            speed  = float(dpg.get_value('ag_move_speed')) if dpg.does_item_exist('ag_move_speed') else 50.0
            self._log(
                f'MoveLinear → [{target["x_mm"]:.1f}, {target["y_mm"]:.1f}, '
                f'{target["z_mm"]:.1f}] yaw={target["yaw_deg"]:.1f}°  {speed:.0f}mm/s'
            )
            threading.Thread(
                target=self._run_move_linear,
                args=(target, speed, on_arrived_cb),
                daemon=True, name='move_linear'
            ).start()
        else:
            # ── jog P-control 模式 ───────────────────────────────────────────
            current_yaw = rot[2] if rot else 0.0
            start = {'x_mm': pos[0], 'y_mm': pos[1], 'z_mm': pos[2],
                     'yaw_deg': current_yaw}
            # 依階段決定目標 yaw
            if phase == 'moving_approach':
                target_yaw = current_yaw          # 前半段：不旋轉
            elif phase == 'moving_grasp':
                target_yaw = self._auto_target.get('yaw_deg', current_yaw)  # 手腕相機 Rz
            else:  # moving_sequence（放置 / 回 Home）
                target_yaw = -135.0
            target_for_plan = dict(self._auto_target)
            target_for_plan['yaw_deg'] = target_yaw
            info = self._traj_plan.plan(start, target_for_plan)
            self._exec_motion.start()
            self._check_arrive.reset()
            self._auto_t0 = time.monotonic()
            self._log(
                f'TrajectoryPlan: duration={info["duration_s"]:.2f}s  '
                f'dist={info["dist_mm"]:.0f}mm  dyaw={info["dyaw_deg"]:.1f}°'
            )
            self._start_motion_thread()

    def _run_move_linear(self, target: dict, speed: float, on_arrived_cb) -> None:
        """在 background thread 呼叫 MoveLinear 服務，完成後執行 callback。"""
        node = get_ros2_node(self._domain_id)

        # 切換到 AUTONOMOUS 模式（關閉 interactivity）
        self._log('切換到 AUTONOMOUS 模式...')
        if not node.call_set_interactivity(enable=False):
            self._set_phase('stopped', '❌ 無法切換到 AUTONOMOUS 模式\n請確認 ROS2 driver 連線正常')
            return

        pos = [target['x_mm'], target['y_mm'], target['z_mm']]
        # MoveLinear 的 rot 是絕對目標姿態（不是速度指令）
        # 必須帶入當前 R、P，只修改需要改變的 Yaw
        cur = list(self._arm.current_rot) if self._arm and self._arm.current_rot else [0.0, 0.0, 0.0]
        if self._phase == 'moving_approach':
            rot = cur                                       # 保持完整當前姿態，只移 XYZ
        elif self._phase == 'moving_grasp':
            rot = [cur[0], cur[1], target['yaw_deg']]      # 保持 R/P，設定夾取 Yaw
        else:  # moving_sequence
            rot = [cur[0], cur[1], -135.0]                 # 保持 R/P，固定放置 Yaw
        self._log(f'MoveLinear 發送中... pos={[round(v,1) for v in pos]} yaw={rot[2]:.1f}°')
        try:
            # sync = 當前 joint 4 角度，固定 null-space 避免持續旋轉
            joints = getattr(self._arm, 'current_joints', [])
            sync = float(joints[3]) if len(joints) >= 4 else 78.0
            ok = node.call_move_linear(pos, rot, speed_mm_s=speed, ref=0,
                                       sync=sync, timeout_sec=60.0)
        finally:
            # 無論成功或失敗都恢復 MANUAL 模式，確保安全
            node.call_set_interactivity(enable=True)
            self._log('已恢復 MANUAL 模式')

        if ok:
            self._log('✅ MoveLinear success')
            cb = self._move_on_arrived
            self._move_on_arrived = None
            if cb:
                cb()
        else:
            self._set_phase('stopped',
                '❌ MoveLinear 失敗\n'
                '可能原因：\n'
                '  1. 目標位置超出工作範圍\n'
                '  2. 路徑上有碰撞\n'
                '  3. 機器人處於 ALARM 或 SUSPENDED 狀態')

    def _step_start_sequence(self, targets: list, on_done_cb) -> None:
        self._move_queue       = list(targets)
        self._sequence_done_cb = on_done_cb   # 保存在獨立欄位，不被 _step_start_moving 覆蓋
        self._step_next_in_sequence()

    def _step_next_in_sequence(self) -> None:
        if not self._move_queue:
            cb = self._sequence_done_cb
            self._sequence_done_cb = None
            if cb:
                cb()
            return
        self._auto_target = self._move_queue.pop(0)
        self._step_start_moving('moving_sequence', self._step_next_in_sequence)

    def _step_after_approach(self) -> None:
        pos = self._arm.current_pos
        txt = (
            f'✅ 已到達器械上方\n'
            f'  x={pos[0]:.1f}  y={pos[1]:.1f}  z={pos[2]:.1f} mm\n\n'
            f'按「確認繼續」啟動手腕相機偵測'
        )
        self._set_phase('confirm_arrived', txt)

    # ── 手腕相機偵測 ──────────────────────────────────────────────────────────

    def _step_start_handcam_detect(self) -> None:
        """啟動手腕相機偵測階段（使用 YoloEngine cam1 已有的偵測結果）。"""
        self._handcam_selected = None
        self._handcam_dets = []
        self._last_handcam_t = 0.0
        self._set_phase('handcam_detecting', '🔍 手腕相機偵測中，等待穩定結果...')

    def _process_handcam(self) -> None:
        """
        手腕相機偵測處理，由 tick() 在 handcam_detecting 階段限速呼叫。
        使用 YoloEngine cam1 已有的偵測結果（避免重複推論）。
        """
        if self._rs is None or self._arm is None:
            return

        # 直接取手部相機偵測器的結果
        dets = self._hand_detector.get_dets()
        if not dets:
            self._log('[手腕相機] 未偵測到目標...')
            return

        # 只取 stable=True 的偵測
        stable_dets = [d for d in dets if d.get('stable')]
        if not stable_dets:
            self._log(f'[手腕相機] 偵測數: {len(dets)}，等待穩定中...')
            return

        # 取深度幀和內參
        depth = self._rs.get_depth_frame(1)
        intr  = self._rs.get_intrinsics(1)
        if depth is None or intr is None:
            self._log('[手腕相機] 深度圖或內參未就緒')
            return

        # 設定 D405 內參
        self._p2h.set_intrinsics(
            intr['fx'], intr['fy'], intr['cx'], intr['cy'])

        # 更新 Flange2Base 即時手臂位姿
        pos = self._arm.current_pos
        rot = self._arm.current_rot
        if pos and rot:
            self._f2b.update_arm_pose(pos, rot)

        # Flange2Base 需要手臂即時位姿，先確認就緒
        if not self._f2b.is_ready:
            self._log('[手腕相機] 等待手臂位姿更新...')
            return

        # 座標轉換鏈
        import copy
        dets = copy.deepcopy(stable_dets)
        for det in dets:
            det['depth_mm'] = self._handcam_depth.get_depth(det['center'], depth)
        print(f'[DBG] depth_mm={[d.get("depth_mm") for d in dets]}  p2h.is_ready={self._p2h.is_ready}  h2f.is_ready={self._h2f.is_ready}  f2b.is_ready={self._f2b.is_ready}')
        dets = self._p2h.project_all(dets)
        print(f'[DBG] after p2h: pos_cam_mm={[d.get("pos_cam_mm") for d in dets]}')
        dets = self._h2f.transform_all(dets)
        print(f'[DBG] after h2f: pos_flange_mm={[d.get("pos_flange_mm") for d in dets]}')
        dets = self._a2rz_h.convert_all(dets)
        dets = self._f2b.transform_all(dets)
        print(f'[DBG] after f2b: pos_base_mm={[d.get("pos_base_mm") for d in dets]}')
        dets = [d for d in dets if d.get('pos_base_mm')]

        # ── HandcamAngle2Yaw：2D 傾角直接轉法蘭 yaw（覆蓋 T_matrix 結果）──
        current_yaw = self._arm.current_rot[2] if self._arm and self._arm.current_rot else 0.0
        # 先儲存 T_matrix 計算的 yaw（用於診斷比對）
        for det in dets:
            det['yaw_base_tmatrix'] = det.get('yaw_base_deg')
        dets = self._handcam_a2y.convert_all(dets, current_yaw=current_yaw)
        # convert_all 輸出 yaw_deg，同步寫入 yaw_base_deg（TargetConsiderGripper 使用）
        for det in dets:
            det['yaw_base_deg'] = det.get('yaw_deg')

        self._handcam_dets = dets

        if not dets:
            self._log('[手腕相機] 座標轉換失敗')
            return

        first = self._which_first_h.get_first(dets)
        if first:
            self._handcam_selected = first
            pos_b      = first.get('pos_base_mm', [0, 0, 0])
            raw_angle  = first.get('angle_deg', 0.0)
            yaw_offset = self._handcam_a2y.offset_deg
            yaw_final  = first.get('yaw_base_deg') or 0.0
            yaw_tmat   = first.get('yaw_base_tmatrix')
            txt = (
                f'手腕相機偵測結果（YoloEngine cam1）：\n'
                f'  conf          = {first["conf"]:.2f}\n'
                f'  depth         = {first.get("depth_mm", 0):.0f} mm\n'
                f'  pos_base      = {[round(v, 1) for v in pos_b]}\n'
                f'\n'
                f'  ★ 2D 傾角（fitEllipse） = {round(raw_angle, 1)}°\n'
                f'  ★ HandcamAngle2Yaw     = -({round(raw_angle, 1)}°) + offset({yaw_offset:+.1f}°) → {round(yaw_final, 1)}°\n'
                f'  ★ T_matrix Rz          = {round(yaw_tmat, 1) if yaw_tmat is not None else "N/A"}°\n'
                f'\n'
                f'請手動對齊後記錄實際 Rz，比對上方 2D 傾角\n'
                f'按「確認繼續」計算夾取位姿'
            )
            self._set_phase('confirm_handcam', txt)

    def _step_compute_grasp(self) -> None:
        source = self._handcam_selected or self._auto_selected
        if not source or not source.get('pos_base_mm'):
            self._set_phase('stopped', '❌ 無有效器械位置')
            return
        grasp_target = self._tcg.compute(source)
        if grasp_target is None:
            self._set_phase('stopped', '❌ TargetConsiderGripper 失敗')
            return
        if grasp_target['z_mm'] < self._grasp_z_limit:
            self._set_phase('detecting',
                f'⚠ 目標 z={grasp_target["z_mm"]:.1f}mm 低於安全下限'
                f'（{GRASP_Z_LIMIT_MM}mm），返回偵測')
            return
        self._auto_target = grasp_target
        txt = (
            f'夾取目標（法蘭 + 夾爪補償）：\n'
            f'  x={grasp_target["x_mm"]:.1f}  '
            f'y={grasp_target["y_mm"]:.1f}  '
            f'z={grasp_target["z_mm"]:.1f} mm\n'
            f'  yaw={grasp_target["yaw_deg"]:.1f}°\n\n'
            f'按「確認繼續」開始移動到夾取位置'
        )
        self._set_phase('confirm_grasp_target', txt)

    def _step_after_grasp(self) -> None:
        pos = self._arm.current_pos
        txt = (
            f'✅ 已到達夾取位置\n'
            f'  x={pos[0]:.1f}  y={pos[1]:.1f}  z={pos[2]:.1f} mm\n\n'
            f'確認夾爪位置後按「確認繼續」（夾爪將閉合）'
        )
        self._set_phase('confirm_grasp_arrived', txt)

    def _step_close_gripper(self) -> None:
        self._setup_gripper()
        self._set_phase('closing_gripper', '夾爪閉合中，等待 1.5s...')
        def _do():
            ok = self._gripper.close()
            self._log(f'GripperControl.close(): {"✅ success" if ok else "❌ failed"}')
            time.sleep(1.5)
            self._set_phase('confirm_gripper_closed',
                            '✅ 夾爪已閉合，確認器械後按「確認繼續」')
        threading.Thread(target=_do, daemon=True).start()

    def _step_record_and_compute_place(self) -> None:
        slot  = self._mem.record(self._auto_selected)
        count = self._mem.total_recorded
        put_site = self._put_site.get(count)
        if put_site is None:
            self._set_phase('stopped', f'❌ 放置點超出範圍（count={count}）')
            return
        self._auto_put_site = put_site
        txt = (
            f'記錄器械 #{count}（slot {slot}）\n'
            f'放置目標：\n'
            f'  x={put_site["x_mm"]:.1f}  '
            f'y={put_site["y_mm"]:.1f}  '
            f'z={put_site["z_mm"]:.1f} mm\n'
            f'  yaw={put_site["yaw_deg"]:.1f}°\n\n'
            f'按「確認繼續」開始放置（3段移動）'
        )
        self._set_phase('confirm_recording', txt)
        # ag_status_line 由 tick() 自動更新，不需另外設定

    def _step_start_place_sequence(self) -> None:
        pos = self._arm.current_pos
        rot = self._arm.current_rot
        targets = self._place_seq.compute(pos, rot, self._auto_put_site)
        if targets is None:
            self._set_phase('stopped', '❌ PlaceSequenceTargets 失敗')
            return
        self._log('開始放置序列：①抬升 → ②橫移 → ③下降')
        self._step_start_sequence(
            [targets['lift'], targets['approach'], targets['place']],
            self._step_after_place)

    def _step_after_place(self) -> None:
        pos = self._arm.current_pos
        txt = (
            f'✅ 已到達放置位置\n'
            f'  x={pos[0]:.1f}  y={pos[1]:.1f}  z={pos[2]:.1f} mm\n\n'
            f'確認器械放置正確後按「確認繼續」（夾爪將張開）'
        )
        self._set_phase('confirm_place_arrived', txt)

    def _step_open_gripper(self) -> None:
        self._set_phase('opening_gripper', '夾爪張開中，等待 1.5s...')
        def _do():
            ok = self._gripper.open()
            self._log(f'GripperControl.open(): {"✅ success" if ok else "❌ failed"}')
            time.sleep(1.5)
            self._step_after_open()
        threading.Thread(target=_do, daemon=True).start()

    def _step_after_open(self) -> None:
        pos = self._arm.current_pos
        rot = self._arm.current_rot
        targets = self._home_seq.compute(pos, rot)
        self._log('開始回 Home：①抬升 → ②Home')
        self._step_start_sequence(
            [targets['lift'], targets['home']],
            self._step_after_home)

    def _step_after_home(self) -> None:
        count = self._mem.total_recorded
        if count >= 3:
            self._set_phase('complete', f'✅ 已完成 {count} 個器械夾取！')
        else:
            self._log(f'已完成 {count}/3，回到偵測繼續下一個')
            yolo_ready = (self._head_detector.is_loaded and
                          self._hand_detector.is_loaded)
            rs_ready   = (self._rs is not None)
            if yolo_ready and rs_ready:
                self._set_phase('detecting',
                    '🔍 YOLO 偵測中，等待穩定目標...\n'
                    '（偵測到目標後會自動進入確認階段）')
                # tick() 會呼叫 _try_inject_detection() 觸發
            else:
                reason = []
                if not yolo_ready: reason.append('YOLO 未就緒')
                if not rs_ready:   reason.append('相機未連線')
                threading.Thread(target=self._stub_detecting, daemon=True).start()
                self._set_phase('detecting',
                    f'⚠ Stub 模式（{", ".join(reason)}）\n等待手臂位置...')

    def _setup_gripper(self) -> None:
        node = get_ros2_node(self._domain_id)
        self._gripper = GripperControl(
            service_callback=lambda idx, val: node.call_gripper_io(idx, val)
        )

    # ═════════════════════════════════════════════════════════════════════════
    # 500Hz 控制執行緒
    # ═════════════════════════════════════════════════════════════════════════

    def _start_motion_thread(self) -> None:
        if self._motion_running:
            return
        self._motion_running = True
        self._motion_thread = threading.Thread(
            target=self._motion_loop, daemon=True, name='auto_grasp_500hz')
        self._motion_thread.start()

    def _motion_loop(self) -> None:
        """500Hz P-control 迴圈，直到 ExecMotion done + CheckArrive 確認到位。"""
        node      = get_ros2_node(self._domain_id)
        interval  = 1.0 / 500.0

        while self._motion_running:
            t0 = time.perf_counter()

            if self._phase not in ('moving_approach', 'moving_grasp', 'moving_sequence'):
                break

            pos = self._arm.current_pos if self._arm else None
            rot = self._arm.current_rot if self._arm else None
            if pos is None:
                time.sleep(interval)
                continue

            t   = time.monotonic() - self._auto_t0
            cmd = self._exec_motion.compute(t, pos, rot, self._traj_plan)
            # moving_approach：不旋轉；moving_grasp / moving_sequence：使用計算出的 Rz
            rot_cmd = [0.0, 0.0, 0.0] if self._phase == 'moving_approach' else cmd['rot']
            node.publish_jog(cmd['vel'], rot_cmd)

            if cmd['done']:
                arrived = self._check_arrive.update(pos, rot, time.monotonic())
                if arrived:
                    node.publish_stop()
                    self._motion_running = False
                    # 呼叫抵達 callback
                    cb = self._move_on_arrived
                    self._move_on_arrived = None
                    if cb:
                        cb()
                    break

            elapsed = time.perf_counter() - t0
            remaining = interval - elapsed
            if remaining > 0:
                time.sleep(remaining)

        self._motion_running = False

    # ═════════════════════════════════════════════════════════════════════════
    # 清理
    # ═════════════════════════════════════════════════════════════════════════

    def cleanup(self) -> None:
        self._motion_running = False
        self._head_detector.stop()
        self._hand_detector.stop()
        node = get_ros2_node(self._domain_id)
        node.publish_stop()
