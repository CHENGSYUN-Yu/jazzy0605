import pyrealsense2 as rs
import numpy as np
import logging
import threading
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    import cupy as cp
    _CUDA = True
    logger.info("CuPy 可用，影像轉換使用 CUDA 加速。")
except ImportError:
    cp = None
    _CUDA = False
    logger.info("CuPy 不可用，影像轉換使用 CPU (numpy)。")


# =============================================================================
# 公開資料結構
# =============================================================================

@dataclass
class DeviceInfo:
    serial:   str
    name:     str
    firmware: str
    index:    int


# =============================================================================
# 私有：單台相機
# =============================================================================

class _Camera:
    """管理單一 RealSense 相機的連線、背景擷取與中斷。"""

    def __init__(self, serial: str, width: int, height: int, fps: int):
        self.serial = serial
        self.width  = width
        self.height = height
        self.fps    = fps

        self._pipeline: rs.pipeline | None         = None
        self._config:   rs.config   | None         = None
        self._profile:  rs.pipeline_profile | None = None

        self._lock       = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self._color_arr: np.ndarray | None = None
        self._ir_arr:    np.ndarray | None = None
        self._depth_arr: np.ndarray | None = None  # float32, mm（已對齊至 color，已套 depth_scale）
        self._color_tex: np.ndarray | None = None
        self._ir_tex:    np.ndarray | None = None
        self._align          = None    # rs.align，connect() 後建立
        self._depth_scale_mm = 1.0     # mm per raw unit（從感測器讀取）

        # 相機內參（連線後從 pipeline profile 讀取）
        self.intrinsics: 'dict|None' = None   # {'fx','fy','cx','cy','w','h'}

        self.is_connected: bool = False

    def connect(self) -> bool:
        # 先嘗試 color + depth + IR；若失敗（D405 無 IR）再 fallback 到 color + depth
        for with_ir in (True, False):
            try:
                self._pipeline = rs.pipeline()
                self._config   = rs.config()
                self._config.enable_device(self.serial)
                self._config.enable_stream(
                    rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
                self._config.enable_stream(
                    rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
                if with_ir:
                    self._config.enable_stream(
                        rs.stream.infrared, 1, self.width, self.height,
                        rs.format.y8, self.fps)

                self._profile = self._pipeline.start(self._config)
                self._has_ir  = with_ir
                break   # 成功，跳出重試迴圈

            except Exception as e:
                self._cleanup()
                if not with_ir:
                    # 兩種設定都失敗
                    logger.error(f"[{self.serial}] 連線失敗：{e}")
                    return False
                logger.warning(f"[{self.serial}] 含 IR 連線失敗，改為 color+depth only：{e}")
                import time as _t; _t.sleep(0.5)   # 等設備完全釋放
                continue

        self._stop_event.clear()

        # 建立 depth-to-color 對齊器
        self._align = rs.align(rs.stream.color)

        # 從感測器讀取 depth_scale（每個 raw unit 對應幾 mm）
        try:
            depth_sensor = self._profile.get_device().first_depth_sensor()
            self._depth_scale_mm = depth_sensor.get_depth_scale() * 1000.0
            logger.info(f"[{self.serial}] depth_scale={self._depth_scale_mm:.4f} mm/unit")
        except Exception as e:
            logger.warning(f"[{self.serial}] 讀取 depth_scale 失敗，使用預設 1.0 mm/unit：{e}")

        # 讀取彩色相機內參
        try:
            color_stream = self._profile.get_stream(rs.stream.color)
            intr = color_stream.as_video_stream_profile().get_intrinsics()
            self.intrinsics = {
                'fx': intr.fx, 'fy': intr.fy,
                'cx': intr.ppx, 'cy': intr.ppy,
                'w':  intr.width, 'h': intr.height,
            }
            logger.info(f"[{self.serial}] 內參: fx={intr.fx:.1f} fy={intr.fy:.1f} "
                        f"cx={intr.ppx:.1f} cy={intr.ppy:.1f}")
        except Exception as e:
            logger.warning(f"[{self.serial}] 讀取內參失敗：{e}")

        self._thread = threading.Thread(
            target=self._capture_loop, daemon=True,
            name=f"rs_{self.serial[:8]}")
        self._thread.start()

        self.is_connected = True
        ir_txt = "with IR" if self._has_ir else "no IR"
        logger.info(f"[{self.serial}] 連線成功（{self.width}×{self.height}@{self.fps}fps, {ir_txt}）")
        return True

    def disconnect(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        self._cleanup()
        logger.info(f"[{self.serial}] 已中斷連線。")

    def get_color(self) -> np.ndarray | None:
        with self._lock:
            return self._color_arr.copy() if self._color_arr is not None else None

    def get_ir(self) -> np.ndarray | None:
        with self._lock:
            return self._ir_arr.copy() if self._ir_arr is not None else None

    def get_depth(self) -> 'np.ndarray | None':
        """回傳深度幀（uint16, 單位 mm）。"""
        with self._lock:
            return self._depth_arr.copy() if self._depth_arr is not None else None

    def get_texture(self, stream: str = "RGB") -> np.ndarray | None:
        with self._lock:
            return self._ir_tex if stream == "IR" else self._color_tex

    def _capture_loop(self) -> None:
        for _ in range(30):
            if self._stop_event.is_set():
                return
            try:
                self._pipeline.wait_for_frames(timeout_ms=500)
            except Exception:
                pass

        logger.info(f"[{self.serial}] 暖機完成，開始正式擷取。")

        while not self._stop_event.is_set():
            try:
                frames         = self._pipeline.wait_for_frames(timeout_ms=200)
                aligned_frames = self._align.process(frames)   # depth 對齊到 color
                color_frame    = aligned_frames.get_color_frame()
                depth_frame    = aligned_frames.get_depth_frame()
                ir_frame       = frames.get_infrared_frame(1) if getattr(self, '_has_ir', False) else None

                color_arr = np.asanyarray(color_frame.get_data()) if color_frame else None
                ir_arr    = np.asanyarray(ir_frame.get_data())    if ir_frame    else None
                # depth：套用 depth_scale 轉換為 mm（D405=×0.1, D435I=×1.0）
                if depth_frame:
                    depth_arr = np.asanyarray(depth_frame.get_data()).astype(np.float32) \
                                * self._depth_scale_mm
                else:
                    depth_arr = None
                color_tex = RealSense._to_texture(color_arr, is_ir=False) if color_arr is not None else None
                ir_tex    = RealSense._to_texture(ir_arr,    is_ir=True)  if ir_arr    is not None else None

                with self._lock:
                    self._color_arr = color_arr
                    self._ir_arr    = ir_arr
                    self._depth_arr = depth_arr
                    self._color_tex = color_tex
                    self._ir_tex    = ir_tex

            except RuntimeError:
                logger.warning(f"[{self.serial}] Pipeline 停止。")
                break
            except Exception as e:
                logger.debug(f"[{self.serial}] 擷取錯誤（已跳過）：{e}")

    def _cleanup(self) -> None:
        if self._pipeline:
            try:
                self._pipeline.stop()
            except Exception:
                pass
        self._pipeline = self._config = self._profile = None
        self.is_connected = False
        with self._lock:
            self._color_arr = self._ir_arr = self._depth_arr = None
            self._color_tex = self._ir_tex = None


# =============================================================================
# 公開：RealSense 主物件
# =============================================================================

class RealSense:
    """
    管理最多 3 台 RealSense 相機的單一入口點。

    用法：
        rs = RealSense(width=1280, height=720, fps=30)
        rs.connect()                      # 掃描並連接所有相機
        tex = rs.get_texture(0, "RGB")    # 取得 Cam 0 的材質
        frame = rs.get_frame(0)           # 取得 Cam 0 的原始 BGR 幀
        rs.disconnect()                   # 中斷所有連線
    """

    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30):
        self.width  = width
        self.height = height
        self.fps    = fps

        self._cameras: list[_Camera | None] = [None, None, None]
        self._devices: list[DeviceInfo]     = []

    # -------------------------------------------------------------------------
    # 主要操作
    # -------------------------------------------------------------------------

    def connect(self) -> int:
        """掃描裝置並連接所有找到的相機（最多 3 台）。回傳成功連線數。"""
        self.disconnect()
        self._scan()

        count = 0
        for i, info in enumerate(self._devices[:3]):
            cam = _Camera(info.serial, self.width, self.height, self.fps)
            self._cameras[i] = cam
            if cam.connect():
                count += 1
                logger.info(f"相機 [{i}] {info.name} 連線成功")
            else:
                logger.error(f"相機 [{i}] {info.name} 連線失敗")

        logger.info(f"共 {count} / {min(len(self._devices), 3)} 台相機連線成功。")
        return count

    def connect_one(self, index: int) -> bool:
        """連接指定索引的相機。若尚未掃描則先掃描。回傳是否成功。"""
        if not self._devices:
            self._scan()
        if index >= len(self._devices) or index >= 3:
            logger.warning(f"相機索引 {index} 超出範圍。")
            return False
        if self._cameras[index] is not None:
            self._cameras[index].disconnect()
        info = self._devices[index]
        cam  = _Camera(info.serial, self.width, self.height, self.fps)
        self._cameras[index] = cam
        result = cam.connect()
        level  = logger.info if result else logger.error
        level(f"相機 [{index}] {info.name} {'連線成功' if result else '連線失敗'}")
        return result

    def disconnect(self) -> None:
        """中斷所有相機連線。"""
        for i, cam in enumerate(self._cameras):
            if cam is not None and cam.is_connected:
                cam.disconnect()
            self._cameras[i] = None

    def disconnect_one(self, index: int) -> None:
        """中斷指定索引的相機連線。"""
        if 0 <= index < 3 and self._cameras[index] is not None:
            self._cameras[index].disconnect()
            self._cameras[index] = None

    # -------------------------------------------------------------------------
    # 資料存取
    # -------------------------------------------------------------------------

    def get_texture(self, cam_index: int, stream: str = "RGB") -> np.ndarray | None:
        """
        取得指定相機的材質資料（float32 RGBA 平坦陣列，可直接傳入 dpg.set_value）。

        Args:
            cam_index: 相機索引 0~2
            stream:    "RGB" 或 "IR"
        """
        cam = self._get(cam_index)
        return cam.get_texture(stream) if cam else None

    def get_frame(self, cam_index: int, stream: str = "RGB") -> np.ndarray | None:
        """
        取得指定相機的原始影像陣列（供視覺處理使用）。

        Args:
            cam_index: 相機索引 0~2
            stream:    "RGB" 回傳 BGR uint8；"IR" 回傳灰階 uint8
        """
        cam = self._get(cam_index)
        if cam is None:
            return None
        return cam.get_ir() if stream == "IR" else cam.get_color()

    def get_depth_frame(self, cam_index: int) -> 'np.ndarray | None':
        """取得深度幀（uint16, 單位 mm）。"""
        cam = self._get(cam_index)
        return cam.get_depth() if cam else None

    def get_intrinsics(self, cam_index: int) -> 'dict | None':
        """取得指定相機的彩色影像內參 {'fx','fy','cx','cy','w','h'}。"""
        cam = self._get(cam_index)
        return cam.intrinsics if cam else None

    # -------------------------------------------------------------------------
    # 屬性
    # -------------------------------------------------------------------------

    @property
    def devices(self) -> list[DeviceInfo]:
        """已掃描到的裝置資訊列表。"""
        return self._devices

    def is_connected(self, index: int) -> bool:
        """回傳指定相機是否已連線。"""
        if 0 <= index < 3:
            cam = self._cameras[index]
            return cam is not None and cam.is_connected
        return False

    @property
    def connected_count(self) -> int:
        """目前成功連線的相機數。"""
        return sum(1 for c in self._cameras if c is not None and c.is_connected)

    @property
    def camera_count(self) -> int:
        """相機物件總數（含連線失敗的）。"""
        return sum(1 for c in self._cameras if c is not None)

    # -------------------------------------------------------------------------
    # 私有
    # -------------------------------------------------------------------------

    def _get(self, index: int) -> _Camera | None:
        if 0 <= index < 3:
            cam = self._cameras[index]
            if cam is not None and cam.is_connected:
                return cam
        return None

    def _scan(self) -> None:
        ctx = rs.context()
        self._devices.clear()
        for i, dev in enumerate(ctx.query_devices()):
            try:
                self._devices.append(DeviceInfo(
                    serial   = dev.get_info(rs.camera_info.serial_number),
                    name     = dev.get_info(rs.camera_info.name),
                    firmware = dev.get_info(rs.camera_info.firmware_version),
                    index    = i,
                ))
                logger.info(f"發現裝置 [{i}]：{self._devices[-1].name}  SN: {self._devices[-1].serial}")
            except Exception as e:
                logger.warning(f"無法讀取第 {i} 台裝置資訊：{e}")

        if not self._devices:
            logger.warning("未偵測到任何 RealSense 裝置。")
            return

        # 固定順序：D435I（頭部）排 Cam 1，D405（手部）排 Cam 2
        # 依型號名稱排序：D435I < D405（字母序），但我們要 D435I 在前
        def _cam_priority(dev: DeviceInfo) -> int:
            name = dev.name.upper()
            if 'D435' in name:
                return 0   # 頭部相機 → Cam 1
            if 'D405' in name:
                return 1   # 手部相機 → Cam 2
            return 2       # 其他
        self._devices.sort(key=_cam_priority)
        for i, d in enumerate(self._devices):
            d.index = i
            logger.info(f"相機排序後 [{i}]：{d.name}  SN: {d.serial}")

    @staticmethod
    def _to_texture(frame: np.ndarray, is_ir: bool = False) -> np.ndarray:
        if _CUDA:
            return RealSense._to_texture_cuda(frame, is_ir)
        return RealSense._to_texture_cpu(frame, is_ir)

    @staticmethod
    def _to_texture_cuda(frame: np.ndarray, is_ir: bool) -> np.ndarray:
        h, w = frame.shape[:2]
        out  = cp.empty(h * w * 4, dtype=cp.float32)
        if is_ir:
            g = cp.asarray(frame).ravel().astype(cp.float32) * (1.0 / 255.0)
            out[0::4] = out[1::4] = out[2::4] = g
        else:
            bgr = cp.asarray(frame).reshape(-1, 3).astype(cp.float32) * (1.0 / 255.0)
            out[0::4] = bgr[:, 2]
            out[1::4] = bgr[:, 1]
            out[2::4] = bgr[:, 0]
        out[3::4] = 1.0
        return cp.asnumpy(out)

    @staticmethod
    def _to_texture_cpu(frame: np.ndarray, is_ir: bool) -> np.ndarray:
        h, w = frame.shape[:2]
        out  = np.empty(h * w * 4, dtype=np.float32)
        if is_ir:
            g = frame.ravel().astype(np.float32) * (1.0 / 255.0)
            out[0::4] = out[1::4] = out[2::4] = g
        else:
            bgr = frame.reshape(-1, 3).astype(np.float32) * (1.0 / 255.0)
            out[0::4] = bgr[:, 2]
            out[1::4] = bgr[:, 1]
            out[2::4] = bgr[:, 0]
        out[3::4] = 1.0
        return out
