"""
yolo_detector_pt.py — YOLO 偵測引擎（PyTorch .pt 版，獨立工廠）

職責：
    使用 Ultralytics YOLO 載入 .pt 模型並執行推論，
    功能與 YoloDetector（ONNX 版）完全相同，但底層推論引擎不同。

    YoloDetector     → ONNX Runtime（需先匯出 .onnx）
    YoloDetector_pt  → Ultralytics PyTorch（直接讀 .pt）

外部介面與 YoloDetector 完全一致，可互換使用：
    detector = YoloDetector_pt('/path/to/model.pt')
    detector.load_model()
    dets = detector.infer(frame_bgr)
    best = detector.get_tracked()
"""

import copy
import json
import math
import os
import queue
import subprocess
import sys
import threading
from multiprocessing.shared_memory import SharedMemory

import cv2
import numpy as np

from .roi_filter import RoiFilter

# ── 追蹤參數（與 YoloDetector 一致）────────────────────────────────────────
_DEFAULT_CONF  = 0.30
_DEFAULT_IMGSZ = 640
_EMA_ALPHA        = 0.12
_CONFIRM_FRAMES   = 3
_GRACE_FRAMES     = 3
_IOU_TRACK_THRESH = 0.25
_RELOCK_FRAMES    = 5


# ── 模組級純函式（與 YoloDetector 一致）────────────────────────────────────

def _iou(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter + 1e-8)


def _mask_edge_orientation(mask: np.ndarray):
    contours, _ = cv2.findContours(mask.astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, None, None, None
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 5:
        return None, None, None, None
    (cx, cy), (w, h), rect_deg = cv2.minAreaRect(contour)
    if h >= w:
        long_deg = rect_deg + 90.0; half_len = h / 2.0
    else:
        long_deg = rect_deg;        half_len = w / 2.0
    while long_deg >  90.0: long_deg -= 180.0
    while long_deg <= -90.0: long_deg += 180.0
    axis_vec = np.array([math.cos(math.radians(long_deg)),
                         math.sin(math.radians(long_deg))], dtype=np.float32)
    return (float(cx), float(cy)), axis_vec, float(half_len), float(long_deg)


# ══════════════════════════════════════════════════════════════════════════════

class YoloDetector_pt:
    """
    YOLO 偵測引擎（Ultralytics PyTorch .pt 版）。

    直接載入 .pt 模型，無需匯出 ONNX。
    包含完整的跨幀追蹤、EMA 平滑、穩定判斷邏輯。
    外部介面與 YoloDetector 完全一致。
    """

    def __init__(self,
                 model_path: str = None,
                 conf_thresh: float = _DEFAULT_CONF,
                 imgsz:       int   = _DEFAULT_IMGSZ):
        self._model_path  = model_path
        self._conf_thresh = conf_thresh
        self._imgsz       = imgsz
        self._roi         = RoiFilter()
        self._results_lock = threading.Lock()

        self._device    = 'cpu'
        self._proc      = None   # Python 3.11 inference subprocess
        self._shm       = None   # 共享記憶體（frame buffer）
        self._frame_buf = None
        self._result_queue: queue.Queue = queue.Queue()
        self._server_hw = (640, 640)

        # 追蹤狀態
        self._tracked_box     = None
        self._smooth_centroid = None
        self._smooth_angle    = None
        self._consec_det      = 0
        self._consec_nodet    = 0
        self._stable          = False
        self._track_iou_miss  = 0
        self._results         = []

    # ── 模型載入 ──────────────────────────────────────────────────────────────

    def load_model(self, force_cpu: bool = False) -> bool:
        """
        啟動 Python 3.11 GPU 推論 subprocess（yolo_infer_server.py）。
        force_cpu 參數保留介面相容，子程序仍會自動選 GPU/CPU。
        """
        try:
            srv_h = srv_w = self._imgsz
            self._server_hw = (srv_h, srv_w)

            # 建立共享記憶體（frame buffer）
            self._shm = SharedMemory(create=True, size=srv_h * srv_w * 3)
            self._frame_buf = np.ndarray(
                (srv_h, srv_w, 3), dtype=np.uint8, buffer=self._shm.buf)

            # 找 server 腳本（同目錄）
            server_script = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'yolo_infer_server.py')

            _py = sys.executable   # 使用目前 venv 的 Python（3.12）
            self._proc = subprocess.Popen(
                [_py, server_script,
                 self._model_path, self._shm.name,
                 str(srv_h), str(srv_w), '3'],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True, bufsize=1,
            )

            # 背景 stderr 印出（暖身 log）
            def _log_stderr():
                for line in self._proc.stderr:
                    print(f'[infer_server] {line.rstrip()}')
            threading.Thread(target=_log_stderr, daemon=True).start()

            # 背景 stdout 讀取 → queue
            self._result_queue = queue.Queue()
            def _read_stdout():
                try:
                    for line in self._proc.stdout:
                        self._result_queue.put(line.rstrip('\n'))
                except Exception:
                    pass
            threading.Thread(target=_read_stdout, daemon=True).start()

            # 等待 READY（含暖身，Blackwell PTX JIT 首次可能需 30s+）
            print(f'[YoloDetector_pt] 等待 server 就緒（含 PTX JIT 暖身）...')
            try:
                ready = self._result_queue.get(timeout=120.0)
            except queue.Empty:
                print('[YoloDetector_pt] server 啟動逾時')
                return False

            if not ready.startswith('READY'):
                print(f'[YoloDetector_pt] 非預期回應: {ready}')
                return False

            parts = dict(p.split('=') for p in ready.split()[1:] if '=' in p)
            self._device = parts.get('device', 'cpu')
            print(f'[YoloDetector_pt] server ready | '
                  f'model={os.path.basename(self._model_path)} | '
                  f'device={self._device} | cuda={parts.get("cuda", "N/A")}')
            return True

        except Exception as e:
            print(f'[YoloDetector_pt] load_model failed: {e}')
            return False

    def __del__(self):
        try:
            if self._proc and self._proc.poll() is None:
                self._proc.stdin.write('STOP\n')
                self._proc.stdin.flush()
                self._proc.wait(timeout=3.0)
        except Exception:
            pass
        try:
            if self._shm:
                self._shm.close()
                self._shm.unlink()
        except Exception:
            pass

    # ── 追蹤狀態重置 ──────────────────────────────────────────────────────────

    def reset_tracking(self):
        self._tracked_box     = None
        self._smooth_centroid = None
        self._smooth_angle    = None
        self._consec_det      = 0
        self._consec_nodet    = 0
        self._stable          = False
        self._track_iou_miss  = 0

    # ── 推論（含追蹤）────────────────────────────────────────────────────────

    def infer(self, frame_bgr: np.ndarray) -> list:
        """
        對單幀 BGR 影像執行推論，維護跨幀追蹤狀態。

        回傳：偵測結果 list（深拷貝），每個元素：
            'center'    : (cx, cy)
            'box'       : (x1, y1, x2, y2)
            'conf'      : float
            'cls_id'    : int
            'mask'      : np.ndarray (HxW bool)
            'angle_deg' : float
            'tracked'   : bool
            'stable'    : bool
        """
        if self._proc is None or self._proc.poll() is not None:
            return []

        H, W = frame_bgr.shape[:2]
        srv_h, srv_w = self._server_hw

        # 縮放至 server 接受大小（若不同）
        if H != srv_h or W != srv_w:
            frame_send = cv2.resize(frame_bgr, (srv_w, srv_h))
            scale_x, scale_y = W / srv_w, H / srv_h
        else:
            frame_send = frame_bgr
            scale_x = scale_y = 1.0

        # 寫入共享記憶體
        np.copyto(self._frame_buf, frame_send)

        # 發送推論指令
        try:
            self._proc.stdin.write(f'INFER {self._conf_thresh:.3f}\n')
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError):
            return []

        # 等待結果（timeout 5s）
        try:
            raw_json = self._result_queue.get(timeout=5.0)
        except queue.Empty:
            return []

        try:
            server_dets = json.loads(raw_json)
        except json.JSONDecodeError:
            return []

        # ── ROI 過濾 + 座標換算回原始幀 ──────────────────────────────────────
        raw_dets = []
        for d in server_dets:
            x1 = d['box'][0] * scale_x
            y1 = d['box'][1] * scale_y
            x2 = d['box'][2] * scale_x
            y2 = d['box'][3] * scale_y
            cx = d['center'][0] * scale_x
            cy = d['center'][1] * scale_y

            if not self._roi.contains(cx, cy):
                continue

            raw_dets.append({
                'center':    (cx, cy),
                'box':       (x1, y1, x2, y2),
                'conf':      float(d['conf']),
                'cls_id':    int(d['cls_id']),
                'mask':      None,             # server 端處理，此層不需要
                'angle_deg': float(d['angle_deg']),
            })

        # ── 無偵測：grace period ──────────────────────────────────────────────
        if len(raw_dets) == 0:
            self._consec_det    = 0
            self._consec_nodet += 1
            if self._consec_nodet >= _GRACE_FRAMES:
                self.reset_tracking()
                with self._results_lock:
                    self._results = []
                return []
            with self._results_lock:
                kept = [dict(r, stable=False) for r in self._results]
                self._results = kept
            return copy.deepcopy(kept)

        # ── 有偵測：IoU 追蹤 ──────────────────────────────────────────────────
        self._consec_nodet  = 0
        self._consec_det   += 1
        confs_arr = np.array([d['conf'] for d in raw_dets])
        boxes_all = [d['box'] for d in raw_dets]

        best_idx = self._update_tracking(confs_arr, boxes_all)

        if self._consec_det >= _CONFIRM_FRAMES:
            self._stable = True

        # EMA bbox
        raw_box = list(boxes_all[best_idx])
        if self._tracked_box is None or not self._stable:
            self._tracked_box = raw_box[:]
        else:
            a = _EMA_ALPHA
            self._tracked_box = [a*r + (1-a)*s
                                 for r, s in zip(raw_box, self._tracked_box)]

        # EMA centroid / angle（直接用 server 回傳的預計算值）
        centroid  = raw_dets[best_idx]['center']
        angle_deg = raw_dets[best_idx]['angle_deg']
        if centroid is not None:
            cx, cy = centroid
            a = _EMA_ALPHA
            if self._smooth_centroid is None or not self._stable:
                self._smooth_centroid = [cx, cy]
                self._smooth_angle    = angle_deg
            else:
                self._smooth_centroid[0] = a*cx + (1-a)*self._smooth_centroid[0]
                self._smooth_centroid[1] = a*cy + (1-a)*self._smooth_centroid[1]
                new_deg = angle_deg
                while new_deg - self._smooth_angle >  90.0: new_deg -= 180.0
                while new_deg - self._smooth_angle < -90.0: new_deg += 180.0
                self._smooth_angle = a*new_deg + (1-a)*self._smooth_angle
                while self._smooth_angle >  90.0: self._smooth_angle -= 180.0
                while self._smooth_angle <= -90.0: self._smooth_angle += 180.0

        # 組裝結果
        final = []
        for i, det in enumerate(raw_dets):
            out = dict(det)
            if i == best_idx:
                out['tracked'] = True
                out['stable']  = self._stable
                x1, y1, x2, y2 = self._tracked_box
                out['box'] = (x1, y1, x2, y2)
                if self._smooth_centroid:
                    out['center'] = (self._smooth_centroid[0], self._smooth_centroid[1])
                if self._smooth_angle is not None:
                    out['angle_deg'] = self._smooth_angle
            else:
                out['tracked'] = False
                out['stable']  = False
            final.append(out)

        with self._results_lock:
            self._results = final
        return copy.deepcopy(final)

    def _update_tracking(self, confs_arr, boxes_all) -> int:
        if self._tracked_box is None:
            appear_mask = confs_arr >= self._conf_thresh
            best_idx = int(np.argmax(confs_arr if not appear_mask.any()
                                     else np.where(appear_mask, confs_arr, -1.0)))
            self._track_iou_miss = 0
            return best_idx
        ious         = [_iou(self._tracked_box, list(b)) for b in boxes_all]
        best_iou_val = max(ious)
        best_iou_idx = int(np.argmax(ious))
        if best_iou_val >= _IOU_TRACK_THRESH:
            self._track_iou_miss = 0
            return best_iou_idx
        self._track_iou_miss += 1
        if self._track_iou_miss < _RELOCK_FRAMES:
            return best_iou_idx
        appear_mask = confs_arr >= self._conf_thresh
        best_idx = int(np.argmax(confs_arr if not appear_mask.any()
                                  else np.where(appear_mask, confs_arr, -1.0)))
        self.reset_tracking()
        self._consec_det = 1
        return best_idx

    # ── 取得結果 ──────────────────────────────────────────────────────────────

    def get_all(self) -> list:
        with self._results_lock:
            return copy.deepcopy(self._results)

    def get_tracked(self):
        with self._results_lock:
            for det in self._results:
                if det.get('tracked') and det.get('stable'):
                    return copy.deepcopy(det)
        return None

    def get_count(self) -> int:
        with self._results_lock:
            return len(self._results)

    def is_empty(self) -> bool:
        with self._results_lock:
            return len(self._results) == 0

    # ── ROI ───────────────────────────────────────────────────────────────────

    def set_roi(self, x1: int, y1: int, x2: int, y2: int):
        self._roi.set(x1, y1, x2, y2)

    def clear_roi(self):
        self._roi.clear()

    def get_roi(self):
        return self._roi.get()

    def has_roi(self) -> bool:
        return self._roi.is_active()

    # ── 設定 ──────────────────────────────────────────────────────────────────

    def set_conf_thresh(self, val: float):
        self._conf_thresh = float(val)

    def set_imgsz(self, imgsz: int):
        self._imgsz = int(imgsz)

    # ── 屬性 ──────────────────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    @property
    def is_stable(self) -> bool:
        return self._stable

    @property
    def model_path(self) -> str:
        return self._model_path
