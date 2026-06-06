"""
yolo_detect_handcam.py — 手部相機 YOLO 偵測引擎（完整獨立工廠）

職責：
    以手部相機（EIH，RealSense D405）的每幀畫面執行 YOLO 偵測。
    完整獨立實作，不依賴 YoloDetector，可單獨演進。

與 YoloDetector 的差異：
    - 無 ROI 過濾（手部相機近距離拍攝，不需要限制範圍）
    - 可使用不同的 YOLO 模型（針對近距離、不同視角優化）
    - 推論邏輯完全獨立，內部修改不影響頭部相機

外部使用方式：
    hcam = YoloDetectHandcam('/models_v5/seg/seg_v1_640.onnx')
    hcam.load_model()
    dets = hcam.infer(frame_bgr)
    best = hcam.get_tracked()
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

# ── 預設常數 ─────────────────────────────────────────────────────────────────
_DEFAULT_CONF  = 0.30
_DEFAULT_NMS   = 0.50
_DEFAULT_IMGSZ = 640

# ── 追蹤參數 ─────────────────────────────────────────────────────────────────
_EMA_ALPHA        = 0.12
_CONFIRM_FRAMES   = 3
_GRACE_FRAMES     = 3
_IOU_TRACK_THRESH = 0.25
_RELOCK_FRAMES    = 5


# ── 模組級純函式 ──────────────────────────────────────────────────────────────

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


def _preprocess(bgr: np.ndarray, imgsz: int):
    h, w    = bgr.shape[:2]
    scale   = imgsz / max(h, w)
    nh, nw  = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas  = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    pad_y   = (imgsz - nh) // 2
    pad_x   = (imgsz - nw) // 2
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    tensor  = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    return tensor[np.newaxis], scale, pad_x, pad_y


def _postprocess_seg(out0, out1, scale, pad_x, pad_y,
                     orig_h, orig_w, conf_thresh, nms_iou, imgsz):
    nm      = 32
    nc      = out0.shape[1] - 4 - nm
    proto_h = out1.shape[2]; proto_w = out1.shape[3]
    preds   = out0[0].T
    if nc == 1:
        confs_all   = preds[:, 4]
        cls_ids_all = np.zeros(len(preds), dtype=np.int32)
    else:
        cls_scores  = preds[:, 4:4 + nc]
        confs_all   = cls_scores.max(axis=1)
        cls_ids_all = cls_scores.argmax(axis=1).astype(np.int32)
    keep = confs_all > conf_thresh
    if not keep.any():
        return [], [], [], []
    preds_f   = preds[keep]; confs_f = confs_all[keep]
    cls_ids_f = cls_ids_all[keep]; coeffs_f = preds_f[:, 4 + nc:]
    cx, cy = preds_f[:, 0], preds_f[:, 1]
    bw, bh = preds_f[:, 2], preds_f[:, 3]
    x1o = np.clip((cx - bw / 2 - pad_x) / scale, 0, orig_w)
    y1o = np.clip((cy - bh / 2 - pad_y) / scale, 0, orig_h)
    x2o = np.clip((cx + bw / 2 - pad_x) / scale, 0, orig_w)
    y2o = np.clip((cy + bh / 2 - pad_y) / scale, 0, orig_h)
    boxes = np.stack([x1o, y1o, x2o, y2o], axis=1)
    idxs  = cv2.dnn.NMSBoxes(boxes.tolist(), confs_f.tolist(), conf_thresh, nms_iou)
    if len(idxs) == 0:
        return [], [], [], []
    idxs      = np.array(idxs).flatten()
    boxes     = boxes[idxs]; confs_f = confs_f[idxs]
    cls_ids_f = cls_ids_f[idxs]; coeffs_f = coeffs_f[idxs]
    protos = out1[0]
    raw    = 1.0 / (1.0 + np.exp(-(coeffs_f @ protos.reshape(nm, -1))))
    raw    = raw.reshape(-1, proto_h, proto_w)
    masks  = []
    for m_proto in raw:
        m_infer = cv2.resize(m_proto, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
        m_crop  = m_infer[pad_y:pad_y + int(round(orig_h * scale)),
                          pad_x:pad_x + int(round(orig_w * scale))]
        masks.append(cv2.resize(m_crop, (orig_w, orig_h),
                                interpolation=cv2.INTER_LINEAR) > 0.5)
    return list(boxes), list(confs_f), masks, [int(c) for c in cls_ids_f]


def _postprocess_detect_nms(out0, scale, pad_x, pad_y, orig_h, orig_w, conf_thresh):
    dets  = out0[0]; valid = dets[:, 4] > conf_thresh; dets = dets[valid]
    if len(dets) == 0:
        return [], [], [], []
    x1o = np.clip((dets[:, 0] - pad_x) / scale, 0, orig_w)
    y1o = np.clip((dets[:, 1] - pad_y) / scale, 0, orig_h)
    x2o = np.clip((dets[:, 2] - pad_x) / scale, 0, orig_w)
    y2o = np.clip((dets[:, 3] - pad_y) / scale, 0, orig_h)
    boxes = list(np.stack([x1o, y1o, x2o, y2o], axis=1))
    masks = []
    for bx1, by1, bx2, by2 in zip(x1o, y1o, x2o, y2o):
        m = np.zeros((orig_h, orig_w), dtype=bool)
        m[int(by1):int(by2), int(bx1):int(bx2)] = True
        masks.append(m)
    return boxes, list(dets[:, 4]), masks, [int(c) for c in dets[:, 5]]


# ══════════════════════════════════════════════════════════════════════════════

class YoloDetectHandcam:
    """
    手部相機（D405）YOLO 偵測引擎。

    完整獨立實作：模型載入、推論、追蹤、EMA 平滑、穩定判斷。
    無 ROI 功能（手部相機近距離拍攝，不需限制範圍）。
    內外隔離：輸出皆深拷貝。
    """

    def __init__(self,
                 model_path: str = None,
                 conf_thresh: float = _DEFAULT_CONF,
                 nms_iou:    float = _DEFAULT_NMS):
        self._model_path  = model_path
        self._conf_thresh = conf_thresh
        self._nms_iou     = nms_iou

        self._results_lock  = threading.Lock()
        self._device        = 'cpu'
        self._proc          = None
        self._shm           = None
        self._frame_buf     = None
        self._result_queue: queue.Queue = queue.Queue()
        self._server_hw     = (640, 640)

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
        """啟動 Python 3.11 GPU 推論 subprocess。"""
        try:
            imgsz = _DEFAULT_IMGSZ
            srv_h = srv_w = imgsz
            self._server_hw = (srv_h, srv_w)

            self._shm = SharedMemory(create=True, size=srv_h * srv_w * 3)
            self._frame_buf = np.ndarray(
                (srv_h, srv_w, 3), dtype=np.uint8, buffer=self._shm.buf)

            server_script = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'yolo_infer_server.py')

            _py = sys.executable
            self._proc = subprocess.Popen(
                [_py, server_script,
                 self._model_path, self._shm.name,
                 str(srv_h), str(srv_w), '3'],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, bufsize=1,
            )

            def _log_stderr():
                for line in self._proc.stderr:
                    print(f'[handcam_server] {line.rstrip()}')
            threading.Thread(target=_log_stderr, daemon=True).start()

            self._result_queue = queue.Queue()
            def _read_stdout():
                try:
                    for line in self._proc.stdout:
                        self._result_queue.put(line.rstrip('\n'))
                except Exception:
                    pass
            threading.Thread(target=_read_stdout, daemon=True).start()

            print('[YoloDetectHandcam] 等待 server 就緒...')
            try:
                ready = self._result_queue.get(timeout=120.0)
            except queue.Empty:
                print('[YoloDetectHandcam] server 啟動逾時')
                return False

            if not ready.startswith('READY'):
                return False

            parts = dict(p.split('=') for p in ready.split()[1:] if '=' in p)
            self._device = parts.get('device', 'cpu')
            print(f'[YoloDetectHandcam] ready | '
                  f'model={os.path.basename(self._model_path)} | '
                  f'device={self._device}')
            return True
        except Exception as e:
            print(f'[YoloDetectHandcam] load_model failed: {e}')
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

    # ── 追蹤狀態 ──────────────────────────────────────────────────────────────

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
            'mask'      : np.ndarray
            'angle_deg' : float
            'tracked'   : bool
            'stable'    : bool
        """
        if self._proc is None or self._proc.poll() is not None:
            return []

        H, W = frame_bgr.shape[:2]
        srv_h, srv_w = self._server_hw

        if H != srv_h or W != srv_w:
            frame_send = cv2.resize(frame_bgr, (srv_w, srv_h))
            scale_x, scale_y = W / srv_w, H / srv_h
        else:
            frame_send = frame_bgr
            scale_x = scale_y = 1.0

        np.copyto(self._frame_buf, frame_send)

        try:
            self._proc.stdin.write(f'INFER {self._conf_thresh:.3f}\n')
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError):
            return []

        try:
            raw_json = self._result_queue.get(timeout=5.0)
        except queue.Empty:
            return []

        try:
            server_dets = json.loads(raw_json)
        except json.JSONDecodeError:
            return []

        # centroid 計算（無 ROI 過濾）
        raw_dets = []
        for d in server_dets:
            x1 = d['box'][0] * scale_x
            y1 = d['box'][1] * scale_y
            x2 = d['box'][2] * scale_x
            y2 = d['box'][3] * scale_y
            cx = d['center'][0] * scale_x
            cy = d['center'][1] * scale_y
            raw_dets.append({
                'center':    (cx, cy),
                'box':       (x1, y1, x2, y2),
                'conf':      float(d['conf']),
                'cls_id':    int(d['cls_id']),
                'mask':      None,
                'angle_deg': float(d['angle_deg']),
            })

        # 無偵測：grace period
        if len(raw_dets) == 0:
            self._consec_det    = 0
            self._consec_nodet += 1
            if self._consec_nodet >= _GRACE_FRAMES:
                self.reset_tracking()
            with self._results_lock:
                kept = [dict(r, stable=False) for r in self._results]
                self._results = kept
            return copy.deepcopy(kept)

        # 有偵測：IoU 追蹤
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
            self._tracked_box = [a * r + (1 - a) * s
                                 for r, s in zip(raw_box, self._tracked_box)]

        # EMA centroid / angle（直接用 server 回傳值）
        centroid  = raw_dets[best_idx]['center']
        angle_deg = raw_dets[best_idx]['angle_deg']
        if centroid is not None:
            cx, cy = centroid
            a = _EMA_ALPHA
            if self._smooth_centroid is None or not self._stable:
                self._smooth_centroid = [cx, cy]
                self._smooth_angle    = angle_deg
            else:
                self._smooth_centroid[0] = a * cx + (1 - a) * self._smooth_centroid[0]
                self._smooth_centroid[1] = a * cy + (1 - a) * self._smooth_centroid[1]
                new_deg = angle_deg
                while new_deg - self._smooth_angle >  90.0: new_deg -= 180.0
                while new_deg - self._smooth_angle < -90.0: new_deg += 180.0
                self._smooth_angle = a * new_deg + (1 - a) * self._smooth_angle
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

    # ── 設定 ──────────────────────────────────────────────────────────────────

    def set_conf_thresh(self, val: float):
        self._conf_thresh = float(val)

    def set_nms_iou(self, val: float):
        self._nms_iou = float(val)

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

    @property
    def infer_imgsz(self) -> int:
        return self._infer_imgsz
