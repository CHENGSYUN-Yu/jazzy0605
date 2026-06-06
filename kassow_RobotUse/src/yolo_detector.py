"""
yolo_detector.py — YOLO 偵測引擎（獨立工廠，含跨幀追蹤）

職責：
    自行載入 YOLO ONNX 模型並執行推論，封裝完整偵測流程：
        載入模型 → 前處理 → ONNX 推論 → 後處理 → ROI 過濾
        → IoU 跨幀追蹤 → EMA 平滑 → 穩定判斷 → 回傳結果

    detect_node 退化為薄薄的 ROS 傳輸層（訂閱相機、發布結果），
    所有偵測智慧集中在此物件。

    同時支援「接收模式」（update()，接收 detect_node 預計算的 all_dets_json）
    以維持向下相容。

    內外隔離：所有輸出皆深拷貝，外部修改不影響內部追蹤狀態。

外部使用方式：
    detector = YoloDetector('/models_v5/seg/seg_v1_640.onnx')
    detector.load_model()
    detector.set_roi(275, 153, 442, 342)

    # 每幀呼叫，自動維護追蹤狀態
    dets = detector.infer(frame_bgr)
    tracked = detector.get_tracked()   # 穩定追蹤目標（None 表示尚未穩定）
    all_dets = detector.get_all()      # 所有 ROI 內偵測結果
"""

import copy
import math
import os
import threading

import cv2
import numpy as np
try:
    import onnxruntime as ort
    _ORT_AVAILABLE = True
except ImportError:
    ort = None
    _ORT_AVAILABLE = False

from .roi_filter import RoiFilter

# ── 預設常數 ─────────────────────────────────────────────────────────────────
_DEFAULT_CONF  = 0.30
_DEFAULT_NMS   = 0.50
_DEFAULT_IMGSZ = 640

# ── 追蹤參數 ─────────────────────────────────────────────────────────────────
_EMA_ALPHA        = 0.12   # EMA 平滑係數（越小越平滑）
_CONFIRM_FRAMES   = 3      # 連續幾幀才算穩定鎖定
_GRACE_FRAMES     = 3      # 消失幾幀後才放棄追蹤
_IOU_TRACK_THRESH = 0.25   # 認定同一物體的最低 IoU
_RELOCK_FRAMES    = 5      # IoU miss 幾幀後才重新選目標


# ── 模組級純函式 ──────────────────────────────────────────────────────────────

def _iou(a, b):
    """兩個 [x1,y1,x2,y2] 框的 IoU。"""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-8)


def _mask_edge_orientation(mask: np.ndarray):
    """最小外接旋轉矩形長邊方向 → (centroid, axis_vec, half_len, angle_deg)。"""
    contours, _ = cv2.findContours(mask.astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, None, None, None
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 5:
        return None, None, None, None
    (cx, cy), (w, h), rect_deg = cv2.minAreaRect(contour)
    if h >= w:
        long_deg = rect_deg + 90.0
        half_len = h / 2.0
    else:
        long_deg = rect_deg
        half_len = w / 2.0
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
    proto_h = out1.shape[2]
    proto_w = out1.shape[3]
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
    preds_f   = preds[keep]
    confs_f   = confs_all[keep]
    cls_ids_f = cls_ids_all[keep]
    coeffs_f  = preds_f[:, 4 + nc:]
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
    boxes     = boxes[idxs]
    confs_f   = confs_f[idxs]
    cls_ids_f = cls_ids_f[idxs]
    coeffs_f  = coeffs_f[idxs]
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


def _postprocess_detect_nms(out0, scale, pad_x, pad_y,
                             orig_h, orig_w, conf_thresh):
    dets  = out0[0]
    valid = dets[:, 4] > conf_thresh
    dets  = dets[valid]
    if len(dets) == 0:
        return [], [], [], []
    x1o = np.clip((dets[:, 0] - pad_x) / scale, 0, orig_w)
    y1o = np.clip((dets[:, 1] - pad_y) / scale, 0, orig_h)
    x2o = np.clip((dets[:, 2] - pad_x) / scale, 0, orig_w)
    y2o = np.clip((dets[:, 3] - pad_y) / scale, 0, orig_h)
    boxes   = list(np.stack([x1o, y1o, x2o, y2o], axis=1))
    confs   = list(dets[:, 4])
    cls_ids = [int(c) for c in dets[:, 5]]
    masks   = []
    for bx1, by1, bx2, by2 in zip(x1o, y1o, x2o, y2o):
        m = np.zeros((orig_h, orig_w), dtype=bool)
        m[int(by1):int(by2), int(bx1):int(bx2)] = True
        masks.append(m)
    return boxes, confs, masks, cls_ids


# ══════════════════════════════════════════════════════════════════════════════

class YoloDetector:
    """
    YOLO 偵測引擎獨立工廠物件（含跨幀追蹤）。

    整合完整偵測流程：推論 → ROI 過濾 → IoU 追蹤 → EMA 平滑 → 穩定判斷。
    detect_node 只需作為薄薄的 ROS 傳輸層使用本物件。

    同時提供 update() 接收預計算結果，維持向下相容。
    """

    def __init__(self,
                 model_path: str = None,
                 conf_thresh: float = _DEFAULT_CONF,
                 nms_iou:    float = _DEFAULT_NMS):
        self._model_path  = model_path
        self._conf_thresh = conf_thresh
        self._nms_iou     = nms_iou
        self._roi         = RoiFilter()
        self._results_lock = threading.Lock()   # 保護 _results 跨執行緒讀寫

        # ONNX session
        self._sess          = None
        self._input_name    = None
        self._infer_imgsz   = _DEFAULT_IMGSZ
        self._is_detect_nms = False

        # 追蹤狀態
        self._tracked_box     = None
        self._smooth_centroid = None
        self._smooth_angle    = None
        self._consec_det      = 0
        self._consec_nodet    = 0
        self._stable          = False
        self._track_iou_miss  = 0

        # 最後一次結果（外部取走時深拷貝）
        self._results = []

    # ── 模型載入 ──────────────────────────────────────────────────────────────

    def load_model(self) -> bool:
        if not _ORT_AVAILABLE:
            print('[YoloDetector] onnxruntime 未安裝，ONNX 路徑不可用（改用 YoloDetector_pt）')
            return False
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        available = ort.get_available_providers()
        use_cuda  = 'CUDAExecutionProvider' in available
        if use_cuda:
            providers = [('CUDAExecutionProvider', {'device_id': 0}),
                         'CPUExecutionProvider']
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1
        else:
            providers = ['CPUExecutionProvider']
            opts.intra_op_num_threads = 2
            opts.inter_op_num_threads = 1
            opts.add_session_config_entry('session.intra_op.allow_spinning', '0')
            opts.add_session_config_entry('session.inter_op.allow_spinning', '0')
        try:
            self._sess        = ort.InferenceSession(self._model_path,
                                                     sess_options=opts,
                                                     providers=providers)
            self._input_name  = self._sess.get_inputs()[0].name
            input_shape       = self._sess.get_inputs()[0].shape
            self._infer_imgsz = int(input_shape[2]) if len(input_shape) >= 3 \
                                else _DEFAULT_IMGSZ
            n_out             = len(self._sess.get_outputs())
            out0_shape        = self._sess.get_outputs()[0].shape
            self._is_detect_nms = (n_out == 1 and len(out0_shape) == 3
                                   and out0_shape[2] == 6)
            provider_used = self._sess.get_providers()[0]
            model_type    = 'detect(NMS)' if self._is_detect_nms else 'seg'
            print(f'[YoloDetector] model={os.path.basename(self._model_path)}'
                  f'  imgsz={self._infer_imgsz}  type={model_type}'
                  f'  provider={provider_used}')
            return True
        except Exception as e:
            print(f'[YoloDetector] model load failed: {e}')
            return False

    # ── 追蹤狀態重置 ──────────────────────────────────────────────────────────

    def reset_tracking(self):
        """清除所有跨幀追蹤狀態（場景切換時呼叫）。"""
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
        對單幀 BGR 影像執行 YOLO 推論，並自動維護跨幀追蹤狀態。

        回傳：偵測結果 list（深拷貝），每個元素：
            {
              'center':    (cx, cy),          # EMA 平滑重心（追蹤目標）或 mask 重心
              'box':       (x1, y1, x2, y2),  # EMA 平滑 bbox（追蹤目標）或原始 bbox
              'conf':      float,
              'cls_id':    int,
              'mask':      np.ndarray,
              'angle_deg': float,              # EMA 平滑角度（追蹤目標）
              'tracked':   bool,               # 是否為 IoU 追蹤目標
              'stable':    bool,               # 追蹤目標是否已穩定（≥ CONFIRM_FRAMES）
            }
        """
        if self._sess is None:
            return []

        H, W = frame_bgr.shape[:2]
        tensor, scale, pad_x, pad_y = _preprocess(frame_bgr, self._infer_imgsz)
        outputs = self._sess.run(None, {self._input_name: tensor})

        if self._is_detect_nms:
            boxes, confs, masks, cls_ids = _postprocess_detect_nms(
                outputs[0], scale, pad_x, pad_y, H, W, self._conf_thresh)
        else:
            boxes, confs, masks, cls_ids = _postprocess_seg(
                outputs[0], outputs[1], scale, pad_x, pad_y,
                H, W, self._conf_thresh, self._nms_iou, self._infer_imgsz)

        # ROI 過濾 + centroid 計算
        raw_dets = []
        for box, conf, mask, cls_id in zip(boxes, confs, masks, cls_ids):
            centroid, _, _, angle_deg = _mask_edge_orientation(mask)
            if centroid is None:
                x1, y1, x2, y2 = box
                centroid  = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                angle_deg = 0.0
            cx, cy = centroid
            if not self._roi.contains(cx, cy):
                continue
            x1, y1, x2, y2 = [float(v) for v in box]
            raw_dets.append({
                'center':    (cx, cy),
                'box':       (x1, y1, x2, y2),
                'conf':      float(conf),
                'cls_id':    int(cls_id),
                'mask':      mask,
                'angle_deg': float(angle_deg) if angle_deg is not None else 0.0,
            })

        # ── 無偵測：grace period 後清除追蹤 ──────────────────────────────────
        if len(raw_dets) == 0:
            self._consec_det    = 0
            self._consec_nodet += 1
            if self._consec_nodet >= _GRACE_FRAMES:
                # Grace 期滿：清除追蹤狀態 + 清空結果，bbox 消失
                self.reset_tracking()
                with self._results_lock:
                    self._results = []
                return []
            # Grace 期間：保持舊結果但標記 stable=False（防止閃爍）
            with self._results_lock:
                kept = [dict(r, stable=False) for r in self._results]
                self._results = kept
            return copy.deepcopy(kept)

        # ── 有偵測：IoU 追蹤 ──────────────────────────────────────────────────
        self._consec_nodet  = 0
        self._consec_det   += 1
        confs_arr = np.array([d['conf'] for d in raw_dets])
        boxes_all = [d['box'] for d in raw_dets]
        masks_all = [d['mask'] for d in raw_dets]

        best_idx = self._update_tracking(confs_arr, boxes_all, masks_all)

        # ── 穩定確認 ──────────────────────────────────────────────────────────
        if self._consec_det >= _CONFIRM_FRAMES:
            self._stable = True

        # ── EMA bbox（追蹤目標）──────────────────────────────────────────────
        raw_box = list(boxes_all[best_idx])
        if self._tracked_box is None or not self._stable:
            self._tracked_box = raw_box[:]
        else:
            a = _EMA_ALPHA
            self._tracked_box = [a * r + (1 - a) * s
                                 for r, s in zip(raw_box, self._tracked_box)]

        # ── EMA centroid / angle（追蹤目標）──────────────────────────────────
        centroid, _, _, angle_deg = _mask_edge_orientation(masks_all[best_idx])
        if centroid is not None:
            a = _EMA_ALPHA
            if self._smooth_centroid is None or not self._stable:
                self._smooth_centroid = list(centroid)
                self._smooth_angle    = angle_deg
            else:
                self._smooth_centroid[0] = a * centroid[0] + (1 - a) * self._smooth_centroid[0]
                self._smooth_centroid[1] = a * centroid[1] + (1 - a) * self._smooth_centroid[1]
                new_deg = angle_deg
                while new_deg - self._smooth_angle >  90.0: new_deg -= 180.0
                while new_deg - self._smooth_angle < -90.0: new_deg += 180.0
                self._smooth_angle = a * new_deg + (1 - a) * self._smooth_angle
                while self._smooth_angle >  90.0: self._smooth_angle -= 180.0
                while self._smooth_angle <= -90.0: self._smooth_angle += 180.0

        # ── 組裝最終結果 ──────────────────────────────────────────────────────
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

    def _update_tracking(self, confs_arr, boxes_all, masks_all) -> int:
        """IoU 追蹤邏輯，回傳本幀 best_idx。"""
        if self._tracked_box is None:
            appear_mask = confs_arr >= self._conf_thresh
            if not appear_mask.any():
                best_idx = int(np.argmax(confs_arr))
            else:
                best_idx = int(np.argmax(np.where(appear_mask, confs_arr, -1.0)))
            self._track_iou_miss = 0
            return best_idx

        ious          = [_iou(self._tracked_box, list(b)) for b in boxes_all]
        best_iou_val  = max(ious)
        best_iou_idx  = int(np.argmax(ious))

        if best_iou_val >= _IOU_TRACK_THRESH:
            self._track_iou_miss = 0
            return best_iou_idx

        self._track_iou_miss += 1
        if self._track_iou_miss < _RELOCK_FRAMES:
            # Grace：保持舊追蹤，選最近的框
            return best_iou_idx

        # 重新選目標
        appear_mask = confs_arr >= self._conf_thresh
        if not appear_mask.any():
            best_idx = int(np.argmax(confs_arr))
        else:
            best_idx = int(np.argmax(np.where(appear_mask, confs_arr, -1.0)))
        self.reset_tracking()
        self._consec_det     = 1
        self._track_iou_miss = 0
        return best_idx

    # ── 接收預計算結果（向下相容）────────────────────────────────────────────

    def update(self, dets_data: dict):
        """
        接收 all_dets_json 解析後的 dict（detect_node 已完成推論的場景）。
        不執行追蹤邏輯，直接封裝結果。
        """
        results = []
        for det in dets_data.get('dets', []):
            conf = det.get('conf', 0.0)
            if conf < self._conf_thresh:
                continue
            cx, cy = det.get('centroid_px', [0.0, 0.0])
            if not self._roi.contains(cx, cy):
                continue
            results.append({
                'center':      (cx, cy),
                'conf':        conf,
                'cls_id':      det.get('cls_id', 0),
                'tracked':     det.get('tracked', False),
                'stable':      det.get('stable', False),
                'angle_deg':   det.get('angle_deg', 0.0),
                'pos_cam_mm':  det.get('pos_cam_mm'),
                'pos_base_mm': det.get('pos_base_mm'),
            })
        with self._results_lock:
            self._results = results

    # ── 取得結果 ──────────────────────────────────────────────────────────────

    def get_all(self) -> list:
        """取得最後一次推論的所有結果（深拷貝）。"""
        with self._results_lock:
            return copy.deepcopy(self._results)

    def get_tracked(self):
        """取得穩定追蹤目標（tracked=True 且 stable=True）；無則返回 None。"""
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

    def set_nms_iou(self, val: float):
        self._nms_iou = float(val)

    # ── 屬性 ──────────────────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._sess is not None

    @property
    def is_stable(self) -> bool:
        """目前追蹤目標是否穩定。"""
        return self._stable

    @property
    def model_path(self) -> str:
        return self._model_path

    @property
    def infer_imgsz(self) -> int:
        return self._infer_imgsz
