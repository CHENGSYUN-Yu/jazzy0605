"""
yolo_engine.py — 單相機 GPU YOLO 推論引擎（獨立工廠物件）

設計：
  - 每個實例負責一台相機，物件間完全獨立
  - 各自載入模型、各自的推論執行緒、各自的 smoother
  - 不共用模型或 Lock，真正並行推論
  - ROI 自動存檔到 config/roi.json，開啟 GUI 自動載入

使用：
    head = YoloEngine(model_path, cam_id=0, fps=10.0)  # D435I
    hand = YoloEngine(model_path, cam_id=1, fps=10.0)  # D405
    head.start(rs)
    hand.start(rs)
    tex  = head.get_overlay_tex()
    dets = head.get_dets()
    head.set_roi(x1, y1, x2, y2)
    head.stop()
"""

import copy
import json
import os
import threading
import time

import cv2
import numpy as np

os.environ.setdefault('YOLO_AUTOINSTALL', 'false')

# ROI 設定存檔路徑
_ROI_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'config', 'roi.json'
)

# 預設信心分門檻（各相機）
_DEFAULT_CONF = {0: 0.15, 1: 0.50}


# ── EMA bbox 平滑器 ───────────────────────────────────────────────────────────

class _BoxSmoother:
    """IoU 追蹤 + EMA 平滑 + 穩定幀數確認。"""

    def __init__(self, alpha: float = 0.12, iou_thresh: float = 0.25,
                 confirm_frames: int = 3):
        self._alpha          = alpha
        self._iou_thresh     = iou_thresh
        self._confirm_frames = confirm_frames
        self._prev: list     = []
        self._consec: list   = []

    def smooth(self, dets: list) -> list:
        if not dets:
            self._prev = []
            self._consec = []
            return dets
        if not self._prev:
            self._prev   = [copy.deepcopy(d) for d in dets]
            self._consec = [1] * len(dets)
            for d in self._prev:
                d['stable'] = False
            return copy.deepcopy(self._prev)

        matched: dict[int, int] = {}
        used_prev: set[int] = set()

        for ni, new_det in enumerate(dets):
            best_iou, best_pi = self._iou_thresh, -1
            for pi, prev_det in enumerate(self._prev):
                if pi in used_prev:
                    continue
                iou = _box_iou(new_det['box'], prev_det['box'])
                if iou > best_iou:
                    best_iou, best_pi = iou, pi
            if best_pi >= 0:
                matched[ni] = best_pi
                used_prev.add(best_pi)

        result, new_prev, new_consec = [], [], []
        a = self._alpha
        for ni, det in enumerate(dets):
            det = copy.deepcopy(det)
            if ni in matched:
                pi     = matched[ni]
                pb     = self._prev[pi]['box']
                nb     = det['box']
                x1     = a * nb[0] + (1 - a) * pb[0]
                y1     = a * nb[1] + (1 - a) * pb[1]
                x2     = a * nb[2] + (1 - a) * pb[2]
                y2     = a * nb[3] + (1 - a) * pb[3]
                det['box']    = (x1, y1, x2, y2)
                det['center'] = ((x1 + x2) / 2, (y1 + y2) / 2)
                consec = self._consec[pi] + 1
            else:
                consec = 1
            det['stable'] = consec >= self._confirm_frames
            new_consec.append(consec)
            new_prev.append(copy.deepcopy(det))
            result.append(det)

        self._prev   = new_prev
        self._consec = new_consec
        return result

    def reset(self) -> None:
        self._prev   = []
        self._consec = []


def _copy_det(d: dict) -> dict:
    return {**d, 'box': tuple(d['box']), 'center': tuple(d['center'])}


def _box_iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


# ── 主要物件 ──────────────────────────────────────────────────────────────────

class YoloEngine:
    """
    單相機 YOLO 推論引擎（獨立工廠物件）。

    每個實例對應一台相機，擁有各自的模型、執行緒、smoother。
    cam_id:
      0 = D435I（頭部相機，GUI Cam 1），預設信心分 0.15
      1 = D405 （手部相機，GUI Cam 2），預設信心分 0.50
    """

    def __init__(self, model_path: str, cam_id: int = 0, fps: float = 10.0):
        self._model_path    = model_path
        self._cam_id        = cam_id
        self._interval      = 1.0 / fps

        self._model         = None
        self._model_lock    = threading.Lock()   # 每個實例各自的 Lock
        self._rs            = None
        self._running       = False
        self._loaded        = False
        self._load_error:  'str | None' = None

        # 結果儲存
        self._result_lock   = threading.Lock()
        self._tex:          'np.ndarray | None' = None
        self._dets:         list = []

        # 信心分門檻
        self._conf_threshold: float = _DEFAULT_CONF.get(cam_id, 0.5)

        # EMA 平滑器
        self._smoother      = _BoxSmoother(alpha=0.12, confirm_frames=3)

        # ROI
        self._roi:         'tuple | None' = None
        self._preview_roi: 'tuple | None' = None
        self._roi_lock      = threading.Lock()
        self._load_roi_from_file()

    # ── 生命週期 ──────────────────────────────────────────────────────────────

    def start(self, rs) -> None:
        """注入 RealSense，背景載入模型後啟動推論執行緒。"""
        self._rs = rs
        threading.Thread(target=self._load_and_run, daemon=True,
                         name=f'yolo_engine_cam{self._cam_id}').start()

    def stop(self) -> None:
        self._running = False
        self._smoother.reset()

    # ── 信心分門檻 ────────────────────────────────────────────────────────────

    def set_conf_threshold(self, val: float) -> None:
        self._conf_threshold = max(0.0, min(1.0, float(val)))

    @property
    def conf_threshold(self) -> float:
        return self._conf_threshold

    # ── ROI API ───────────────────────────────────────────────────────────────

    def set_roi(self, x1: int, y1: int, x2: int, y2: int) -> None:
        roi = (int(min(x1, x2)), int(min(y1, y2)),
               int(max(x1, x2)), int(max(y1, y2)))
        with self._roi_lock:
            self._roi         = roi
            self._preview_roi = None
        self._save_roi_to_file()

    def clear_roi(self) -> None:
        with self._roi_lock:
            self._roi         = None
            self._preview_roi = None
        self._save_roi_to_file()

    def get_roi(self) -> 'tuple | None':
        with self._roi_lock:
            return self._roi

    def set_preview_roi(self, x1: int, y1: int, x2: int, y2: int) -> None:
        with self._roi_lock:
            self._preview_roi = (int(min(x1, x2)), int(min(y1, y2)),
                                 int(max(x1, x2)), int(max(y1, y2)))

    def clear_preview_roi(self) -> None:
        with self._roi_lock:
            self._preview_roi = None

    def _load_roi_from_file(self) -> None:
        try:
            if not os.path.exists(_ROI_CONFIG):
                return
            with open(_ROI_CONFIG) as f:
                data = json.load(f)
            key = f'cam{self._cam_id}'
            if key in data and data[key]:
                self._roi = tuple(data[key])
                print(f'[YoloEngine cam{self._cam_id}] ROI 載入：{self._roi}')
        except Exception as e:
            print(f'[YoloEngine cam{self._cam_id}] ROI 載入失敗：{e}')

    def _save_roi_to_file(self) -> None:
        try:
            os.makedirs(os.path.dirname(_ROI_CONFIG), exist_ok=True)
            data = {}
            if os.path.exists(_ROI_CONFIG):
                with open(_ROI_CONFIG) as f:
                    data = json.load(f)
            with self._roi_lock:
                data[f'cam{self._cam_id}'] = list(self._roi) if self._roi else None
                with open(_ROI_CONFIG, 'w') as f:
                    json.dump(data, f, indent=2)
        except Exception as e:
            print(f'[YoloEngine cam{self._cam_id}] ROI 存檔失敗：{e}')

    # ── 結果存取（thread-safe）────────────────────────────────────────────────

    def get_overlay_tex(self) -> 'np.ndarray | None':
        with self._result_lock:
            return self._tex.copy() if self._tex is not None else None

    def get_dets(self) -> list:
        with self._result_lock:
            return copy.deepcopy(self._dets)

    def get_det_count(self) -> int:
        return len(self.get_dets())

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def load_error(self) -> 'str | None':
        return self._load_error

    @property
    def cam_id(self) -> int:
        return self._cam_id

    # ── 內部：載入模型 ────────────────────────────────────────────────────────

    def _load_and_run(self) -> None:
        try:
            from ultralytics import YOLO
            import torch
            print(f'[YoloEngine cam{self._cam_id}] 載入模型...')
            self._model = YOLO(self._model_path)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            with self._model_lock:
                self._model(dummy, device=device, verbose=False, imgsz=640)
            self._loaded  = True
            self._running = True
            print(f'[YoloEngine cam{self._cam_id}] 就緒（{device}）')
            t = threading.Thread(
                target=self._infer_loop, args=(device,),
                daemon=True, name=f'yolo_infer_cam{self._cam_id}')
            t.start()
        except Exception as e:
            self._load_error = str(e)
            print(f'[YoloEngine cam{self._cam_id}] 載入失敗：{e}')

    # ── 內部：推論迴圈 ────────────────────────────────────────────────────────

    def _infer_loop(self, device: str) -> None:
        from src.realsense import RealSense as _RS
        while self._running:
            t0 = time.perf_counter()
            if self._rs is None:
                time.sleep(self._interval)
                continue
            frame = self._rs.get_frame(self._cam_id)
            if frame is None:
                time.sleep(self._interval)
                continue
            try:
                with self._model_lock:
                    results = self._model(
                        frame, device=device, verbose=False, imgsz=640)
                with self._roi_lock:
                    roi     = self._roi
                    preview = self._preview_roi
                dets = self._parse_results(results, roi, self._conf_threshold)
                dets = self._smoother.smooth(dets)
                tex  = self._draw_overlay(frame, dets, roi, preview)
                with self._result_lock:
                    self._dets = dets
                    self._tex  = tex
            except Exception as e:
                print(f'[YoloEngine cam{self._cam_id}] 推論錯誤：{e}')
            elapsed = time.perf_counter() - t0
            wait = self._interval - elapsed
            if wait > 0:
                time.sleep(wait)

    # ── 內部：解析結果 ────────────────────────────────────────────────────────

    @staticmethod
    def _parse_results(results, roi: 'tuple|None',
                       conf_threshold: float = 0.5) -> list:
        dets = []
        if not results:
            return dets
        r = results[0]
        if r.boxes is None:
            return dets
        for i, box in enumerate(r.boxes):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            if roi is not None:
                rx1, ry1, rx2, ry2 = roi
                if not (rx1 <= cx <= rx2 and ry1 <= cy <= ry2):
                    continue
            conf = float(box.conf[0])
            if conf < conf_threshold:
                continue
            angle_deg = 0.0
            if r.masks is not None:
                try:
                    mask = r.masks.data[i].cpu().numpy()
                    angle_deg = _mask_angle(mask)
                except Exception:
                    pass
            dets.append({'box': (x1, y1, x2, y2), 'center': (cx, cy),
                         'conf': conf, 'cls_id': int(box.cls[0]),
                         'angle_deg': angle_deg})
        return dets

    # ── 內部：繪製 overlay ────────────────────────────────────────────────────

    @staticmethod
    def _draw_overlay(frame: np.ndarray, dets: list,
                      roi: 'tuple|None', preview_roi: 'tuple|None') -> np.ndarray:
        from src.realsense import RealSense as _RS
        vis = frame.copy()
        for det in dets:
            x1, y1, x2, y2 = [int(v) for v in det['box']]
            cx, cy = int(det['center'][0]), int(det['center'][1])
            color  = (0, 255, 100) if det.get('stable') else (160, 160, 160)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis, f'{det["conf"]:.2f}', (x1, max(y1 - 4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            cv2.drawMarker(vis, (cx, cy), color, cv2.MARKER_CROSS, 14, 2)
        if roi is not None:
            rx1, ry1, rx2, ry2 = [int(v) for v in roi]
            overlay = vis.copy()
            cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), (0, 255, 0), -1)
            cv2.addWeighted(overlay, 0.10, vis, 0.90, 0, vis)
            cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)
            cv2.putText(vis, 'ROI', (rx1 + 4, ry1 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if preview_roi is not None:
            px1, py1, px2, py2 = [int(v) for v in preview_roi]
            _draw_dashed_rect(vis, px1, py1, px2, py2, (0, 180, 255), 2)
        return _RS._to_texture(vis)


# ── 工具函式 ──────────────────────────────────────────────────────────────────

def _mask_angle(mask: np.ndarray) -> float:
    try:
        pts = np.column_stack(np.where(mask > 0.5))
        if len(pts) < 5:
            return 0.0
        _, (_, _), angle = cv2.fitEllipse(pts[:, ::-1].astype(np.float32))
        return float(angle)
    except Exception:
        return 0.0


def _draw_dashed_rect(img: np.ndarray, x1: int, y1: int,
                      x2: int, y2: int, color: tuple, thickness: int,
                      dash: int = 10) -> None:
    pts = [((x1,y1),(x2,y1)), ((x2,y1),(x2,y2)),
           ((x2,y2),(x1,y2)), ((x1,y2),(x1,y1))]
    for (ax, ay), (bx, by) in pts:
        dist = int(((bx-ax)**2 + (by-ay)**2)**0.5)
        if dist == 0:
            continue
        for i in range(0, dist, dash * 2):
            t0 = i / dist
            t1 = min((i + dash) / dist, 1.0)
            p0 = (int(ax + (bx-ax)*t0), int(ay + (by-ay)*t0))
            p1 = (int(ax + (bx-ax)*t1), int(ay + (by-ay)*t1))
            cv2.line(img, p0, p1, color, thickness, cv2.LINE_AA)
