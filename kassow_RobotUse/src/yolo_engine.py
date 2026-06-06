"""
yolo_engine.py — 雙相機 GPU YOLO 推論引擎

設計：
  - 一個 YOLO model 實例，GPU 共用（Lock 序列化推論避免 VRAM 競爭）
  - 每台相機一條 daemon thread，各自 ~10fps
  - 兩條 thread 用 50ms 交錯，讓 GPU 時間錯開
  - get_overlay_tex(cam_id) / get_dets(cam_id) thread-safe
  - ROI 自動存檔到 config/roi.json，開啟 GUI 自動載入

使用：
    engine = YoloEngine('models/best20260603.pt')
    engine.start(rs)
    engine.set_roi(0, x1, y1, x2, y2)   # 設定 cam0 ROI 並存檔
    engine.set_preview_roi(0, x1, y1, x2, y2)  # 拖移中的預覽（不存檔）
    engine.clear_roi(0)
"""

import json
import os
import threading
import time

import cv2
import numpy as np

os.environ.setdefault('YOLO_AUTOINSTALL', 'false')

_MODEL_LOCK = threading.Lock()


# ── EMA bbox 平滑器 ───────────────────────────────────────────────────────────

class _BoxSmoother:
    """
    跨幀 IoU 追蹤 + EMA 平滑，消除 bbox 縮放閃爍。

    alpha: EMA 對新觀測的權重（0=完全不更新，1=完全用新值）
      0.3 → 平滑優先（適合靜態場景）
      0.5 → 平衡
    iou_thresh: 低於此值的偵測視為新物件（不繼承歷史軌跡）
    """

    def __init__(self, alpha: float = 0.35, iou_thresh: float = 0.25):
        self._alpha      = alpha
        self._iou_thresh = iou_thresh
        self._prev: list = []   # 上一幀平滑後的 dets

    def smooth(self, dets: list) -> list:
        if not dets:
            self._prev = []
            return dets
        if not self._prev:
            self._prev = [_copy_det(d) for d in dets]
            return list(dets)

        matched: dict[int, int] = {}   # new_idx → prev_idx
        used_prev: set[int] = set()

        # 貪婪 IoU 匹配
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

        # 套用 EMA
        result, new_prev = [], []
        a = self._alpha
        for ni, det in enumerate(dets):
            det = _copy_det(det)
            if ni in matched:
                pb = self._prev[matched[ni]]['box']
                nb = det['box']
                x1 = a * nb[0] + (1 - a) * pb[0]
                y1 = a * nb[1] + (1 - a) * pb[1]
                x2 = a * nb[2] + (1 - a) * pb[2]
                y2 = a * nb[3] + (1 - a) * pb[3]
                det['box']    = (x1, y1, x2, y2)
                det['center'] = ((x1 + x2) / 2, (y1 + y2) / 2)
            result.append(det)
            new_prev.append(_copy_det(det))

        self._prev = new_prev
        return result

    def reset(self) -> None:
        self._prev = []


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

# ROI 設定存檔路徑（相對於 kassow_RobotUse/）
_ROI_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'config', 'roi.json'
)


class YoloEngine:
    """
    雙相機 YOLO 推論引擎。

    cam_ids: 要偵測的相機索引列表，例如 [0, 1]
      0 = D435I（頭部相機，GUI 顯示為 Cam 1）
      1 = D405 （手部相機，GUI 顯示為 Cam 2）
    """

    def __init__(self, model_path: str, cam_ids: list[int] = None,
                 fps: float = 10.0):
        self._model_path = model_path
        self._cam_ids    = cam_ids if cam_ids is not None else [0, 1]
        self._interval   = 1.0 / fps

        self._model      = None
        self._rs         = None
        self._running    = False

        # 每台相機獨立的結果儲存
        self._results: dict[int, dict] = {
            cid: {'tex': None, 'dets': [], 'lock': threading.Lock()}
            for cid in self._cam_ids
        }
        self._threads: list[threading.Thread] = []
        self._loaded = False
        self._load_error: str | None = None
        self._conf_threshold: float = 0.5

        # 每台相機各自一個 EMA 平滑器
        self._smoothers: dict[int, _BoxSmoother] = {
            cid: _BoxSmoother(alpha=0.35) for cid in self._cam_ids
        }

        # ROI：committed（用於過濾）和 preview（拖移中顯示）
        self._roi:         dict[int, 'tuple|None'] = {cid: None for cid in self._cam_ids}
        self._preview_roi: dict[int, 'tuple|None'] = {cid: None for cid in self._cam_ids}
        self._roi_lock = threading.Lock()
        self._load_roi_from_file()   # 自動載入上次儲存的 ROI

    # ── 生命週期 ──────────────────────────────────────────────────────────────

    def start(self, rs) -> None:
        self._rs = rs
        threading.Thread(target=self._load_and_run, daemon=True,
                         name='yolo_engine_init').start()

    def stop(self) -> None:
        self._running = False
        for s in self._smoothers.values():
            s.reset()

    # ── 信心分門檻 ────────────────────────────────────────────────────────────

    def set_conf_threshold(self, val: float) -> None:
        """設定偵測信心分門檻（0.0–1.0），低於此值的偵測結果會被過濾掉。"""
        self._conf_threshold = max(0.0, min(1.0, float(val)))

    @property
    def conf_threshold(self) -> float:
        return self._conf_threshold

    # ── ROI API ───────────────────────────────────────────────────────────────

    def set_roi(self, cam_id: int, x1: int, y1: int,
                x2: int, y2: int) -> None:
        """設定 ROI 並存檔（committed）。"""
        roi = (int(min(x1, x2)), int(min(y1, y2)),
               int(max(x1, x2)), int(max(y1, y2)))
        with self._roi_lock:
            self._roi[cam_id] = roi
            self._preview_roi[cam_id] = None
        self._save_roi_to_file()

    def clear_roi(self, cam_id: int) -> None:
        """清除 ROI 並存檔。"""
        with self._roi_lock:
            self._roi[cam_id] = None
            self._preview_roi[cam_id] = None
        self._save_roi_to_file()

    def get_roi(self, cam_id: int) -> 'tuple|None':
        with self._roi_lock:
            return self._roi.get(cam_id)

    def set_preview_roi(self, cam_id: int, x1: int, y1: int,
                        x2: int, y2: int) -> None:
        """設定拖移預覽 ROI（不存檔，不影響過濾）。"""
        with self._roi_lock:
            self._preview_roi[cam_id] = (
                int(min(x1, x2)), int(min(y1, y2)),
                int(max(x1, x2)), int(max(y1, y2)))

    def clear_preview_roi(self, cam_id: int) -> None:
        with self._roi_lock:
            self._preview_roi[cam_id] = None

    def _load_roi_from_file(self) -> None:
        try:
            if not os.path.exists(_ROI_CONFIG):
                return
            with open(_ROI_CONFIG) as f:
                data = json.load(f)
            for cid in self._cam_ids:
                key = f'cam{cid}'
                if key in data and data[key]:
                    self._roi[cid] = tuple(data[key])
            print(f'[YoloEngine] ROI 載入：{self._roi}')
        except Exception as e:
            print(f'[YoloEngine] ROI 載入失敗：{e}')

    def _save_roi_to_file(self) -> None:
        try:
            os.makedirs(os.path.dirname(_ROI_CONFIG), exist_ok=True)
            with self._roi_lock:
                data = {f'cam{cid}': list(self._roi[cid])
                        if self._roi.get(cid) else None
                        for cid in self._cam_ids}
            with open(_ROI_CONFIG, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f'[YoloEngine] ROI 存檔失敗：{e}')

    # ── 結果存取（thread-safe）────────────────────────────────────────────────

    def get_overlay_tex(self, cam_id: int) -> 'np.ndarray | None':
        """取得最新 overlay 材質（dpg-ready float32 RGBA）。"""
        r = self._results.get(cam_id)
        if r is None:
            return None
        with r['lock']:
            return r['tex']

    def get_dets(self, cam_id: int) -> list:
        """取得最新偵測結果列表。"""
        r = self._results.get(cam_id)
        if r is None:
            return []
        with r['lock']:
            return list(r['dets'])

    def get_det_count(self, cam_id: int) -> int:
        return len(self.get_dets(cam_id))

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def load_error(self) -> 'str | None':
        return self._load_error

    # ── 內部：載入模型 ────────────────────────────────────────────────────────

    def _load_and_run(self) -> None:
        try:
            from ultralytics import YOLO
            import torch
            print(f'[YoloEngine] 載入模型 {self._model_path} ...')
            self._model = YOLO(self._model_path)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            # 暖機：空白推論讓 CUDA kernel 預先編譯
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            with _MODEL_LOCK:
                self._model(dummy, device=device, verbose=False, imgsz=640)
            self._loaded = True
            print(f'[YoloEngine] 模型就緒（device={device}），啟動 {len(self._cam_ids)} 條偵測執行緒')
            self._running = True
            for i, cam_id in enumerate(self._cam_ids):
                offset = i * (self._interval / len(self._cam_ids))
                t = threading.Thread(
                    target=self._infer_loop,
                    args=(cam_id, offset, device),
                    daemon=True,
                    name=f'yolo_cam{cam_id}'
                )
                t.start()
                self._threads.append(t)
        except Exception as e:
            self._load_error = str(e)
            print(f'[YoloEngine] 載入失敗：{e}')

    # ── 內部：推論迴圈（每台相機一條）────────────────────────────────────────

    def _infer_loop(self, cam_id: int, start_offset: float,
                    device: str) -> None:
        from src.realsense import RealSense as _RS
        time.sleep(start_offset)

        while self._running:
            t0 = time.perf_counter()

            if self._rs is None:
                time.sleep(self._interval)
                continue

            frame = self._rs.get_frame(cam_id)
            if frame is None:
                time.sleep(self._interval)
                continue

            try:
                with _MODEL_LOCK:
                    results = self._model(
                        frame, device=device, verbose=False, imgsz=640)
                with self._roi_lock:
                    roi     = self._roi.get(cam_id)
                    preview = self._preview_roi.get(cam_id)
                dets = self._parse_results(results, frame.shape, roi,
                                           self._conf_threshold)
                dets = self._smoothers[cam_id].smooth(dets)
                tex  = self._draw_overlay(frame, dets, roi, preview)
                with self._results[cam_id]['lock']:
                    self._results[cam_id]['dets'] = dets
                    self._results[cam_id]['tex']  = tex
            except Exception as e:
                print(f'[YoloEngine cam{cam_id}] 推論錯誤：{e}')

            elapsed = time.perf_counter() - t0
            wait = self._interval - elapsed
            if wait > 0:
                time.sleep(wait)

    # ── 內部：解析結果 ────────────────────────────────────────────────────────

    @staticmethod
    def _parse_results(results, frame_shape,
                       roi: 'tuple|None',
                       conf_threshold: float = 0.5) -> list:
        """將 ultralytics Results 轉為標準 det dict，並套用 ROI 過濾。"""
        dets = []
        if not results:
            return dets
        r = results[0]
        boxes = r.boxes
        if boxes is None:
            return dets
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            # ROI 過濾：中心點必須在範圍內
            if roi is not None:
                rx1, ry1, rx2, ry2 = roi
                if not (rx1 <= cx <= rx2 and ry1 <= cy <= ry2):
                    continue

            conf   = float(box.conf[0])
            if conf < conf_threshold:
                continue
            cls_id = int(box.cls[0])
            angle_deg = 0.0
            if r.masks is not None:
                try:
                    mask = r.masks.data[i].cpu().numpy()
                    angle_deg = _mask_angle(mask)
                except Exception:
                    pass

            dets.append({
                'box':       (x1, y1, x2, y2),
                'center':    (cx, cy),
                'conf':      conf,
                'cls_id':    cls_id,
                'angle_deg': angle_deg,
            })
        return dets

    # ── 內部：繪製 overlay ────────────────────────────────────────────────────

    @staticmethod
    def _draw_overlay(frame: np.ndarray, dets: list,
                      roi: 'tuple|None',
                      preview_roi: 'tuple|None') -> np.ndarray:
        """在 BGR frame 上畫偵測框 + ROI 矩形，回傳 dpg 材質。"""
        from src.realsense import RealSense as _RS
        vis = frame.copy()

        # 偵測框
        for det in dets:
            x1, y1, x2, y2 = [int(v) for v in det['box']]
            cx, cy = int(det['center'][0]), int(det['center'][1])
            color  = (0, 255, 100)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis, f'{det["conf"]:.2f}',
                        (x1, max(y1 - 4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            cv2.drawMarker(vis, (cx, cy), color,
                           cv2.MARKER_CROSS, 14, 2)

        # 已儲存的 ROI（實線綠框 + 半透明填充）
        if roi is not None:
            rx1, ry1, rx2, ry2 = [int(v) for v in roi]
            overlay = vis.copy()
            cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), (0, 255, 0), -1)
            cv2.addWeighted(overlay, 0.10, vis, 0.90, 0, vis)
            cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)
            cv2.putText(vis, 'ROI', (rx1 + 4, ry1 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 拖移預覽 ROI（虛線藍框）
        if preview_roi is not None:
            px1, py1, px2, py2 = [int(v) for v in preview_roi]
            _draw_dashed_rect(vis, px1, py1, px2, py2, (0, 180, 255), 2)

        return _RS._to_texture(vis)


# ── 工具函式 ──────────────────────────────────────────────────────────────────

def _mask_angle(mask: np.ndarray) -> float:
    """從二值 mask 估計物體長軸角度（度）。"""
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
    """在 img 上畫虛線矩形（用於 ROI 拖移預覽）。"""
    pts = [
        ((x1, y1), (x2, y1)),
        ((x2, y1), (x2, y2)),
        ((x2, y2), (x1, y2)),
        ((x1, y2), (x1, y1)),
    ]
    for (ax, ay), (bx, by) in pts:
        dist = int(((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5)
        if dist == 0:
            continue
        for i in range(0, dist, dash * 2):
            t0 = i / dist
            t1 = min((i + dash) / dist, 1.0)
            p0 = (int(ax + (bx - ax) * t0), int(ay + (by - ay) * t0))
            p1 = (int(ax + (bx - ax) * t1), int(ay + (by - ay) * t1))
            cv2.line(img, p0, p1, color, thickness, cv2.LINE_AA)
