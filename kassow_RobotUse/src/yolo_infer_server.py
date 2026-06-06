"""
yolo_infer_server.py — GPU 推論伺服器（Python 3.11，Blackwell SM 12.0）

由 yolo_detector_pt.py 以 subprocess 方式啟動。
- 影像幀：透過 multiprocessing.shared_memory 傳入（零拷貝）
- 推論結果：JSON 寫至 stdout，每行一筆

啟動參數：
    python3.11 yolo_infer_server.py <model_path> <shm_name> <H> <W> <C>

通訊協定：
    Client → Server stdin : "INFER {conf:.3f}\\n"  每幀發一次
    Server → Client stdout: json list of detections + "\\n"
    Client → Server stdin : "STOP\\n"              結束
"""

import json
import math
import sys

import cv2
import numpy as np
from multiprocessing.shared_memory import SharedMemory


def _mask_edge_orientation(mask: np.ndarray):
    """最小外接矩形長邊方向 → (centroid, angle_deg)。"""
    contours, _ = cv2.findContours(mask.astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None, 0.0
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 5:
        return None, 0.0
    (cx, cy), (w, h), rect_deg = cv2.minAreaRect(contour)
    long_deg = rect_deg + 90.0 if h >= w else rect_deg
    while long_deg >  90.0: long_deg -= 180.0
    while long_deg <= -90.0: long_deg += 180.0
    return (float(cx), float(cy)), float(long_deg)


def main():
    import torch
    from ultralytics import YOLO

    if len(sys.argv) < 6:
        sys.stderr.write('Usage: yolo_infer_server.py model_path shm_name H W C\n')
        sys.exit(1)

    model_path = sys.argv[1]
    shm_name   = sys.argv[2]
    H, W, C    = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        gpu   = torch.cuda.get_device_name(0)
        sm    = torch.cuda.get_device_capability(0)
        vram  = torch.cuda.get_device_properties(0).total_memory >> 20
        sys.stderr.write(
            f'[infer_server] GPU={gpu} SM{sm[0]}.{sm[1]} {vram}MiB '
            f'CUDA={torch.version.cuda}\n')
    else:
        sys.stderr.write('[infer_server] CUDA 不可用，使用 CPU\n')

    model = YOLO(model_path)
    model.to(device)

    # 暖身：初始化 CUDA context + PTX JIT cache（Blackwell 首次編譯較慢）
    sys.stderr.write('[infer_server] 暖身推論中...\n')
    dummy = np.zeros((H, W, C), dtype=np.uint8)
    for _ in range(3):
        model(dummy, imgsz=640, verbose=False, conf=0.30)
    sys.stderr.write('[infer_server] 暖身完成\n')
    sys.stderr.flush()

    # 掛載共享記憶體（client 已建立）
    shm = SharedMemory(name=shm_name)
    frame_buf = np.ndarray((H, W, C), dtype=np.uint8, buffer=shm.buf)

    # 通知 client 已就緒
    sys.stdout.write(f'READY device={device} cuda={torch.version.cuda}\n')
    sys.stdout.flush()

    conf_thresh = 0.30

    for line in sys.stdin:
        cmd = line.strip()
        if not cmd:
            continue
        if cmd == 'STOP':
            break

        parts = cmd.split()
        if parts[0] != 'INFER':
            continue
        if len(parts) > 1:
            try:
                conf_thresh = float(parts[1])
            except ValueError:
                pass

        # 複製當前幀（避免推論期間被覆蓋）
        frame = frame_buf.copy()

        # YOLO 推論
        results = model(frame, imgsz=640, verbose=False, conf=conf_thresh)
        result  = results[0]

        dets = []
        if result.boxes is not None and len(result.boxes) > 0:
            boxes_np = result.boxes.xyxy.cpu().numpy()
            confs_np = result.boxes.conf.cpu().numpy()
            cls_np   = result.boxes.cls.cpu().numpy().astype(int)

            raw_masks = None
            if result.masks is not None:
                raw_masks = result.masks.data.cpu().numpy()  # [N, H', W']

            for i, (box, conf_val, cls_id) in enumerate(zip(boxes_np, confs_np, cls_np)):
                # 建立 mask（縮放至幀大小）
                mask = None
                if raw_masks is not None and i < len(raw_masks):
                    m = raw_masks[i]
                    if m.shape != (H, W):
                        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)
                    mask = (m > 0.5)

                # mask → centroid + angle
                if mask is not None:
                    centroid, angle_deg = _mask_edge_orientation(mask)
                else:
                    centroid = None
                    angle_deg = 0.0

                if centroid is None:
                    x1, y1, x2, y2 = box
                    centroid  = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

                dets.append({
                    'box':       [float(v) for v in box],
                    'conf':      float(conf_val),
                    'cls_id':    int(cls_id),
                    'center':    [float(centroid[0]), float(centroid[1])],
                    'angle_deg': float(angle_deg),
                })

        sys.stdout.write(json.dumps(dets) + '\n')
        sys.stdout.flush()

    shm.close()
    sys.stderr.write('[infer_server] 結束\n')


if __name__ == '__main__':
    main()
