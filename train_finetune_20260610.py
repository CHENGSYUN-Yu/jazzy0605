"""
Fine-tune from best20260603.pt on surgical_instrument_seg.v1i.yolo26 dataset.
Hyperparameters mirror best20260603.pt train_args; imgsz adjusted to 640
because Roboflow pre-processed all images to 640x640.
"""

from ultralytics import YOLO
import shutil
import os

BASE_MODEL  = "kassow_RobotUse/models/best20260603.pt"
DATA_YAML   = "/home/bf-robotics/jazzy0605/surgical_instrument_seg.v1i.yolo26/data.yaml"
PROJECT     = "runs/seg_20260610"
RUN_NAME    = "yolo11m_finetune_20260610"
SAVE_DIR    = "kassow_RobotUse/models"
OUTPUT_NAME = "best_finetune_20260610.pt"

model = YOLO(BASE_MODEL)

results = model.train(
    data=DATA_YAML,
    task="segment",

    # training schedule
    epochs=30,
    patience=12,
    batch=4,
    imgsz=640,
    cache="ram",
    device=0,
    workers=8,

    # optimizer
    optimizer="SGD",
    lr0=0.0001,
    lrf=0.01,
    momentum=0.9,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    cos_lr=True,
    nbs=64,
    amp=True,

    # loss weights
    box=7.5,
    cls=0.5,
    dfl=1.5,

    # mask
    overlap_mask=True,
    mask_ratio=4,
    dropout=0.0,

    # augmentation
    hsv_h=0.015,
    hsv_s=0.5,
    hsv_v=0.4,
    degrees=60,
    translate=0.2,
    scale=0.5,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.5,
    bgr=0.0,
    mosaic=0.0,
    mixup=0.0,
    cutmix=0.0,
    copy_paste=0.0,
    copy_paste_mode="flip",
    auto_augment="randaugment",
    erasing=0.4,
    close_mosaic=10,

    # misc
    pretrained=True,
    seed=0,
    deterministic=True,
    single_cls=False,
    rect=False,
    multi_scale=0.0,
    val=True,
    plots=True,
    verbose=True,
    save=True,
    save_period=-1,

    # output
    project=PROJECT,
    name=RUN_NAME,
    exist_ok=True,
)

# copy best.pt → kassow_RobotUse/models/
src = os.path.join(PROJECT, RUN_NAME, "weights", "best.pt")
dst = os.path.join(SAVE_DIR, OUTPUT_NAME)
os.makedirs(SAVE_DIR, exist_ok=True)
shutil.copy2(src, dst)

print(f"\n[Done] Best model saved to: {dst}")
print(f"  mAP50(B)    : {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.5f}")
print(f"  mAP50-95(B) : {results.results_dict.get('metrics/mAP50-95(B)', 'N/A'):.5f}")
print(f"  mAP50(M)    : {results.results_dict.get('metrics/mAP50(M)', 'N/A'):.5f}")
print(f"  mAP50-95(M) : {results.results_dict.get('metrics/mAP50-95(M)', 'N/A'):.5f}")
