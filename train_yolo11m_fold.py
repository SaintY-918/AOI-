
import os
import shutil
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from ultralytics import YOLO


BASE_DIR    = os.path.abspath(os.path.dirname(__file__))
DATA_ROOT   = os.path.join(BASE_DIR, 'aoi')
SRC_IMG     = os.path.join(DATA_ROOT, 'train_images', 'train_images')
CSV_PATH    = os.path.join(DATA_ROOT, 'train.csv')
PROJECT_DIR = os.path.join(BASE_DIR, 'runs', 'yolo11m_aoi')
MODEL_BASE  = 'yolo11m-cls.pt'
FOLDS       = 5
EPOCHS      = 25
IMGSZ       = 224
BATCH       = 32
WORKERS     = 8
SEED        = 42
PATIENCE    = 5
DROPOUT_P   = 0.1

# Augmentation params
AUGMENT_PARAMS = {
    'augment'      : True,
    'hsv_h'        : 0.0,
    'hsv_s'        : 0.0,
    'hsv_v'        : 0.2,
    'degrees'      : 10,
    'translate'    : 0.1,
    'scale'        : 0.4,
    'fliplr'       : 0.0,
    'flipud'       : 0.0,
    'mosaic'       : 0.0,
    'mixup'        : 0.0,
    'cutmix'       : 0.0,
    'auto_augment' : None,
    'erasing'      : 0.0,
}


# Prepare splits & folders (once)

df = pd.read_csv(CSV_PATH)  # columns: ID, Label
X  = df['ID'].values
y  = df['Label'].values
num_classes = df['Label'].nunique()
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    fold_dir  = os.path.join(PROJECT_DIR, f'yolo11m_fold{fold}')
    # remove old and create train/val/class directories
    if os.path.exists(fold_dir):
        shutil.rmtree(fold_dir)
    for split, idxs in [('train', train_idx), ('val', val_idx)]:
        for cls in sorted(df['Label'].astype(str).unique()):
            os.makedirs(os.path.join(fold_dir, split, cls), exist_ok=True)
        for idx in idxs:
            fn = df.iloc[idx]['ID']
            cls = str(df.iloc[idx]['Label'])
            src = os.path.join(SRC_IMG, fn)
            dst = os.path.join(fold_dir, split, cls, fn)
            os.symlink(src, dst)


# 5-Fold Training Loop

val_acc_list = []
for fold in range(FOLDS):
    fold_dir = os.path.join(PROJECT_DIR, f'yolo11m_fold{fold}')
    print(f"\n=== Fold {fold} training ===")
    #  載入模型
    model = YOLO(MODEL_BASE)
    #  插入 Dropout 在最後的 classification head 中
    head = model.model.model[-1]
    if hasattr(head, 'linear') and isinstance(head.linear, nn.Linear):
        in_feats = head.linear.in_features
        head.linear = nn.Sequential(
            nn.Dropout(p=DROPOUT_P),
            nn.Linear(in_feats, num_classes)
        )
    else:
        raise RuntimeError("找不到 head.linear，無法插入 Dropout")

    #  訓練
    model.train(
        data          = fold_dir,
        epochs        = EPOCHS,
        imgsz         = IMGSZ,
        batch         = BATCH,
        lr0           = 1e-3,
        # lrf           = 0.01,
        lrf           = 0.005,
        warmup_epochs = 5,
        warmup_bias_lr= 0.1,
        warmup_momentum=0.8,
        cos_lr        = True,
        optimizer     = 'AdamW',
        amp           = True,
        half          = True,
        workers       = WORKERS,
        project       = PROJECT_DIR,
        name          = f'yolo11m_fold{fold}',
        exist_ok      = True,
        seed          = SEED,
        patience      = PATIENCE,
        **AUGMENT_PARAMS
    )

    #  驗證並記錄 Accuracy
    val_metrics = model.val(
        data    = fold_dir,
        batch   = BATCH,
        half    = True,
        workers = WORKERS
    )
    top1 = val_metrics.top1
    print(f"Fold {fold} Val Top-1: {top1:.4f}, Top-5: {val_metrics.top5:.4f}")
    val_acc_list.append(top1)

    #  複製最佳權重
    best_src = os.path.join(PROJECT_DIR, f'yolo11m_fold{fold}', 'weights', 'best.pt')
    best_dst = os.path.join(PROJECT_DIR, f'best_yolo11m_fold{fold}.pt')
    shutil.copy2(best_src, best_dst)


# Cross-Validation Summary

print("\n===== Cross-Validation Summary =====")
for fold, acc in enumerate(val_acc_list):
    print(f"Fold {fold}: Val Top-1 Accuracy = {acc:.4f}")
avg_acc = sum(val_acc_list) / len(val_acc_list)
print(f"Average Val Top-1 Accuracy over {FOLDS} folds = {avg_acc:.4f}")
