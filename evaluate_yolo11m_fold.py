
import os
import pandas as pd
import numpy as np
import torch
from ultralytics import YOLO
from tqdm import tqdm


# Config

BASE_DIR    = os.path.abspath(os.path.dirname(__file__))
PROJECT_DIR = os.path.join(BASE_DIR, 'runs', 'yolo11m_aoi')
FOLDS       = 5
IMGSZ       = 224
BATCH       = 32
WORKERS     = 8

TEST_CSV    = "/home/saint/projects/aoi_classification/aoi/test.csv"
IMG_ROOT    = "/home/saint/projects/aoi_classification/aoi/test_images/test_images"


# Ensemble Prediction Function

def predict_ensemble(models, img_path):
    """
    使用多個模型的平均機率，對單張影像做集成預測。
    回傳平均機率 numpy 陣列。
    """
    fold_probs = []
    for model in models:
        results = model.predict(
            source    = img_path,
            imgsz     = IMGSZ,
            batch     = BATCH,
            workers   = WORKERS,
            half      = True,
            verbose   = False
        )
        # results[0].probs 回傳 Probs 物件，底層存放 torch.Tensor
        probs_obj = results[0].probs
        # 提取底層 tensor 或 numpy array
        if hasattr(probs_obj, 'data') and isinstance(probs_obj.data, torch.Tensor):
            raw = probs_obj.data
        elif isinstance(probs_obj, torch.Tensor):
            raw = probs_obj
        else:
            raw = np.array(probs_obj)
        # 轉成 numpy
        raw = raw.cpu().numpy() if isinstance(raw, torch.Tensor) else raw
        fold_probs.append(raw)

    # 對各 fold 機率做平均
    avg_probs = np.mean(fold_probs, axis=0)
    return avg_probs


def main():
    #  讀取測試集 CSV
    df_test = pd.read_csv(TEST_CSV, dtype={'ID': str})
    if 'Label' not in df_test.columns:
        df_test['Label'] = -1

    #  載入各折最佳權重
    models = []
    for fold in range(FOLDS):
        weight_path = os.path.join(PROJECT_DIR, f'best_yolo11m_fold{fold}.pt')
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"找不到權重檔: {weight_path}")
        models.append(YOLO(weight_path))

    #  對每張影像做集成推論，並顯示進度條
    preds = []
    for _, row in tqdm(df_test.iterrows(), total=len(df_test), desc="推論進度"):
        img_id   = row['ID']
        img_path = os.path.join(IMG_ROOT, img_id)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"找不到影像檔: {img_path}")

        # 單張影像集成預測
        avg_probs  = predict_ensemble(models, img_path)
        pred_label = int(np.argmax(avg_probs))
        preds.append(pred_label)

    #  寫回並輸出結果
    df_test['Label'] = preds
    output_csv = os.path.join(os.path.dirname(TEST_CSV), 'test_pred_with_progress.csv')
    df_test.to_csv(output_csv, index=False)
    print(f"推論完成，結果已輸出至 {output_csv}")

if __name__ == "__main__":
    main()
