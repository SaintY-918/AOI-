#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_ensemble.py

說明：針對測試集做推論，使用：
  - 5 折 ConvNeXt 模型 ensemble
  - 三輸入 ViT Multi3 模型
  - 5 折 YOLO11m-cls 模型 ensemble
並以指定權重加權融合

權重參數：
  W_CONV + W_VIT + W_YOLO = 1.0

輸入：test.csv（第一欄為 ID），圖片根目錄 img_root
輸出：ensemble_results.csv
"""

import os
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
from torchvision.transforms.functional import adjust_sharpness
from transformers import ViTFeatureExtractor, ViTModel
from ultralytics import YOLO


W_CONV      = 0.2
W_VIT       = 0.6
W_YOLO      = 0.2
NUM_CLASSES = 6
BATCH_SIZE  = 16
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 路徑設定
TEST_CSV   = "/home/saint/projects/aoi_classification/aoi/test.csv"
IMG_ROOT   = "/home/saint/projects/aoi_classification/aoi/test_images/test_images"
CONV_DIR   = "runs/convnext_aoi"
VIT_CKPT   = "vit_multi3_1.pth"
YOLO_DIR   = "runs/yolo11m_aoi_1"
OUTPUT_CSV = "ensemble_results.csv"


def custom_collate_fn(batch):
    fns, convs, x1s, x2s, x3s, raws = zip(*batch)
    convs = torch.stack(convs, dim=0)
    x1s   = torch.stack(x1s,   dim=0)
    x2s   = torch.stack(x2s,   dim=0)
    x3s   = torch.stack(x3s,   dim=0)
    return list(fns), convs, x1s, x2s, x3s, list(raws)


class EnsembleTestDataset(Dataset):
    def __init__(self, csv_path, img_root, conv_tf, vit_tfs, fe):
        self.df = pd.read_csv(csv_path)
        self.img_root = img_root
        self.conv_tf = conv_tf
        self.vit_tfs = vit_tfs
        self.fe = fe

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        fn = self.df.iloc[idx, 0]
        path = os.path.join(self.img_root, fn)
        img  = Image.open(path).convert('RGB')

        # ConvNeXt tensor
        x_conv = self.conv_tf(img)

        # ViT 三支前處理後再轉 pixel_values
        x1 = self.fe(self.vit_tfs[0](img), return_tensors='pt')["pixel_values"].squeeze(0)
        x2 = self.fe(self.vit_tfs[1](img), return_tensors='pt')["pixel_values"].squeeze(0)
        x3 = self.fe(self.vit_tfs[2](img), return_tensors='pt')["pixel_values"].squeeze(0)

        return fn, x_conv, x1, x2, x3, img


# ConvNeXt Base 

def build_convnext(dropout=0.1):
    m = convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT)
    orig = m.classifier
    in_feats = orig[2].in_features
    m.classifier = nn.Sequential(
        orig[0],
        nn.Dropout(dropout),
        orig[1],
        nn.Linear(in_feats, NUM_CLASSES)
    )
    return m.to(DEVICE).eval()


# ViT Multi3 模型

class AOI_ViT_Multi3(nn.Module):
    def __init__(self, dropout=0.1, pretrained="google/vit-base-patch16-224"):
        super().__init__()
        self.vit1 = ViTModel.from_pretrained(pretrained)
        self.vit2 = ViTModel.from_pretrained(pretrained)
        self.vit3 = ViTModel.from_pretrained(pretrained)
        hidden = self.vit1.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden * 3, NUM_CLASSES)
        )

    def forward(self, x1, x2, x3):
        o1 = self.vit1(pixel_values=x1).pooler_output
        o2 = self.vit2(pixel_values=x2).pooler_output
        o3 = self.vit3(pixel_values=x3).pooler_output
        cat = torch.cat([o1, o2, o3], dim=1)
        return self.classifier(cat)


# YOLO 5 折集成推論 

def predict_yolo_fold_ensemble(models, img_path, imgsz=224):
    probs = []
    for m in models:
        res = m.predict(source=img_path, imgsz=imgsz, device=DEVICE, verbose=False)
        probs.append(res[0].probs.data.to(DEVICE))
    return torch.stack(probs, dim=0).mean(dim=0)


def main():
    # 路徑檢查
    for p in [TEST_CSV, IMG_ROOT, CONV_DIR, VIT_CKPT, YOLO_DIR]:
        assert os.path.exists(p), f"找不到路徑: {p}"

    # Transforms & Feature Extractor
    fe = ViTFeatureExtractor.from_pretrained(
        "google/vit-base-patch16-224", do_rescale=False
    )
    # ConvNeXt 前處理
    conv_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])
    # ViT 原 evaluate_vit_multi3.py 的三支前處理
    resize_tensor = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    def sharpen_tf(img):
        return adjust_sharpness(img, 4.0)
    def center_crop256(img):
        return transforms.CenterCrop(224)(transforms.Resize(256)(img))
    vit_tfs = [
        lambda img: resize_tensor(img),
        lambda img: resize_tensor(sharpen_tf(img)),
        lambda img: resize_tensor(center_crop256(img))
    ]

    # DataLoader
    ds = EnsembleTestDataset(TEST_CSV, IMG_ROOT, conv_tf, vit_tfs, fe)
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )

    #  載入 ConvNeXt 5 折
    conv_models = [build_convnext() for _ in range(5)]
    for i, m in enumerate(conv_models):
        ckpt = os.path.join(CONV_DIR, f"best_model_fold{i}.pth")
        m.load_state_dict(torch.load(ckpt, map_location=DEVICE))

    #  載入 ViT Multi3
    vit = AOI_ViT_Multi3().to(DEVICE).eval()
    vit.load_state_dict(torch.load(VIT_CKPT, map_location=DEVICE))

    #  載入 YOLO11m-cls 5 折
    yolo_models = []
    for i in range(5):
        path = os.path.join(YOLO_DIR, f"best_yolo11m_fold{i}.pt")
        model = YOLO(path, task="classify")
        model.model.to(DEVICE).eval()
        yolo_models.append(model)

    #  推論 & 加權融合
    results = []
    with torch.no_grad():
        for fn, x_conv, x1, x2, x3, raw in tqdm(loader, desc="Ensemble"):
            x_conv = x_conv.to(DEVICE)
            x1, x2, x3 = x1.to(DEVICE), x2.to(DEVICE), x3.to(DEVICE)

            # ConvNeXt ensemble
            p_conv = torch.stack(
                [F.softmax(m(x_conv), dim=1) for m in conv_models],
                dim=0
            ).mean(dim=0)

            # ViT prediction
            p_vit = F.softmax(vit(x1, x2, x3), dim=1)

            # YOLO ensemble
            p_yolo = torch.stack(
                [predict_yolo_fold_ensemble(yolo_models, os.path.join(IMG_ROOT, img_id))
                 for img_id in fn],
                dim=0
            )

            # 加權融合
            p_final = W_CONV * p_conv + W_VIT * p_vit + W_YOLO * p_yolo
            preds = p_final.argmax(dim=1).cpu().tolist()

            # 顯示與儲存
            for idx, img_id in enumerate(fn):
                print(f"[Inference] {img_id}")
                print(f"  ConvNeXt: {p_conv[idx].cpu().tolist()}")
                print(f"  ViT     : {p_vit[idx].cpu().tolist()}")
                print(f"  YOLO    : {p_yolo[idx].cpu().tolist()}")
                print(f"  Ensemble: {p_final[idx].cpu().tolist()}")
                print(f"  Pred    : {preds[idx]}\n")
                results.append((img_id, preds[idx]))

    #  輸出 CSV
    pd.DataFrame(results, columns=["ID", "label"]).to_csv(OUTPUT_CSV, index=False)
    print(f"✅ 推論完成，結果寫入 {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
