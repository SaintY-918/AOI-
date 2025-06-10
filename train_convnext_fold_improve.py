
import os
import argparse
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score
from tqdm import tqdm


#  固定隨機種子，確保實驗可重現

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


class CSVDataset(Dataset):
    def __init__(self, csv_path: str, img_root: str, transform):
        df = pd.read_csv(csv_path)
        # CSV 第一欄 ID 為檔名，第二欄 Label
        self.img_paths = df['ID'].apply(lambda x: os.path.join(img_root, x)).tolist()
        self.labels    = df['Label'].astype(int).tolist()
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        # 灰階影像 convert('RGB')→三通道複製
        img = Image.open(self.img_paths[idx]).convert('RGB')
        x   = self.transform(img)
        y   = self.labels[idx]
        return x, y


#  訓練與驗證流程

def train_and_evaluate(
    fold: int,
    seed: int,
    batch_size: int,
    epochs: int,
    lr: float,
    img_size: int,
    workers: int
):
    # 路徑設定
    img_root   = "aoi/train_images/train_images"
    train_csv  = f"aoi/tmp_fold{fold}_train.csv"
    val_csv    = f"aoi/tmp_fold{fold}_val.csv"
    output_dir = "runs/convnext_aoi"
    os.makedirs(output_dir, exist_ok=True)

    
    #  定義 train / val 兩套 transforms
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0), ratio=(1.0, 1.0)),
        transforms.RandomRotation(15),
        transforms.RandomAdjustSharpness(4.0, p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5, 0.5, 0.5],  
            [0.5, 0.5, 0.5]
        ),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5]
        ),
    ])

   
    train_ds = CSVDataset(train_csv, img_root, train_transform)
    val_ds   = CSVDataset(val_csv,   img_root, val_transform)

    # 可重現的 shuffle
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        generator=g,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True
    )

   
    num_classes = len(set(train_ds.labels))
    model = convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT)

    # 取出原始 classifier，插入 Dropout
    orig_cls = model.classifier  # LayerNorm → Flatten → Linear
    in_feats = orig_cls[2].in_features

    model.classifier = nn.Sequential(
        orig_cls[0],            # LayerNorm
        nn.Dropout(p=0.1),      # 新增 Dropout(p=0.1)
        orig_cls[1],            # Flatten
        nn.Linear(in_feats, num_classes)
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

 
    #  Optimizer / Scheduler / Loss 
  
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        # Training 
        model.train()
        running_loss = 0.0
        train_bar = tqdm(
            train_loader,
            desc=f"[Fold {fold}] Ep{epoch}/{epochs} Train",
            leave=False
        )
        for x, y in train_bar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss   = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_train_loss = running_loss / len(train_ds)
        print(f"[Fold {fold}] Epoch {epoch} 平均 Train Loss: {avg_train_loss:.4f}")

        # Validation 
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                preds = model(x).argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y.numpy())
        acc = accuracy_score(all_labels, all_preds)
        print(f"[Fold {fold}] Epoch {epoch}  Val Acc: {acc:.4f}")

        # Save best 
        if acc > best_acc:
            best_acc = acc
            save_path = os.path.join(output_dir, f"best_model_fold{fold}.pth")
            torch.save(model.state_dict(), save_path)

    print(f"[Fold {fold}] Best Val Acc: {best_acc:.4f}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fold", type=int, required=True,
        help="指定要跑的 fold 編號 (0 ~ 4)"
    )
    args = parser.parse_args()
    FOLD = args.fold

    # 超參數
    SEED       = 42
    BATCH_SIZE = 16
    EPOCHS     = 10
    LR         = 1e-4
    #LR         = 7e-5
    IMG_SIZE   = 224
    WORKERS    = 4

    # 初始化隨機種子
    seed_everything(SEED)

    # 執行訓練與驗證
    train_and_evaluate(FOLD, SEED, BATCH_SIZE, EPOCHS, LR, IMG_SIZE, WORKERS)
