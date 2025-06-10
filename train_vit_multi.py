"""

三分支 ViT 訓練腳本：針對 AOI 灰階圖像，整合三種視覺變因
1. trans_base  ：基本幾何旋轉
2. trans_sharp ：邊緣強化
3. trans_zoom  ：隨機縮放區塊

"""

import os
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from transformers import ViTFeatureExtractor, ViTModel, get_scheduler
from tqdm import tqdm


# 全域參數設定

DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_CLASSES   = 6
BATCH_SIZE    = 16
NUM_EPOCHS    = 7
LEARNING_RATE = 1.5e-4
NUM_WORKERS   = 4

IMG_DIR       = "aoi/train_images/train_images"
CSV_PATH      = "aoi/train.csv"
OUTPUT_WEIGHTS= "vit_multi3_1.pth"
FE_NAME       = "google/vit-base-patch16-224"


# Dataset 定義（三分支）

class AOI_Dataset3(Dataset):
    def __init__(self, img_dir, csv_path, feature_extractor, transform1, transform2, transform3):
        self.img_dir = img_dir
        self.df = pd.read_csv(csv_path)
        self.feature_extractor = feature_extractor
        self.transform1 = transform1
        self.transform2 = transform2
        self.transform3 = transform3

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        fn = self.df.iloc[idx, 0]
        label = int(self.df.iloc[idx, 1])
        img_path = os.path.join(self.img_dir, fn)
        img = Image.open(img_path).convert("RGB")

        x1 = self.feature_extractor(self.transform1(img), return_tensors="pt")["pixel_values"].squeeze(0)
        x2 = self.feature_extractor(self.transform2(img), return_tensors="pt")["pixel_values"].squeeze(0)
        x3 = self.feature_extractor(self.transform3(img), return_tensors="pt")["pixel_values"].squeeze(0)

        return x1, x2, x3, label


# 模型定義（三 ViT 串接）

class AOI_ViT_Multi3(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.vit1 = ViTModel.from_pretrained(FE_NAME)
        self.vit2 = ViTModel.from_pretrained(FE_NAME)
        self.vit3 = ViTModel.from_pretrained(FE_NAME)
        hidden = self.vit1.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden * 3, num_classes)
        )

    def forward(self, x1, x2, x3):
        out1 = self.vit1(pixel_values=x1).pooler_output
        out2 = self.vit2(pixel_values=x2).pooler_output
        out3 = self.vit3(pixel_values=x3).pooler_output
        cat = torch.cat([out1, out2, out3], dim=1)
        return self.classifier(cat)


#  訓練主程式

def main():
    fe = ViTFeatureExtractor.from_pretrained(FE_NAME, do_rescale=False)

    transform_base = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(15),
        transforms.ToTensor()
    ])

    transform_sharp = transforms.Compose([
        transforms.RandomAdjustSharpness(4.0, p=1.0),
        transforms.Resize((224, 224)),
        transforms.RandomRotation(15),
        transforms.ToTensor()
    ])

    transform_zoom = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0), ratio=(1.0, 1.0)),
        transforms.RandomRotation(15),
        transforms.ToTensor()
    ])

    train_ds = AOI_Dataset3(IMG_DIR, CSV_PATH, fe, transform_base, transform_sharp, transform_zoom)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

    model = AOI_ViT_Multi3(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=NUM_EPOCHS * len(train_loader))

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        correct = total = 0
        running_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch}/{NUM_EPOCHS}]", leave=True)
        for x1, x2, x3, labels in loop:
            x1 = x1.to(DEVICE)
            x2 = x2.to(DEVICE)
            x3 = x3.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(x1, x2, x3)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            running_loss += loss.item() * x1.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += x1.size(0)
            loop.set_postfix(loss=loss.item())

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(f"Epoch {epoch}: Loss={epoch_loss:.4f}, Accuracy={epoch_acc:.4f}")

        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict()
        }, f"checkpoint_epoch{epoch}.pth")

    torch.save(model.state_dict(), OUTPUT_WEIGHTS)
    print(f"\n✅ 訓練完成，模型已儲存至 {OUTPUT_WEIGHTS}")

if __name__ == "__main__":
    main()
