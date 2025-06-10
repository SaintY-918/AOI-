
import os
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from transformers import ViTFeatureExtractor, ViTModel
from tqdm import tqdm


#  全域參數設定

DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_CLASSES  = 6
BATCH_SIZE   = 16
NUM_WORKERS  = 4

TEST_CSV     = "/home/saint/projects/aoi_classification/aoi/test.csv"
TEST_IMG_DIR = "/home/saint/projects/aoi_classification/aoi/test_images/test_images"
WEIGHTS_PATH = "vit_multi3_1.pth"
FE_NAME      = "google/vit-base-patch16-224"

#  Dataset（三分支測試）

def deterministic_sharpen(img: Image.Image, factor: float=4.0):
    return transforms.functional.adjust_sharpness(img, factor)

def center_resized_crop(img: Image.Image, size=224):
    return transforms.CenterCrop(size)(transforms.Resize(256)(img))

class AOI_TestDataset3(Dataset):
    def __init__(self, img_dir, csv_path, feature_extractor):
        self.img_dir = img_dir
        self.df = pd.read_csv(csv_path)
        self.filenames = self.df.iloc[:, 0].tolist()
        self.feature_extractor = feature_extractor
        self.resize_tensor = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fn = self.filenames[idx]
        img_path = os.path.join(self.img_dir, fn)
        img = Image.open(img_path).convert("RGB")

        # 分支1：基本
        x1 = self.feature_extractor(self.resize_tensor(img), return_tensors="pt")["pixel_values"].squeeze(0)

        # 分支2：銳化
        sharpened = deterministic_sharpen(img, factor=4.0)
        x2 = self.feature_extractor(self.resize_tensor(sharpened), return_tensors="pt")["pixel_values"].squeeze(0)

        # 分支3：模擬縮放裁切（使用 CenterCrop）
        zoomed = center_resized_crop(img)
        x3 = self.feature_extractor(self.resize_tensor(zoomed), return_tensors="pt")["pixel_values"].squeeze(0)

        return fn, x1, x2, x3


#  模型定義（三分支）

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
        return self.classifier(torch.cat([out1, out2, out3], dim=1))


#  推論流程

def main():
    print("TEST_IMG_DIR:", TEST_IMG_DIR)
    fe = ViTFeatureExtractor.from_pretrained(FE_NAME, do_rescale=False)

    test_ds = AOI_TestDataset3(TEST_IMG_DIR, TEST_CSV, fe)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model = AOI_ViT_Multi3(num_classes=NUM_CLASSES).to(DEVICE)
    state_dict = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    all_fns = []
    all_preds = []

    with torch.no_grad():
        loop = tqdm(test_loader, desc="推論中", total=len(test_loader))
        for fns, x1, x2, x3 in loop:
            x1 = x1.to(DEVICE)
            x2 = x2.to(DEVICE)
            x3 = x3.to(DEVICE)
            logits = model(x1, x2, x3)
            preds = logits.softmax(dim=1).argmax(dim=1).cpu().tolist()

            all_fns.extend(fns)
            all_preds.extend(preds)

    df = pd.read_csv(TEST_CSV)
    df["Label"] = all_preds
    out_csv = os.path.join(os.path.dirname(TEST_CSV), "vit_multi3_1.csv")
    df.to_csv(out_csv, index=False, encoding='utf-8')
    print(f"\n✅ Submission 已輸出至：{out_csv}")

if __name__ == "__main__":
    main()
