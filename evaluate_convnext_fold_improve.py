
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

class AOITestDataset(Dataset):
    def __init__(self, csv_path, img_root, transform):
        self.df = pd.read_csv(csv_path)
        self.img_root = img_root
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx, 0]
        img_path = os.path.join(self.img_root, img_name)
        img = Image.open(img_path).convert('RGB')
        x = self.transform(img)
        return img_name, x

def build_model(num_classes=6, dropout_p=0.1, device='cpu'):
    model = convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT)
    orig_cls = model.classifier
    in_feats = orig_cls[2].in_features
    model.classifier = nn.Sequential(
        orig_cls[0],            # LayerNorm
        nn.Dropout(p=dropout_p),
        orig_cls[1],            # Flatten
        nn.Linear(in_feats, num_classes)
    )
    model.to(device)
    model.eval()
    return model

def main():
    # 預設參數
    TEST_CSV   = "/home/saint/projects/aoi_classification/aoi/test.csv"
    IMG_ROOT   = "/home/saint/projects/aoi_classification/aoi/test_images/test_images"
    CKPT_DIR   = "runs/convnext_aoi"
    OUTPUT_CSV = "convnext_ensemble.csv"
    BATCH_SIZE = 16
    DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

    # 確認路徑
    assert os.path.isfile(TEST_CSV), f"找不到 test_csv: {TEST_CSV}"
    assert os.path.isdir(IMG_ROOT), f"找不到 img_root: {IMG_ROOT}"
    assert os.path.isdir(CKPT_DIR), f"找不到 ckpt_dir: {CKPT_DIR}"

    device = torch.device(DEVICE)

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5]
        ),
    ])

    # Dataset & DataLoader
    dataset = AOITestDataset(TEST_CSV, IMG_ROOT, transform)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    
    # 載入 5 個 fold 的模型
    
    models = []
    for i in range(5):
        ckpt_path = os.path.join(CKPT_DIR, f"best_model_fold{i}.pth")
        assert os.path.isfile(ckpt_path), f"找不到 {ckpt_path}"
        m = build_model(num_classes=6, dropout_p=0.1, device=device)
        state = torch.load(ckpt_path, map_location=device)
        m.load_state_dict(state)
        models.append(m)

    
    # 推論：softmax 機率平均 ensemble

    results = []
    with torch.no_grad():
        for img_names, imgs in tqdm(loader, desc="Ensemble Inference"):
            imgs = imgs.to(device)
            # 收集各 fold 的機率
            probs = []
            for m in models:
                logits = m(imgs)                  # [B, C]
                probs.append(F.softmax(logits, dim=1))
            # 平均
            prob_mean = torch.stack(probs, dim=0).mean(dim=0)  # [B, C]
            preds = prob_mean.argmax(dim=1).cpu().tolist()

            for name, p in zip(img_names, preds):
                results.append((name, p))

    
    # 寫入 CSV
    
    df_out = pd.DataFrame(results, columns=["ID", "label"])
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ 推論完成，結果已寫入 {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
