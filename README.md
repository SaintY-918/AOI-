# AOI 瑕疵分類專案

本專案針對 AOI 缺陷影像進行多模型分類，整合 **ViT**、**ConvNeXt**、**YOLOv11-cls** 三種架構，進行 **5-fold Ensemble** 與 **加權融合推論**，以提升分類準確率。

## 📂 資料集來源

資料集由工業技術研究院於 Aidea 平台提供，作為參賽者訓練分類模型之用。

🔗 [Aidea 資料集連結](https://aidea-web.tw/topic/285ef3be-44eb-43dd-85cc-f0388bf85u 24.04 LTS (Kernel 5.15, WSL2 on Windows 11)
- **Python 版本**：3.12.3
- **GPU**：NVIDIA GeForce RTX 4070
- **CUDA 版本**：12.1

### 📦 主要套件版本

| 套件名稱         | 版本       |
|------------------|------------|
| PyTorch          | 2.5.1      |
| Torchvision      | 0.20.1     |
| Transformers     | 4.52.4     |
| Ultralytics YOLO | 8.3.146    |
| Pandas           | 最新版     |
| Pillow           | 最新版     |
| NumPy            | 最新版     |
| Matplotlib       | 最新版     |
| TQDM             | 最新版     |

---

你也可以加入以下區塊來補充：

## 📊 模型架構與融合策略

（簡要說明 ViT、ConvNeXt、YOLOv11-cls 的特點與融合方式）

## 🚀 執行方式

```bash
# 安裝依賴
pip install -r requirements.txt

# 執行推論
python inference.py --config config.yaml

