# AOI 瑕疵分類專案

本專案針對 AOI 缺陷影像進行多模型分類，整合ViT、ConvNeXt、YOLOv11-cls三種架構，進行5-fold Ensemble 與加權融合推論，以提升分類準確率。

## 資料集來源

資料集由工業技術研究院於 Aidea 平台提供，作為參賽者訓練分類模型之用。

🔗 [Aidea 資料集連結](https://aidea-web.tw/topic/285ef3be-44eb-43dd-85cc-f0388bf85ea4)

## 環境與套件

- **作業系統**：Ubuntu 24.04 LTS (Kernel 5.15, WSL2 on Windows 11)

- **Python 版本**：3.12.3

- **GPU**：NVIDIA GeForce RTX 4070

- **CUDA 版本**：12.1

### 套件版本

| 套件名稱         | 版本       |
|------------------|------------|
| PyTorch          | 2.5.1      |
| Torchvision      | 0.20.1     |
| Transformers     | 4.52.4     |
| Ultralytics YOLO | 8.3.146    |


---

## 資料集描述


本專案使用的資料集包含 AOI 缺陷影像，影像格式為 PNG，解析度統一為 **512x512** 像素。

- **訓練資料**：共 **2,528 張影像**
- **測試資料**：共 **10,142 張影像**
- **影像類別**：共 **6 類**
  - 1 類為正常影像（Normal）
  - 5 類為瑕疵影像（Defect Types）

## 資料增強




## 模型架構與融合策略


## 執行方式

```bash
# 安裝依賴
pip install -r requirements.txt

# 執行推論
python inference.py --config config.yaml

