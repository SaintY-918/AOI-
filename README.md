AOI 瑕疵分類

本專案針對 AOI 缺陷影像進行多模型分類，整合 ViT、ConvNeXt、YOLO11-cls 三種架構進行 5-fold Ensemble 與加權融合推論。

資料集來源

由工業技術研究院於Aidea提供資料集，作為參賽者訓練分類模型

https://aidea-web.tw/topic/285ef3be-44eb-43dd-85cc-f0388bf85ea4

環境與套件
IDE：VS Code
作業系統：Ubuntu 24.04 LTS (Kernel 5.15, WSL2 on Windows 11)
Python：3.12.3
GPU : NVIDIA GeForce RTX 4070
CUDA version : 12.1
深度學習框架
PyTorch : 2.5.1
Torchvision : 0.20.1
Transformers : 4.52.4
Ultralytics YOLO : 8.3.146
資料處理：Pandas, Pillow, NumPy, Matplotlib, TQDM

