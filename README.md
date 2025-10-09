# SurgicalCamSwitch

**SurgicalCamSwitch** is a deep learning–based video analysis and time-series prediction framework designed for **optimal surgical camera selection** in open surgery environments.  
It integrates multiple architectures including **Transformer**, **VAE**, and **CNN** models to perform **multi-task learning**, **anomaly detection**, and **visual analytics** of multi-view surgical videos.

---

## 🖼️ System Overview

<p align="center">
  <img src="figs/system_overview.png" alt="System Overview of SurgicalCamSwitch" width="700"/>
</p>

**Figure 1.** Overall architecture of the *SurgicalCamSwitch* framework.  
The system captures synchronized multi-view surgical videos, extracts visual and temporal features, and predicts the optimal viewpoint through a transformer-based temporal modeling network.

---

## 🚀 Features

- **Multi-Model Integration:** Supports 30+ deep time-series architectures (Autoformer, TimesNet, Informer, iTransformer, etc.).  
- **Multi-Task Learning:** Handles long-term forecasting, anomaly detection, imputation, and classification tasks.  
- **VAE Representation Learning:** Enables latent-space modeling for surgical scene feature reconstruction.  
- **Automated Experiment Scripts:** Each model provides individual `.sh` scripts for reproducible experiments.  
- **Open Surgery Optimization:** Learns to predict the best surgical camera viewpoint under dynamic operating conditions.

---

## 🚀 item menu

```
SurgicalCamSwitch/
├── Exp_saito_vae/              # VAE 实验与训练脚本
│   ├── VAE.py
│   ├── loader.py
│   ├── train_vae.py
│   ├── train_TIME.py
│   ├── ego_creater.py
│   ├── val.py / val.sh
│   └── vae.sh
│
├── data_provider/              # 数据接口模块
│   ├── data_factory.py
│   ├── data_loader.py
│   └── m4.py / uea.py
│
├── exp/                        # 任务实验框架
│   ├── exp_long_term_forecasting.py
│   ├── exp_short_term_forecasting.py
│   ├── exp_classification.py
│   ├── exp_imputation.py
│   ├── exp_anomaly_detection.py
│   └── exp_basic.py
│
├── models/                     # 主体模型库
│   ├── Transformer.py / Autoformer.py / Crossformer.py
│   ├── TimesNet.py / TimeMixer.py / iTransformer.py
│   ├── VAE.py / Lstm.py / TCN.py / Res_BiLstm.py
│   ├── Informer.py / PatchTST.py / Nonstationary_Transformer.py
│   └── ...（共 30+ 模型）
│
├── layers/                     # 通用层结构模块
│   ├── AutoCorrelation.py / FourierCorrelation.py
│   ├── Transformer_EncDec.py / Conv_Blocks.py
│   ├── Embed.py / StandardNorm.py
│   └── Pyraformer_EncDec.py / MultiWaveletCorrelation.py
│
├── scripts/                    # 训练与测试脚本
│   ├── camera_script/          # 各模型实验脚本（开放手术镜头任务）
│   │   ├── TimesNet.sh / Transformer.sh / Autoformer.sh ...
│   │   ├── Ablation_yolov5s.sh / Ablation_yolov12.sh ...
│   │   └── VAE.sh / TCNandLSTM.sh / PatchTST.sh
│
├── utils/                      # 工具库
│   ├── metrics.py / losses.py / masking.py
│   ├── dtw.py / dtw_metric.py / timefeatures.py
│   ├── evaluate_classification.py
│   └── print_args.py / tools.py / augmentation.py
│
├── figs/                       # 结果可视化
│   ├── Accuracy_MVOR.pdf
│   ├── Accuracy_Seq_Surg.pdf
│   ├── PRF1_by_method.pdf
│   └── conuts.py / randar.py
│
└── .gitignore                  # 忽略大文件与缓存配置
```

---

## 🛠️ environment configs

```bash
# 创建环境
conda create -n surgicalcam python=3.10
conda activate surgicalcam

# 安装依赖
pip install torch torchvision torchaudio
pip install numpy pandas scikit-learn matplotlib tqdm
pip install git-filter-repo
```

---

## 📚 cite

coming soon

---

