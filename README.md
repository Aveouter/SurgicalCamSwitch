# SurgicalCamSwitch

**SurgicalCamSwitch** 是一个基于深度学习的视频分析与时间序列预测系统，旨在解决开放式外科手术场景中的最佳摄像头镜头智能选择问题。  
系统融合了多种 Transformer、VAE、CNN 与时序预测模型，支持多任务训练、异常检测与可视化分析。

---

## 🚀 项目结构

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

## 🧠 功能简介

- **多模型融合**：支持 Autoformer、TimesNet、Informer、iTransformer 等 30+ 模型；
- **多任务学习**：涵盖长短期预测、异常检测、补全、分类等任务；
- **开放手术镜头选择**：支持多摄像头视频序列的最优视角预测；
- **VAE 潜空间建模**：实现基于变分自编码器的特征学习；
- **实验脚本化管理**：每个模型都配备独立 `.sh` 训练脚本，支持批量运行。

---

## 🛠️ 环境配置

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

## 📚 引用

coming soon

---

