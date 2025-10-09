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

## ⚙️ 数据与大文件管理

为避免 GitHub 大文件限制（>100MB）：

1. 历史已通过 `git filter-repo` 清理：
   ```bash
   git filter-repo --force --invert-paths --path-glob "Exp_saito_vae/*.csv"
   git filter-repo --force --strip-blobs-bigger-than 100M
   ```
2. `.gitignore` 已配置忽略：
   ```
   Exp_saito_vae/*.csv
   ```
3. 大型数据文件应存储于：
   - GitHub Release 附件；
   - 或 Git LFS。

---

## 📊 提交与推送流程

```bash
# 添加修改
git add .

# 提交更新
git commit -m "fix: unify line endings and update scripts"

# 推送到远端
git push

# 若进行了历史重写，请使用强推
git push -u origin main --force
```

---

## ⚡ 历史清理与版本记录

2025 年 10 月，项目完成历史重构以删除大型 CSV 文件，主要 commit 流程如下：
```
698be73  →  1287b75  →  ef0cd03  →  3baa127
```
- 使用 `git filter-repo` 清理历史；
- 重新添加远端；
- 强制推送；
- 修复行尾格式与脚本文件。

---

## 📚 引用

> Liu X., et al. *An Optimal Surgical Camera Selection Framework Based on Multi-View Temporal Modeling.*  
> Chongqing Institute of Green and Intelligent Technology, Chinese Academy of Sciences, 2025.

---

## 📫 联系方式

- **Author:** Xinyu Liu  
- **Affiliation:** Chongqing Institute of Green and Intelligent Technology, CAS  
- **GitHub:** [Aveouter](https://github.com/Aveouter)  
- **Email:** liuxinyu@cigit.ac.cn
