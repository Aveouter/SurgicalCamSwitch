# SurgicalCamSwitch

**SurgicalCamSwitch** is a deep learning–based video analysis and time-series prediction framework designed for **optimal surgical camera selection** in open surgery environments.  
It integrates multiple architectures including **Transformer**, **VAE**, and **CNN** models to perform **multi-task learning**, **anomaly detection**, and **visual analytics** of multi-view surgical videos.

---

🎉 News

It gives me great pleasure to inform you that our manuscript “TSP-OCS: A Time-Series Prediction for Optimal Camera Selection in Multi-Viewpoint Surgical Video Analysis” has been accepted for publication in the IEEE Journal of Biomedical and Health Informatics (J-BHI).

We would like to express our sincere gratitude to all contributors and reviewers for their valuable support!

## Overview
<p align="center">
  <img src="figs/Graph Abstract.jpg " alt="Graph Abstract" width="700"/>
</p>

## Abstract
Recording opensurgery procedures is essen
tial for educational and clinical evaluation purposes; how
ever, traditional single-camera methods often face chal
lenges such as occlusions caused by the surgeon’s head
 and body, as well as limitations due to fixed camera angles,
 which undermine the comprehensibility of the recorded
 surgical content. In this study, we specifically focus on
 open thyroidectomy and employ a multi-viewpoint camera
 recording setup, in which six synchronized cameras cap
ture the surgery from different angles simultaneously. We
 develop a supervised time-series prediction framework to
 automatically select the most informative camera views,
 ensuring better coverage of critical steps.
 Our model forecasts camera selections by extracting
 and fusing visual and semantic features from thyroidec
tomy videos using pre-trained models, followed by tempo
ral modeling with TimeBlocks. We constructed a dataset
 of five thyroidectomy procedures with synchronized six
view recordings and conducted experiments. The results
 show that our method achieves stable accuracy compared
 with existing baselines and outperforms several main
stream time-series prediction models in this specific sur
gical scenario. This work provides an initial exploration of
 multi-view camera selection for thyroidectomy, with poten
tial value for surgical video documentation and training.


## 🖼️ System Overview

<p align="center">
  <img src="figs/NetArch.png" alt="System Overview of SurgicalCamSwitch" width="700"/>
</p>

**Figure 1.** Overall architecture of the *SurgicalCamSwitch* framework.  
The system captures synchronized multi-view surgical videos, extracts visual and temporal features, and predicts the optimal viewpoint through a transformer-based temporal modeling network.

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

@article{liu2025tsp,
  title={TSP-OCS: A Time-Series Prediction for Optimal Camera Selection in Multi-Viewpoint Surgical Video Analysis},
  author={Liu, Xinyu and Lin, Xiaoguang and Liu, Xiang and Yang, Yong and Wang, Hongqian and Sun, Qilong},
  journal={IEEE Journal of Biomedical and Health Informatics},
  year={2025},
  publisher={IEEE}
}

---

