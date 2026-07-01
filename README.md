# TSP-OCS

**TSP-OCS** is a deep learning framework for **time-series prediction of optimal camera selection** in multi-view open-surgery videos.
The repository focuses on supervised camera-view prediction with temporal models and maintained baseline architectures.

---

## News

It gives me great pleasure to inform you that our manuscript “TSP-OCS: A Time-Series Prediction for Optimal Camera Selection in Multi-Viewpoint Surgical Video Analysis” has been accepted for publication in the IEEE Journal of Biomedical and Health Informatics (J-BHI).

We would like to express our sincere gratitude to all contributors and reviewers for their valuable support!

## Overview
<p align="center">
  <img src="figs/Graph Abstract.jpg " alt="Graph Abstract" width="700"/>
</p>

## Abstract
Recording open-surgery procedures is essential for educational and clinical evaluation purposes. Traditional single-camera methods often suffer from occlusions caused by the surgeon's head and body, as well as limitations from fixed camera angles, which reduce the comprehensibility of the recorded surgical content.

This study focuses on open thyroidectomy and uses a multi-viewpoint recording setup where six synchronized cameras capture surgery from different angles. TSP-OCS provides a supervised time-series prediction framework that forecasts the most informative camera view by extracting visual and semantic features from thyroidectomy videos and modeling temporal dependencies. The work offers an initial exploration of multi-view camera selection for thyroidectomy, with potential value for surgical video documentation and training.

## System Overview

<p align="center">
  <img src="figs/NetArch.png" alt="System Overview of SurgicalCamSwitch" width="700"/>
</p>

**Figure 1.** Overall architecture of the *SurgicalCamSwitch* framework.  
The system captures synchronized multi-view surgical videos, extracts visual and temporal features, and predicts the optimal viewpoint through a transformer-based temporal modeling network.

---

## Repository Layout

```text
.
|-- data_provider/          # Dataset loading and train/val/test split logic
|-- exp/                    # Training and evaluation workflows
|-- layers/                 # Shared model layers
|-- models/                 # Maintained model definitions
|-- scripts/camera_script/  # Camera-selection experiment launch scripts
|-- utils/                  # Metrics, losses, plotting, and training helpers
|-- run.py                  # Main training/testing entrypoint
`-- detect.py               # Detection/testing entrypoint
```

Generated artifacts such as checkpoints, result files, TensorBoard runs, local data, and Python bytecode are ignored by Git.

## Supported Scope

This repository keeps the maintained TSP-OCS camera-selection code path and baseline time-series models. The standalone `Exp_saito_vae` experiment, VAE-specific scripts/model, committed `__pycache__` files, and incomplete M4 summary code have been removed to keep the project reproducible and concise.

The M4 and UEA dataset paths still require their source implementations (`data_provider/m4.py` and `data_provider/uea.py`) if those tasks are needed. They are not part of the cleaned repository.

## Environment

Create a Python environment:

```bash
conda create -n surgicalcam python=3.10
conda activate surgicalcam
```

Install core dependencies:

```bash
pip install torch torchvision torchaudio
pip install numpy pandas scikit-learn matplotlib tqdm
pip install einops reformer-pytorch sktime tensorboard
```

Install extra model dependencies only when the corresponding model is used.

## Usage

Train or evaluate through `run.py`. Example:

```bash
python run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id camera \
  --model TimesNet \
  --data Camera2 \
  --root_path ./data/camera \
  --data_path data_20220722.csv \
  --features MS \
  --target label \
  --camera \
  --loss CrossEntropyLoss
```

For reproducible camera-selection runs, use the scripts under `scripts/camera_script/` and adjust `root_path`, `data_path`, GPU settings, and sequence lengths for the local dataset.

## Citation

```bibtex
@article{liu2025tsp,
  title={TSP-OCS: A Time-Series Prediction for Optimal Camera Selection in Multi-Viewpoint Surgical Video Analysis},
  author={Liu, Xinyu and Lin, Xiaoguang and Liu, Xiang and Yang, Yong and Wang, Hongqian and Sun, Qilong},
  journal={IEEE Journal of Biomedical and Health Informatics},
  year={2025},
  publisher={IEEE}
}
```
