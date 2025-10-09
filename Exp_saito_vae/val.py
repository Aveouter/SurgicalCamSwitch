import glob
from math import ceil
import random
import re
import pandas as pd
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import os
from VAE import VAE  # Assuming VAE is defined in vae_model.
from pathlib import Path
from typing import Optional, Union

def strip_module_prefix(state_dict):
    # 判断是否带 "module." 前缀
    if any(k.startswith("module.") for k in state_dict.keys()):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def evaluate_model(model_path, mode = "Surgery-out", surgery = "video20220722"):
    """
    Evaluate the VAE model for camera selection accuracy and other metrics.

    Args:
        csv_path (str): Path to the CSV file containing ground truth labels.
        ego_dir (str): Directory containing ego images.
        cam_dirs (list): List of directories containing camera images.
        model_path (str): Path to the trained VAE model weights.
        mode (str): Evaluation mode, either "surgeryour" or "sequence-out".
        total (int): Number of samples to evaluate.

    Returns:
        dict: Dictionary containing accuracy, precision, recall, and F1-score.
    """
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:1")
    model = VAE(latent_dim=64).to(device)

    # 加载权重并处理前缀
    state_dict = torch.load(model_path, map_location=device)
    state_dict = strip_module_prefix(state_dict)

    # 加载权重到模型中
    model.load_state_dict(state_dict)
    model.eval()

    y_true = []
    y_pred = []

    if mode == 'Surgery-out':
        file_list = [surgery]
    elif mode == 'Senquence-out':
        file_list = [surgery]

    pd_dict = {}
    folders = []
    path_img = {}
    for video in file_list:
        clean_id = re.sub(r"\D", "", video)  # 去除非数字字符，例如得到 '20220722'
        csv_path = os.path.join("/baksv/CIGIT/GXN_Liuxy/Time-Series-Library/vae/Exp_saito_vae", f"total_data_{clean_id}.csv")
        df = pd.read_csv(csv_path)
        pd_dict[video] = df.reset_index(drop=True)
        folder = os.path.join("/baksv/CIGIT/GXN_Liuxy/LenSe", video, "screenshots")
        folders.append(folder)
        ego_folder = os.path.join(folder, "ego")
        image_paths = sorted(glob.glob(os.path.join(ego_folder, "**", "*.jpg"), recursive=True))
        if mode == 'Senquence-out': # 取 ego文件夹下后30%的fram编号
            path_img[folder] = [re.search(r"(\d{4})(?=\.jpg$)", os.path.basename(path)).group(1) for path in image_paths[-ceil(len(image_paths) * 0.3):]]
        else: # 取 ego文件夹下所有的fram编号
            path_img[folder] = [re.search(r"(\d{4})(?=\.jpg$)", os.path.basename(path)).group(1) for path in image_paths]

    for _, folder in enumerate(folders):
        for imgs in tqdm(path_img[folder],
                        desc=f"{Path(folder).parent.name} frames",
                        leave=False): 
        # for i,imgs in enumerate(path_img[folder]): 
        #     if i % 100    == 0:
        #         print(f"{i}/{len(path_img[folder])}   Processing {Path(folder).parent.name} frame {imgs}")
            ego_img = transform_img(Image.open(os.path.join("/baksv/CIGIT/GXN_Liuxy/LenSe", video, "screenshots", "ego", f'ego_screenshot_{imgs}.jpg')).convert('RGB')).unsqueeze(0).to(device)
            img_list = [transform_img(Image.open(os.path.join("/baksv/CIGIT/GXN_Liuxy/LenSe", video, "screenshots", f"{len}", f'len{len}_screenshot_{imgs}.jpg')).convert('RGB'),
                                      augment=False).unsqueeze(0).to(device) for len in range(1,7)]
            pred_cam = val(img_list,ego_img,model)
            # print(f"imgs:{imgs}")
            true_cam = pd_dict[Path(folder).parent.name].loc[df["time"] == int(imgs), "label"].values[0]
            # print(f"pred_cam:{pred_cam},true_cam:{true_cam}")
            y_pred.append(int(pred_cam))
            y_true.append(int(true_cam))
    # Compute metrics
    accuracy = sum(1 for true, pred in zip(y_true, y_pred) if true == pred) / len(y_true)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print(f"Camera Selection Accuracy ({mode}): {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }


def val(img_list,ego,model):
    with torch.no_grad():
        mu_ego, _ = model.encoder(ego)
        z_ego = mu_ego.squeeze(0)
        distances = []
        for cam_img in img_list:
            mu_cam, _ = model.encoder(cam_img)
            z_cam = mu_cam.squeeze(0)

            dist = torch.norm(z_ego - z_cam).item()
            distances.append(dist)
    pred_cam = distances.index(min(distances)) + 1  # 预测的摄像头编号（1~6）
    return pred_cam

def transform_img(img, augment: bool = True, device: Optional[Union[torch.device, str]] = None):
    """
    参数
    -------
    img : PIL.Image
    augment : bool            True → 随机增强；False → 仅 Resize
    device : torch.device | str | None
        - "cuda" / torch.device("cuda"):   强制放到 GPU
        - "cpu"  / torch.device("cpu") :   强制放到 CPU
        - None (默认) : 若系统可用 GPU 则自动用 "cuda"，否则用 "cpu"

    返回
    -------
    torch.Tensor  (C, 224, 224)  已放入指定 device
    """
    # 自动推断设备
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # ---------- 基础预处理 ----------
    basic_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    if not augment:
        return basic_tf(img).to(device)

    # ---------- 带数据增强的完整 pipeline ----------
    aug_tf = transforms.Compose([
        transforms.RandomApply([
            transforms.Lambda(
                lambda im: transforms.Resize((224, 224))(
                    transforms.CenterCrop(size=random.randint(180, 200))(im)
                )
            )
        ], p= 0.2),

        transforms.RandomApply([
            transforms.RandomPerspective(
                distortion_scale=0.5,
                p=1.0,
                interpolation=InterpolationMode.BICUBIC
            )
        ], p=0.2),

        transforms.RandomApply([
            transforms.RandomRotation(degrees=(-180, 180))
        ], p=0.2),

        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=(0.9, 1.65),
                contrast=(1.0, 1.35),
                saturation=(1.0, 1.4)
            )
        ], p=0.2),

        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.5, 1.5))
        ], p=0.2),

        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    return aug_tf(img).to(device)
