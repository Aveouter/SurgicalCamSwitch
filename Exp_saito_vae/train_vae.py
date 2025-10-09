#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py – Multi-GPU VAE Training Script
"""

import argparse
import os
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 依赖你的本地实现
from loader import EgoImageDataset, VAELoss
from VAE import VAE

# --------------------------- Argument Parser --------------------------- #
def get_args():
    timestr = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="Multi-GPU VAE Trainer")

    # 训练超参
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--latent_dim", type=int, default=64)

    # 数据与模式
    parser.add_argument("--mode", type=str, default="Surgery-out",
                        choices=["Surgery-out", "Senquence-out"])
    parser.add_argument("--surgery", type=str, default="video20220714",
                        choices=["video20220722", "video20220729",
                                 "video20220801", "video20221110",
                                 "video20230315"])
    parser.add_argument("--ego_dir", type=str,
                        default=r"D:\LiuXy\CIGIT\CameraSwitchingAlgorithm\surgery_video")
    parser.add_argument("--csv_path", type=str, default="data/data_20220722.csv")
    parser.add_argument("--cam_dirs", type=str, nargs="+",
                        default=[f"data/{i}" for i in range(1, 7)])

    # 文件与日志
    parser.add_argument("--save_path", type=str, default="weight/vae.pth")
    parser.add_argument("--log_file", type=str, default=f"train_{timestr}.log")

    # 多 GPU
    parser.add_argument("--gpu_ids", type=int, nargs="+", default=[1],
                        help="List of GPU ids, e.g. --gpu_ids 0 1 2")

    return parser.parse_args()


# --------------------------- Main --------------------------- #
def main():
    args = get_args()

    # ---------------- Device & Model ---------------- #
    gpu_ids = args.gpu_ids
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{gpu_ids[0]}" if use_cuda else "cpu")

    model = VAE(latent_dim=args.latent_dim).to(device)

    if use_cuda and len(gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
        print(f"[Info] DataParallel enabled on GPUs: {gpu_ids}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = VAELoss()

    # ---------------- Dataset & DataLoader ---------------- #
    dataset = EgoImageDataset(args.ego_dir, mode=args.mode,
                              surgery=args.surgery, augment=True)
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=0,
                            pin_memory=use_cuda)

    # ---------------- Logging Setup ---------------- #
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    with open(args.log_file, "w") as log:
        log.write("Training VAE with parameters:\n")
        for k, v in vars(args).items():
            log.write(f"{k}: {v}\n")

    best_loss = float("inf")

    # ---------------- Training Loop ---------------- #
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            anchor = batch["anchor"].to(device, non_blocking=True)
            positive = batch["positive"].to(device, non_blocking=True)

            optimizer.zero_grad()
            recon, mu, logvar       =   model(anchor)
            recon_a, mu_a, logvar_a =   model(positive)

            # 重构损失
            rec_loss = 0.5 * (F.mse_loss(recon, anchor) + F.mse_loss(recon_a, positive))

            # KL 散度
            kl  = -0.5 * (1 + logvar   - mu.pow(2)   - logvar.exp()).sum(dim=1).mean()
            kl += -0.5 * (1 + logvar_a - mu_a.pow(2) - logvar_a.exp()).sum(dim=1).mean()
            kl_loss = 0.5 * kl

            # 对比损失
            contrastive_loss = 0.5 * ((mu - mu_a).pow(2).sum(dim=1).mean())

            # 总损失（权重 α=1, β=1, γ=10）
            loss = rec_loss + kl_loss + 10.0 * contrastive_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}")

        with open(args.log_file, "a") as log:
            log.write(f"Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}\n")

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = args.save_path.replace(".pth", "_best.pth")
            torch.save(model.state_dict(), best_path)

        # 清理显存
        torch.cuda.empty_cache()

    # Save final
    torch.save(model.state_dict(), args.save_path)
    print(f"[Done] Training complete. Best model saved to {best_path}")

    # ---------------- (Optional) Evaluation ---------------- #
    # 你可以在此处添加评估逻辑，确保在 no_grad 环境下运行
    # model.eval()
    # ...

if __name__ == "__main__":
    main()


    # # Evaluate accuracy
    # transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor()
    # ])
    # df = pd.read_csv(args.csv_path)
    # correct = 0
    # with torch.no_grad():
    #     for idx, row in tqdm(df.head(110).iterrows(), total=110, desc="Evaluating"):
    #         frame_id = f"{idx:04d}"
    #         gt_label = row["label"]
    #         ego_path = os.path.join(args.ego_dir, f"ego_screenshot_{frame_id}.jpg")
    #         ego_img = transform(Image.open(ego_path).convert("RGB")).unsqueeze(0).to(device)
    #         mu_ego, _ = model.encoder(ego_img)

    #         distances = []
    #         for cam_id, cam_dir in enumerate(args.cam_dirs, 1):
    #             cam_path = os.path.join(cam_dir, f"len{cam_id}_screenshot_{frame_id}.jpg")
    #             cam_img = transform(Image.open(cam_path).convert("RGB")).unsqueeze(0).to(device)
    #             mu_cam, _ = model.encoder(cam_img)
    #             distances.append(torch.norm(mu_ego - mu_cam).item())

    #         pred = distances.index(min(distances)) + 1
    #         correct += int(pred == gt_label)

    # acc = correct / 110
    # print(f"\n✅ Camera Selection Accuracy: {acc * 100:.2f}%")
    # with open(args.log_file, "a") as log:
    #     log.write(f"Final Accuracy: {acc * 100:.2f}%\n")
