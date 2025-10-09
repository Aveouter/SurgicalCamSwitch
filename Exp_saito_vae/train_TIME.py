import argparse
import os
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from loader import EgoImageDataset, VAELoss
from VAE import VAE

import time, contextlib

class BatchTimer:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.load_time = 0.0
        self.fwd_time = 0.0
        self.bwd_time = 0.0
        self.batch_start = 0.0
    
    @contextlib.contextmanager
    def track(self, phase):
        start = time.perf_counter()
        yield
        elapsed = time.perf_counter() - start
        if phase == 'load':
            self.load_time = elapsed
        elif phase == 'fwd':
            self.fwd_time = elapsed
        elif phase == 'bwd':
            self.bwd_time = elapsed
    
    def start_batch(self):
        self.batch_start = time.perf_counter()
    
    def get_batch_time(self):
        return time.perf_counter() - self.batch_start
    
    def get_times(self):
        return {
            'load': self.load_time,
            'fwd': self.fwd_time,
            'bwd': self.bwd_time,
            'total': self.load_time + self.fwd_time + self.bwd_time
        }

def get_args():
    timestr = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="Multi-GPU VAE Trainer")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--mode", type=str, default="Surgery-out",
                        choices=["Surgery-out", "Senquence-out"])
    parser.add_argument("--surgery", type=str, default="video20220714",
                        choices=["video20220722", "video20220729",
                                 "video20220801", "video20221110",
                                 "video20230315"])
    parser.add_argument("--ego_dir", type=str,
                        default=r"D:\\LiuXy\\CIGIT\\CameraSwitchingAlgorithm\\surgery_video")
    parser.add_argument("--csv_path", type=str, default="data/data_20220722.csv")
    parser.add_argument("--cam_dirs", type=str, nargs="+",
                        default=[f"data/{i}" for i in range(1, 7)])
    parser.add_argument("--save_path", type=str, default="weight/vae.pth")
    parser.add_argument("--log_file", type=str, default=f"logs/train_{timestr}.log")
    parser.add_argument("--batch_log_file", type=str, default=f"logs/batch_times_{timestr}.csv")
    parser.add_argument("--gpu_ids", type=int, nargs="+", default=[2,3,4,5],
                        help="List of GPU ids, e.g. --gpu_ids 0 1 2")

    return parser.parse_args()

def main():
    args = get_args()
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    os.makedirs(os.path.dirname(args.batch_log_file), exist_ok=True)

    device = torch.device(f"cuda:{args.gpu_ids[0]}" if torch.cuda.is_available() else "cpu")
    model = VAE(latent_dim=args.latent_dim).to(device)

    if torch.cuda.is_available() and len(args.gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)
        print(f"[Info] DataParallel enabled on GPUs: {args.gpu_ids}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    dataset = EgoImageDataset(args.ego_dir, mode=args.mode,
                              surgery=args.surgery, augment=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=4,
                            pin_memory=torch.cuda.is_available())

    writer = SummaryWriter(log_dir="runs/vae_timing")

    best_loss = float("inf")
    timer = BatchTimer()

    # Write headers to log files
    with open(args.log_file, "w") as log:
        log.write("Training Log\n")
        log.write(f"Start time: {pd.Timestamp.now()}\n")
        log.write(f"Configuration: {vars(args)}\n\n")

    with open(args.batch_log_file, "w") as log:
        log.write("epoch,batch_idx,load_time,fwd_time,bwd_time,total_time,loss\n")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        epoch_start = time.perf_counter()

        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Ep {epoch+1}/{args.epochs}", leave=False)):
            timer.start_batch()
            
            # Data loading
            with timer.track('load'):
                anchor = batch["anchor"].to(device, non_blocking=True)
                positive = batch["positive"].to(device, non_blocking=True)

            optimizer.zero_grad()
            
            # Forward pass
            with timer.track('fwd'):
                recon,   mu,   logvar   = model(anchor)
                recon_a, mu_a, logvar_a = model(positive)
                rec_loss = 0.5 * (F.mse_loss(recon, anchor) + F.mse_loss(recon_a, positive))
                kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()
                kl += -0.5 * (1 + logvar_a - mu_a.pow(2) - logvar_a.exp()).sum(dim=1).mean()
                kl_loss = 0.5 * kl
                contrastive_loss = 0.5 * ((mu - mu_a).pow(2).sum(dim=1).mean())
                loss = rec_loss + kl_loss + 10.0 * contrastive_loss

            # Backward pass
            with timer.track('bwd'):
                loss.backward()
                optimizer.step()

            # Calculate batch time and get timing info
            batch_time = timer.get_batch_time()
            times = timer.get_times()
            total_loss += loss.item()

            # Log batch timing information
            with open(args.batch_log_file, "a") as log:
                log.write(f"{epoch+1},{batch_idx},{times['load']:.6f},{times['fwd']:.6f},"
                         f"{times['bwd']:.6f},{batch_time:.6f},{loss.item():.6f}\n")

            # TensorBoard logging for each batch
            if batch_idx % 10 == 0:  # Log every 10 batches to avoid too many points
                writer.add_scalar("Time/batch_load_time", times['load'], epoch * len(dataloader) + batch_idx)
                writer.add_scalar("Time/batch_fwd_time", times['fwd'], epoch * len(dataloader) + batch_idx)
                writer.add_scalar("Time/batch_bwd_time", times['bwd'], epoch * len(dataloader) + batch_idx)
                writer.add_scalar("Time/batch_total_time", batch_time, epoch * len(dataloader) + batch_idx)
                writer.add_scalar("Loss/batch_loss", loss.item(), epoch * len(dataloader) + batch_idx)

            timer.reset()

        # Epoch summary
        avg_loss = total_loss / len(dataloader)
        epoch_time = time.perf_counter() - epoch_start

        with open(args.log_file, "a") as log:
            log.write(f"\n[Epoch {epoch+1}] Avg loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s\n")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = args.save_path.replace(".pth", "_best.pth")
            torch.save(model.state_dict(), best_path)

        torch.cuda.empty_cache()

    writer.close()
    torch.save(model.state_dict(), args.save_path)
    
    # Final summary
    with open(args.log_file, "a") as log:
        log.write(f"\nTraining completed. Best loss: {best_loss:.4f}\n")
        log.write(f"Best model saved to: {best_path}\n")
        log.write(f"Batch timing details saved to: {args.batch_log_file}\n")

    print(f"[Done] Training complete. Best model saved to {best_path}")
    print(f"Detailed batch timing information saved to {args.batch_log_file}")

if __name__ == "__main__":
    main()