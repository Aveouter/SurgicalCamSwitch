import os
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import pandas as pd

# ========== Residual Blocks ==========
class ResidualDecodeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False, stride=1):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) if upsample else nn.Identity()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.upsample(x)
        identity = self.shortcut(out)
        out = self.relu(self.bn1(self.conv1(out)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            identity = self.downsample(x)
        return self.relu(out + identity)

# ========== VAE Model ==========
class VAE_ResNet18_Encoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )
        self.layer1 = self._make_layer(64, 64, 2, 1)
        self.layer2 = self._make_layer(64, 128, 2, 2)
        self.layer3 = self._make_layer(128, 256, 2, 2)
        self.layer4 = self._make_layer(256, 512, 2, 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(nn.Linear(512, 1024), nn.BatchNorm1d(1024), nn.ReLU())
        self.fc_mu = nn.Linear(1024, latent_dim)
        self.fc_logvar = nn.Linear(1024, latent_dim)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        for i in range(blocks):
            s = stride if i == 0 else 1
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, s),
                nn.BatchNorm2d(out_channels)
            ) if i == 0 and (s != 1 or in_channels != out_channels) else None
            layers.append(BasicBlock(in_channels if i == 0 else out_channels, out_channels, s, downsample))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x).view(x.size(0), -1)
        x = self.fc(x)
        return self.fc_mu(x), self.fc_logvar(x)

class VAE_ResNet18_Decoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(latent_dim, 512), nn.BatchNorm1d(512), nn.ReLU())
        self.init_upsample = nn.Upsample(scale_factor=7, mode='bilinear', align_corners=True)
        self.blocks = nn.Sequential(
            ResidualDecodeBlock(512, 512, upsample=True),
            ResidualDecodeBlock(512, 512),
            ResidualDecodeBlock(512, 256, upsample=True),
            ResidualDecodeBlock(256, 256),
            ResidualDecodeBlock(256, 128, upsample=True),
            ResidualDecodeBlock(128, 128),
            ResidualDecodeBlock(128, 64, upsample=True),
            ResidualDecodeBlock(64, 64)
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 2, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.fc(z).unsqueeze(-1).unsqueeze(-1)
        x = self.init_upsample(x)
        x = self.blocks(x)
        return self.deconv(x)

class VAE(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.encoder = VAE_ResNet18_Encoder(latent_dim)
        self.decoder = VAE_ResNet18_Decoder(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

# ========== Dataset ==========
class EgoImageDataset(Dataset):
    def __init__(self, image_dir, augment=True):
        self.image_files = sorted([
            os.path.join(image_dir, f) for f in os.listdir(image_dir)
            if f.lower().endswith(('jpg', 'png', 'jpeg'))
        ])
        self.augment = augment
        self.base_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.augment_transform = transforms.Compose([
            transforms.RandomApply([
                transforms.Lambda(lambda img: transforms.Resize((224, 224))(
                    transforms.CenterCrop(random.randint(180, 200))(img)))
            ], p=0.5),
            transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
            transforms.RandomApply([
                transforms.RandomRotation(degrees=(-180, 180))
            ], p=0.5),
            transforms.ColorJitter(brightness=(0.9, 1.65), contrast=(1.0, 1.35), saturation=(1.0, 1.4)),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.5, 1.5))
            ], p=0.5),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img = Image.open(self.image_files[idx]).convert("RGB")
        if self.augment:
            return {"anchor": self.augment_transform(img), "positive": self.augment_transform(img)}
        else:
            tensor_img = self.base_transform(img)
            return {"anchor": tensor_img, "positive": tensor_img}

# ========== Loss ==========
class VAELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, recon_x, x, mu, logvar):
        recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + 1e-3 * kl_loss

# ========== Main ==========
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--save_path", type=str, default="weight/vae.pth")
    parser.add_argument("--log_file", type=str, default="train.log")
    parser.add_argument("--csv_path", type=str, default="data/data_20220722.csv")
    parser.add_argument("--ego_dir", type=str, default="data/ego")
    parser.add_argument("--cam_dirs", type=str, nargs='+', default=[f"data/{i}" for i in range(1, 7)])
    args = parser.parse_args()

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = VAE(latent_dim=args.latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    dataset = EgoImageDataset(args.ego_dir, augment=True)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Train with logging
    with open(args.log_file, "w") as log:
        loss_fn = VAELoss()
        best_loss = float('inf')
        for epoch in range(args.epochs):
            model.train()
            total_loss = 0
            for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}"):
                anchor = batch["anchor"].to(device)
                optimizer.zero_grad()
                recon, mu, logvar = model(anchor)

                recon_loss = F.mse_loss(recon, anchor)
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / anchor.size(0)

                loss = recon_loss + 1e-3 * kl_loss

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            log.write(f"Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}\n")
            print(f"Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}")
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), args.save_path.replace(".pth", "_best.pth"))
        torch.save(model.state_dict(), args.save_path)

    print(f"Training complete. Best model saved to {args.save_path.replace('.pth', 'vae_best.pth')}")
    model.load_state_dict(torch.load(args.save_path.replace(".pth", "_best.pth")))
    model.eval()

    # Evaluate accuracy
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    df = pd.read_csv(args.csv_path)
    correct = 0
    with torch.no_grad():
        for idx, row in tqdm(df.head(110).iterrows(), total=110, desc="Evaluating"):
            frame_id = f"{idx:04d}"
            gt_label = row["label"]
            ego_path = os.path.join(args.ego_dir, f"ego_screenshot_{frame_id}.jpg")
            ego_img = transform(Image.open(ego_path).convert("RGB")).unsqueeze(0).to(device)
            mu_ego, _ = model.encoder(ego_img)

            distances = []
            for cam_id, cam_dir in enumerate(args.cam_dirs, 1):
                cam_path = os.path.join(cam_dir, f"len{cam_id}_screenshot_{frame_id}.jpg")
                cam_img = transform(Image.open(cam_path).convert("RGB")).unsqueeze(0).to(device)
                mu_cam, _ = model.encoder(cam_img)
                distances.append(torch.norm(mu_ego - mu_cam).item())

            pred = distances.index(min(distances)) + 1
            correct += int(pred == gt_label)

    acc = correct / 110
    print(f"\n✅ Camera Selection Accuracy: {acc * 100:.2f}%")
    with open(args.log_file, "a") as log:
        log.write(f"Final Accuracy: {acc * 100:.2f}%\n")
