import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import random

# -----------------------------
# 1️⃣ Dataset class
# -----------------------------
class ConeDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.img_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".jpg")]

        # SimCLR-style augmentations (colour-preserving)
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0),  # preserve hue
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        img = Image.open(img_path).convert("RGB")
        # Two augmented views
        return self.transform(img), self.transform(img)

# -----------------------------
# 2️⃣ Small CNN encoder
# -----------------------------
class SmallCNN(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
        )
        self.out_dim = out_dim

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return x

# -----------------------------
# 3️⃣ Projection head (1-2 layer MLP)
# -----------------------------
class ProjectionHead(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=128, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)

CHECKPOINT_PATH = "simclr_cone_checkpoint.pth"

def save_checkpoint(epoch, encoder, projector, optimizer):
    torch.save({
        "epoch": epoch,
        "encoder": encoder.state_dict(),
        "projector": projector.state_dict(),
        "optimizer": optimizer.state_dict()
    }, CHECKPOINT_PATH)


def load_checkpoint(encoder, projector, optimizer, device):
    if not os.path.exists(CHECKPOINT_PATH):
        return 0

    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    encoder.load_state_dict(ckpt["encoder"])
    projector.load_state_dict(ckpt["projector"])
    optimizer.load_state_dict(ckpt["optimizer"])

    print(f"✅ Loaded checkpoint from epoch {ckpt['epoch']}")
    return ckpt["epoch"]
# -----------------------------
# 4️⃣ NT-Xent Loss
# -----------------------------
def nt_xent_loss(z1, z2, temperature=0.5):
    batch_size = z1.size(0)
    # L2 normalize embeddings
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    z = torch.cat([z1, z2], dim=0)
    sim_matrix = torch.matmul(z, z.T) / temperature

    # mask out self-similarity
    mask = (~torch.eye(2*batch_size, 2*batch_size, dtype=bool)).to(z.device)
    sim_matrix = sim_matrix.masked_select(mask).view(2*batch_size, -1)

    # positive similarities
    pos_sim = torch.sum(z1 * z2, dim=1) / temperature
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)

    loss = -torch.log(torch.exp(pos_sim) / torch.sum(torch.exp(sim_matrix), dim=1))
    return loss.mean()


# -----------------------------
# 5️⃣ Training loop
# -----------------------------
def train_simclr(img_dir, epochs=100, batch_size=16, lr=1e-4, device="cpu"):
    dataset = ConeDataset(img_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    encoder = SmallCNN().to(device)
    projector = ProjectionHead().to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(projector.parameters()),
        lr=lr,
        weight_decay=1e-4
    )

    start_epoch = load_checkpoint(encoder, projector, optimizer, device)

    for epoch in range(start_epoch, epochs):
        encoder.train()
        projector.train()

        total_loss = 0
        for x1, x2 in loader:
            x1, x2 = x1.to(device), x2.to(device)

            z1 = projector(encoder(x1))
            z2 = projector(encoder(x2))

            loss = nt_xent_loss(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(projector.parameters()),
                5.0
            )
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}")

        save_checkpoint(epoch + 1, encoder, projector, optimizer)

    return encoder

# -----------------------------
# 6️⃣ Example usage
# -----------------------------


if __name__ == "__main__":
    device = "cpu"
    encoder = train_simclr("/home/specapoorv/irc-cv-stack/data/cropped_imgs")
    # Now you can get embeddings for two images and compute similarity
