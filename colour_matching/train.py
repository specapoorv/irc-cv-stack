import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from datetime import datetime

# =============================
# CONFIG
# =============================
IMG_DIR = "/home/specapoorv/irc-cv-stack/data_colour/dataset"
EPOCHS = 50
BATCH_SIZE = 16
LR = 1e-4
TEMPERATURE = 0.4
VAL_SPLIT = 0.1
DEBUG_EVERY = 5
DEBUG_K = 5

CHECKPOINT_PATH = "simclr_checkpoint_22dec_800.pth"
METRICS_PATH = "metrics.json"
DEBUG_DIR = "debug_samples"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ColourDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, allowed_colors=None, transform=None):
        if allowed_colors is None:
            allowed_colors = ["blue","cyan","green","red","yellow","orange"]

        self.color_to_idx = {color: idx for idx, color in enumerate(allowed_colors)}
        self.imgs_per_color = {color: [] for color in allowed_colors}

        for color in allowed_colors:
            folder = os.path.join(img_dir, color)
            if not os.path.exists(folder):
                continue
            for img_name in os.listdir(folder):
                if img_name.lower().endswith(('.png','.jpg','.jpeg')):
                    self.imgs_per_color[color].append(os.path.join(folder, img_name))

        self.colors = list(self.imgs_per_color.keys())
        self.transform = transform or transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0),
            transforms.ToTensor()
        ])

        # Flatten list for indexing
        self.all_imgs = []
        for color in self.colors:
            for path in self.imgs_per_color[color]:
                self.all_imgs.append((path, color))

        print(f"Loaded {len(self.all_imgs)} images from {len(self.colors)} colors.")

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        img1_path, color = self.all_imgs[idx]
        img1 = Image.open(img1_path).convert("RGB")
        img1 = self.transform(img1)

        # pick another image of same color
        img2_path = self.imgs_per_color[color][torch.randint(len(self.imgs_per_color[color]), (1,)).item()]
        img2 = Image.open(img2_path).convert("RGB")
        img2 = self.transform(img2)

        return img1, img2



class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x):
        x = self.net(x)
        return x.view(x.size(0), -1)

class ProjectionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

    def forward(self, x):
        return self.net(x)

# =============================
# LOSSES & METRICS
# =============================
def nt_xent_loss(z1, z2, temperature):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    N = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.T) / temperature

    mask = ~torch.eye(2*N, device=z.device).bool()
    sim = sim.masked_select(mask).view(2*N, -1)

    pos = torch.sum(z1 * z2, dim=1) / temperature
    pos = torch.cat([pos, pos], dim=0)

    loss = -torch.log(torch.exp(pos) / torch.sum(torch.exp(sim), dim=1))
    return loss.mean()

def alignment(z1, z2):
    return torch.mean(torch.norm(z1 - z2, dim=1) ** 2).item()

def uniformity(z):
    sq_pdist = torch.pdist(z, p=2).pow(2)
    return torch.log(torch.mean(torch.exp(-2 * sq_pdist))).item()

# =============================
# DEBUG IMAGE SAVING
# =============================
def save_debug_samples(epoch, encoder, projector, dataset):
    os.makedirs(DEBUG_DIR, exist_ok=True)
    out = os.path.join(DEBUG_DIR, f"epoch_{epoch:03d}")
    os.makedirs(out, exist_ok=True)

    encoder.eval()
    projector.eval()

    with torch.no_grad():
        for i in range(DEBUG_K):
            x1, x2 = dataset[i]
            x1 = x1.unsqueeze(0).to(DEVICE)
            x2 = x2.unsqueeze(0).to(DEVICE)

            z1 = F.normalize(projector(encoder(x1)), dim=1)
            z2 = F.normalize(projector(encoder(x2)), dim=1)

            sim = torch.sum(z1 * z2).item()

            save_image(x1, f"{out}/img_{i}_view1.png")
            save_image(x2, f"{out}/img_{i}_view2.png")

            with open(f"{out}/sim.txt", "a") as f:
                f.write(f"Image {i}: cosine similarity = {sim:.4f}\n")

# =============================
# TRAINING
# =============================
def train():
    dataset = ColourDataset(IMG_DIR)
    val_size = int(len(dataset) * VAL_SPLIT)
    train_set, val_set = random_split(dataset, [len(dataset)-val_size, val_size])

    train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, BATCH_SIZE, shuffle=False)

    encoder = SmallCNN().to(DEVICE)
    projector = ProjectionHead().to(DEVICE)

    opt = torch.optim.Adam(
        list(encoder.parameters()) + list(projector.parameters()),
        lr=LR, weight_decay=1e-4
    )

    metrics = []

    for epoch in range(1, EPOCHS + 1):
        encoder.train()
        projector.train()

        loss_sum = 0
        grad_sum = 0
        pos_sim_sum = 0
        neg_sim_sum = 0

        for x1, x2 in train_loader:
            x1, x2 = x1.to(DEVICE), x2.to(DEVICE)

            z1 = projector(encoder(x1))
            z2 = projector(encoder(x2))

            loss = nt_xent_loss(z1, z2, TEMPERATURE)

            opt.zero_grad()
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(projector.parameters()), 5.0
            )
            opt.step()

            z1n = F.normalize(z1, dim=1)
            z2n = F.normalize(z2, dim=1)

            pos_sim_sum += torch.mean(torch.sum(z1n * z2n, dim=1)).item()
            neg_sim_sum += torch.mean(torch.mm(z1n, z2n.T)).item()

            loss_sum += loss.item()
            grad_sum += grad

        encoder.eval()
        projector.eval()
        val_loss = 0

        with torch.no_grad():
            for x1, x2 in val_loader:
                x1, x2 = x1.to(DEVICE), x2.to(DEVICE)
                val_loss += nt_xent_loss(
                    projector(encoder(x1)),
                    projector(encoder(x2)),
                    TEMPERATURE
                ).item()

        record = {
            "epoch": epoch,
            "train_loss": float(loss_sum / len(train_loader)),
            "val_loss": float(val_loss / len(val_loader)),
            "pos_cos_sim": float(pos_sim_sum / len(train_loader)),
            "neg_cos_sim": float(neg_sim_sum / len(train_loader)),
            "embedding_variance": torch.var(z1n, dim=0).mean().item(),
            "alignment": alignment(z1n, z2n),
            "uniformity": uniformity(torch.cat([z1n, z2n], dim=0)),
            "grad_norm": float(grad_sum / len(train_loader)),
            "lr": float(opt.param_groups[0]["lr"]),
            "time": datetime.now().isoformat()
        }


        metrics.append(record)

        with open(METRICS_PATH, "w") as f:
            json.dump(metrics, f, indent=2)

        torch.save({
            "encoder": encoder.state_dict(),
            "projector": projector.state_dict(),
            "epoch": epoch
        }, CHECKPOINT_PATH)

        print(f"Epoch {epoch:03d} | "
              f"Loss {record['train_loss']:.3f} | "
              f"PosSim {record['pos_cos_sim']:.3f} | "
              f"Val {record['val_loss']:.3f}")

        if epoch % DEBUG_EVERY == 0:
            save_debug_samples(epoch, encoder, projector, dataset)

# =============================
# RUN
# =============================
if __name__ == "__main__":
    train()
