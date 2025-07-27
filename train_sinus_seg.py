import os
import torch
import torch.nn as nn
import numpy as np
import nibabel as nib
import pandas as pd
from tqdm import tqdm
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from models.sinus_seg_unet3d import UNet3D

# -------------------- CONFIG -------------------- #
data_dir = "data"
mask_dir = "sinus_masks"
checkpoint_dir = "checkpoints_sinus"
log_path = "outputs_sinus/train_log.csv"
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs("outputs_sinus", exist_ok=True)
os.makedirs("visualizations", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- Subject Splits -------------------- #
train_ids = [f"subj{str(i).zfill(3)}" for i in range(1, 13)]      # subj001–subj012
val_ids   = [f"subj{str(i).zfill(3)}" for i in range(13, 16)]     # subj013–subj015

# -------------------- Hyperparameters -------------------- #
n_epochs = 100
lr = 1e-4
batch_size = 1
target_shape = (128, 128, 128)

# -------------------- Helper Functions -------------------- #
def normalize(volume):
    vmin, vmax = np.percentile(volume, 1), np.percentile(volume, 99)
    return np.clip((volume - vmin) / (vmax - vmin), 0, 1)

def resize(volume, target_shape, is_mask=False):
    zoom_factors = [t / o for t, o in zip(target_shape, volume.shape)]
    order = 0 if is_mask else 1
    return zoom(volume, zoom_factors, order=order)

def load_and_preprocess(img_path, mask_path, target_shape):
    img = nib.load(img_path).get_fdata().astype(np.float32)
    mask = nib.load(mask_path).get_fdata().astype(np.float32)
    img = normalize(img)
    mask = (mask > 0).astype(np.float32)
    img = resize(img, target_shape, is_mask=False)
    mask = resize(mask, target_shape, is_mask=True)
    img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
    mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)
    return img, mask

def random_flip(img, mask):
    if np.random.rand() > 0.5:
        img = np.flip(img, axis=2).copy()
        mask = np.flip(mask, axis=2).copy()
    if np.random.rand() > 0.5:
        img = np.flip(img, axis=3).copy()
        mask = np.flip(mask, axis=3).copy()
    if np.random.rand() > 0.5:
        img = np.flip(img, axis=4).copy()
        mask = np.flip(mask, axis=4).copy()
    return img, mask

def dice_loss(pred, target, epsilon=1e-6):
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = (pred * target).sum()
    return 1 - (2. * intersection + epsilon) / (pred.sum() + target.sum() + epsilon)

class ComboLoss(nn.Module):
    def __init__(self, alpha=0.5, pos_weight=5.0):
        super().__init__()
        self.alpha = alpha
        self.bce = nn.BCELoss()
        self.pos_weight = pos_weight

    def forward(self, pred, target):
        dice = dice_loss(pred, target)
        bce = self.bce(pred, target)
        return self.alpha * dice + (1 - self.alpha) * bce

# -------------------- Model -------------------- #
model = UNet3D().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = ComboLoss(alpha=0.7, pos_weight=5.0)

# -------------------- Training -------------------- #
log = []
for epoch in range(1, n_epochs + 1):
    model.train()
    train_losses = []

    for subj_id in tqdm(train_ids, desc=f"[Epoch {epoch}] Training"):
        img_path = os.path.join(data_dir, f"{subj_id}_OAT.nii")
        mask_path = os.path.join(mask_dir, f"{subj_id}_sinus.nii")
        img, mask = load_and_preprocess(img_path, mask_path, target_shape)
        img, mask = random_flip(img.numpy(), mask.numpy())
        img = torch.from_numpy(img).float().to(device)
        mask = torch.from_numpy(mask).float().to(device)

        pred = model(img)
        loss = criterion(pred, mask)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        train_losses.append(loss.item())

    model.eval()
    val_losses = []
    with torch.no_grad():
        for subj_id in val_ids:
            img_path = os.path.join(data_dir, f"{subj_id}_OAT.nii")
            mask_path = os.path.join(mask_dir, f"{subj_id}_sinus.nii")
            img, mask = load_and_preprocess(img_path, mask_path, target_shape)
            img = img.float().to(device)
            mask = mask.float().to(device)

            pred = model(img)
            val_loss = criterion(pred, mask)
            val_losses.append(val_loss.item())

            # Visualization using slice 104 (rescaled ≈ slice 102 in resized volume)
            slice_axial = 102
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.imshow(img.cpu().numpy()[0, 0, :, :, slice_axial], cmap='gray')
            plt.title("Image (Axial @104)")
            plt.axis('off')
            plt.subplot(1, 3, 2)
            plt.imshow(mask.cpu().numpy()[0, 0, :, :, slice_axial], cmap='jet', alpha=0.5)
            plt.title("Mask (Axial @104)")
            plt.axis('off')
            plt.subplot(1, 3, 3)
            plt.imshow(pred.cpu().numpy()[0, 0, :, :, slice_axial], cmap='jet', alpha=0.5)
            plt.title("Prediction (Axial @104)")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f"visualizations/val_{subj_id}_epoch{epoch}_axial104.png")
            plt.close()

    train_avg = np.mean(train_losses)
    val_avg = np.mean(val_losses)
    train_dice = 1 - train_avg
    val_dice = 1 - val_avg

    print(f"Epoch {epoch:03d} | Train Dice Loss: {train_avg:.4f} (Score: {train_dice:.4f}) | "
          f"Val Dice Loss: {val_avg:.4f} (Score: {val_dice:.4f})")

    torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"epoch_{epoch}.pth"))
    log.append([epoch, train_avg, val_avg, train_dice, val_dice])

# -------------------- Save Log -------------------- #
df = pd.DataFrame(log, columns=["Epoch", "TrainDiceLoss", "ValDiceLoss", "TrainDiceScore", "ValDiceScore"])
df.to_csv(log_path, index=False)
print("Training complete.")
