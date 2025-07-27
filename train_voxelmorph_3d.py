import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
import random

from models.voxel_unet3d import VoxelMorph3D
from utils.losses import ncc_loss, gradient_loss
from utils.data_loader import load_nifti, normalize, save_nifti

# ========== CONFIG ========== #
original_dir = "data"
augmented_dir = "data_augmented"
template_path = os.path.join(original_dir, "OAT_template.nii")
output_dir = "outputs"
ckpt_dir = "checkpoints"
target_shape = (128, 128, 128)

os.makedirs(output_dir, exist_ok=True)
os.makedirs(ckpt_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== Use Predefined Splits ========== #
def read_split_file(path):
    with open(path) as f:
        return [line.strip() for line in f.readlines()]

train_subjects = read_split_file("splits/train.txt")
val_subjects   = read_split_file("splits/val.txt")

print(f" Loaded {len(train_subjects)} training files and {len(val_subjects)} validation files.")


print(f"Training on {len(train_subjects)} subjects, validating on {len(val_subjects)}")

# ========== Hyperparameters ========== #
lambda_smooth = 10.0
n_epochs = 30
lr = 1e-4

# ========== Load Template ========== #
template_np, template_affine = load_nifti(template_path, target_shape=target_shape)
template_np = normalize(template_np)
template_tensor = torch.from_numpy(template_np).float().unsqueeze(0).unsqueeze(0).to(device)

# ========== Initialize Model ========== #
model = VoxelMorph3D().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# ========== Training Loop ========== #
best_val_loss = float("inf")
logs = []

for epoch in range(1, n_epochs + 1):
    model.train()
    train_losses = []

    for subj_path in tqdm(train_subjects, desc=f"Epoch {epoch} - Training"):
        moving_np, _ = load_nifti(subj_path, target_shape=target_shape)
        moving_np = normalize(moving_np)

        moving_tensor = torch.from_numpy(moving_np).unsqueeze(0).unsqueeze(0).float().to(device)
        input_tensor = torch.cat([moving_tensor, template_tensor], dim=1)

        flow = model(input_tensor)
        grid = F.affine_grid(torch.eye(3, 4).unsqueeze(0).to(device), template_tensor.shape, align_corners=True)
        warped = F.grid_sample(moving_tensor, grid + flow.permute(0, 2, 3, 4, 1), align_corners=True)

        loss = ncc_loss(warped, template_tensor) + lambda_smooth * gradient_loss(flow)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

    # ========== Validation ========== #
    model.eval()
    val_losses = []

    with torch.no_grad():
        for subj_path in tqdm(val_subjects, desc=f"Epoch {epoch} - Validation"):
            moving_np, _ = load_nifti(subj_path, target_shape=target_shape)
            moving_np = normalize(moving_np)

            moving_tensor = torch.from_numpy(moving_np).unsqueeze(0).unsqueeze(0).float().to(device)
            input_tensor = torch.cat([moving_tensor, template_tensor], dim=1)

            flow = model(input_tensor)
            grid = F.affine_grid(torch.eye(3, 4).unsqueeze(0).to(device), template_tensor.shape, align_corners=True)
            warped = F.grid_sample(moving_tensor, grid + flow.permute(0, 2, 3, 4, 1), align_corners=True)

            loss = ncc_loss(warped, template_tensor) + lambda_smooth * gradient_loss(flow)
            val_losses.append(loss.item())

            subj_id = os.path.basename(subj_path).replace(".nii", "")
            save_nifti(warped.cpu().squeeze().numpy(), template_affine,
                       os.path.join(output_dir, f"{subj_id}_warped_epoch{epoch}.nii"))

    # ========== Logging ========== #
    train_loss = np.mean(train_losses)
    val_loss = np.mean(val_losses)
    logs.append([epoch, train_loss, val_loss])
    print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")

    torch.save(model.state_dict(), os.path.join(ckpt_dir, f"model_epoch_{epoch}.pth"))
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), os.path.join(ckpt_dir, f"best_model.pth"))

# ========== Save Loss CSV ========== #
df = pd.DataFrame(logs, columns=["Epoch", "TrainLoss", "ValLoss"])
df.to_csv(os.path.join(output_dir, "train_val_loss.csv"), index=False)
print("Training complete.")
print(f" Best validation loss achieved: {best_val_loss:.5f}")

