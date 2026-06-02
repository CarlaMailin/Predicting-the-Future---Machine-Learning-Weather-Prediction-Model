# ── evaluate.py ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, r2_score
import xarray as xr
import cfgrib

# ── CONSTANTS — must match training ──────────────────────────────────────────
MODEL_PATH = 'best_precip_model.pt'
LAG        = 24 # time steps
HORIZON    = 24
BATCH      = 512
HIDDEN     = 64    # must match what you trained with
KERNEL     = 3
N_FEATURES = 7


# ── COPY THESE FROM YOUR TRAINING SCRIPT ─────────────────────────────────────
# (or save/load them from disk — see note below)
device = torch.device('cpu')

# ── 1. RELOAD DATA ────────────────────────────────────────────────────────────
# paste your full data loading block here (same as training script)
# ending with X_te_sc, y_test, X_va_sc, y_val, train_end, ds0

# ── 2. RELOAD MODEL ───────────────────────────────────────────────────────────
class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hidden, ks=3):
        super().__init__()
        self.hidden = hidden
        self.conv   = nn.Conv2d(in_ch + hidden, 4 * hidden, ks, padding=ks//2)
    def forward(self, x, h, c):
        i, f, o, g = self.conv(torch.cat([x, h], 1)).chunk(4, 1)
        c = torch.sigmoid(f)*c + torch.sigmoid(i)*torch.tanh(g)
        h = torch.sigmoid(o)*torch.tanh(c)
        return h, c

class ConvLSTMModel(nn.Module):
    def __init__(self, in_ch, hidden=32, ks=3):
        super().__init__()
        self.cell = ConvLSTMCell(in_ch, hidden, ks)
        self.fc   = nn.Conv2d(hidden, 1, 1)
    def forward(self, x):
        B, T, C, H, W = x.shape
        h = torch.zeros(B, self.cell.hidden, H, W)
        c = torch.zeros(B, self.cell.hidden, H, W)
        for t in range(T):
            h, c = self.cell(x[:, t], h, c)
        return self.fc(h).squeeze(1)

model = ConvLSTMModel(N_FEATURES, HIDDEN, KERNEL)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print(f"Model loaded from {MODEL_PATH}")

# ── 3. PREDICT ────────────────────────────────────────────────────────────────
def predict(X_sc):
    preds = []
    with torch.no_grad():
        for xb in DataLoader(torch.from_numpy(X_sc), batch_size=BATCH):
            preds.append(model(xb).numpy())
    return np.clip(np.expm1(np.concatenate(preds, axis=0)), 0, None)

y_test_pred = predict(X_te_sc)
y_val_pred  = predict(X_va_sc)
y_test_mm   = np.expm1(y_test)
y_val_mm    = np.expm1(y_val)

# ── 4. METRICS ────────────────────────────────────────────────────────────────
test_rmse = np.sqrt(mean_squared_error(y_test_mm.flatten(), y_test_pred.flatten()))
test_r2   = r2_score(y_test_mm.flatten(), y_test_pred.flatten())
print(f"=== Overall ===")
print(f"Test RMSE : {test_rmse:.4f} mm")
print(f"Test R²   : {test_r2:.4f}")

# 17:00 mask
test_times     = pd.to_datetime(ds0.time.values[train_end + LAG : train_end + LAG + len(y_test)])
mask_17        = test_times.hour == 17
y_test_17      = y_test_mm[mask_17]
y_test_pred_17 = y_test_pred[mask_17]
rmse_17 = np.sqrt(mean_squared_error(y_test_17.flatten(), y_test_pred_17.flatten()))
r2_17   = r2_score(y_test_17.flatten(), y_test_pred_17.flatten())
print(f"\n=== 17:00 only ===")
print(f"Test RMSE : {rmse_17:.4f} mm")
print(f"Test R²   : {r2_17:.4f}")
print(f"Samples   : {mask_17.sum()}")

# ── 5. LOSS CURVES ────────────────────────────────────────────────────────────
# load from saved file (add this to training script to save them):
# np.save('train_losses.npy', train_losses)
# np.save('val_losses.npy',   val_losses)
train_losses = np.load('train_losses.npy')
val_losses   = np.load('val_losses.npy')

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].plot(train_losses, label='Train')
axes[0].plot(val_losses,   label='Val')
axes[0].set_title('Loss Curve (linear)')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('MSE')
axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[1].plot(train_losses, label='Train')
axes[1].plot(val_losses,   label='Val')
axes[1].set_yscale('log')
axes[1].set_title('Loss Curve (log scale)')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('MSE (log)')
axes[1].legend(); axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('loss_curves.png', dpi=150)
plt.show()

# ── 6. SPATIAL MAPS ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for row, (obs, pred, label) in enumerate([
    (y_test_mm,  y_test_pred,    'All hours'),
    (y_test_17,  y_test_pred_17, '17:00 only')
]):
    im0 = axes[row,0].imshow(obs.mean(axis=0),              cmap='Blues')
    axes[row,0].set_title(f'Observed mean ({label})')
    plt.colorbar(im0, ax=axes[row,0], label='mm')
    im1 = axes[row,1].imshow(pred.mean(axis=0),             cmap='Blues')
    axes[row,1].set_title(f'Predicted mean ({label})')
    plt.colorbar(im1, ax=axes[row,1], label='mm')
    im2 = axes[row,2].imshow(np.abs(obs-pred).mean(axis=0), cmap='Reds')
    axes[row,2].set_title(f'MAE ({label})')
    plt.colorbar(im2, ax=axes[row,2], label='mm')
plt.tight_layout()
plt.savefig('spatial_maps.png', dpi=150)
plt.show()