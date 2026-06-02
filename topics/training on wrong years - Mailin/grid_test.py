# ── IMPORTS ───────────────────────────────────────────────────────────────────
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import xarray as xr
import cfgrib

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
MODEL_PATH    = 'best_precip_model.pt'
LAG           = 72
HORIZON       = 24
N_TRIALS      = 3
OPTUNA_EPOCHS = 10
FINAL_EPOCHS  = 40
BATCH         = 256
SEED          = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
datasets = cfgrib.open_datasets("data/grid-timeseries_sel_vars_1940.grib")
ds0      = datasets[0]
ref_lat  = ds0.latitude
ref_lon  = ds0.longitude
ref_time = ds0.time

channels, chan_names = [], []

for var in ['sp', 'tcc', 'u10', 'v10', 't2m']:
    channels.append(ds0[var].values.astype(np.float32))
    chan_names.append(var)

ds1 = datasets[1]
for var in ['ssrd', 'tp']:
    arr        = ds1[var]
    arr_summed = arr.sum(dim='step')
    daily_time = arr.time.values
    arr_daily  = xr.DataArray(
        arr_summed.values / 24,
        coords={'time': daily_time, 'latitude': ref_lat, 'longitude': ref_lon},
        dims=['time', 'latitude', 'longitude']
    )
    arr_hourly = arr_daily.reindex(time=ref_time, method='ffill').ffill(dim='time').bfill(dim='time')
    channels.append(arr_hourly.values.astype(np.float32))
    chan_names.append(var)

X_raw  = np.stack(channels, axis=1).astype(np.float32)   # (T, 7, 5, 5)
tp_idx = chan_names.index('tp')
y_raw  = channels[tp_idx].copy()                          # (T, 5, 5)
N      = X_raw.shape[0]

# ── 2. CLEAN ──────────────────────────────────────────────────────────────────
for i in range(X_raw.shape[1]):
    ch     = X_raw[:, i]
    ch_mean = np.nanmean(ch)
    X_raw[:, i] = np.where(np.isnan(ch), ch_mean, ch)
    X_raw[:, i] = np.where(np.isinf(ch), ch_mean, X_raw[:, i])

y_raw = np.where(np.isnan(y_raw), np.nanmean(y_raw), y_raw)
y_raw = np.where(np.isinf(y_raw), np.nanmean(y_raw), y_raw)

# ── 3. LOG TRANSFORM ──────────────────────────────────────────────────────────
y_raw              = np.log1p(y_raw * 1000)               # m → mm → log1p
X_raw[:, tp_idx]   = np.log1p(X_raw[:, tp_idx] * 1000)

# ── 4. LAGGED FEATURES ────────────────────────────────────────────────────────
def add_lagged_features(X_raw, chan_names, lags=[1, 3, 6, 12, 24, 48, 72], roll=24):
    T, C, H, W   = X_raw.shape
    new_channels = []
    new_names    = []
    for i, name in enumerate(chan_names):
        ch = X_raw[:, i]
        new_channels.append(ch)
        new_names.append(name)
        for lag in lags:
            lagged = np.concatenate([
                np.full((lag, H, W), np.nan, dtype=np.float32),
                ch[:-lag]
            ], axis=0)
            new_channels.append(lagged.astype(np.float32))
            new_names.append(f"{name}_lag{lag}h")
        ch_2d   = ch.reshape(T, -1)
        roll_2d = pd.DataFrame(ch_2d).rolling(window=roll, min_periods=1).mean().values
        new_channels.append(roll_2d.reshape(T, H, W).astype(np.float32))
        new_names.append(f"{name}_roll24h")
    return np.stack(new_channels, axis=1), new_names

X_raw, chan_names = add_lagged_features(X_raw, chan_names)

# Fill NaNs from lagging
for i in range(X_raw.shape[1]):
    ch      = X_raw[:, i]
    ch_mean = np.nanmean(ch)
    X_raw[:, i] = np.where(np.isnan(ch), ch_mean, ch)

N_FEATURES = len(chan_names)   # 63
N          = X_raw.shape[0]
print(f"Channels: {N_FEATURES}")
print(f"X_raw: {X_raw.shape}")

# ── 5. SPLIT ──────────────────────────────────────────────────────────────────
train_end = int(0.70 * N)
test_end  = int(0.90 * N)
print(f"N={N} | train={train_end} | test={test_end-train_end} | val={N-test_end}")

# ── 6. WINDOWS ────────────────────────────────────────────────────────────────
def make_windows(X, y, lag=LAG, horizon=HORIZON):
    Xw, yw = [], []
    for i in range(lag, len(X) - horizon):
        Xw.append(X[i - lag:i])
        yw.append(y[i + horizon])
    return np.array(Xw, dtype=np.float32), np.array(yw, dtype=np.float32)

X_train_w, y_train = make_windows(X_raw[:train_end],         y_raw[:train_end])
X_test_w,  y_test  = make_windows(X_raw[train_end:test_end], y_raw[train_end:test_end])
X_val_w,   y_val   = make_windows(X_raw[test_end:],          y_raw[test_end:])
print(f"train {X_train_w.shape} | test {X_test_w.shape} | val {X_val_w.shape}")

# ── 7. SCALE ──────────────────────────────────────────────────────────────────
def fit_scale(Xw):
    N, T, C, H, W = Xw.shape
    sc = StandardScaler()
    sc.fit(Xw.reshape(-1, C))
    return sc

def transform(Xw, sc):
    N, T, C, H, W = Xw.shape
    return sc.transform(Xw.reshape(-1, C)).reshape(N, T, C, H, W).astype(np.float32)

scaler   = fit_scale(X_train_w)
X_tr_sc  = transform(X_train_w, scaler)
X_te_sc  = transform(X_test_w,  scaler)
X_va_sc  = transform(X_val_w,   scaler)

y_mean, y_std = y_train.mean(), y_train.std()
y_tr_n = (y_train - y_mean) / y_std
y_te_n = (y_test  - y_mean) / y_std
y_va_n = (y_val   - y_mean) / y_std

# ── 8. DATASET ────────────────────────────────────────────────────────────────
class SeqDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
    def __len__(self):        return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

def loader(X, y, shuffle): return DataLoader(SeqDS(X, y), batch_size=BATCH, shuffle=shuffle)

# ── 9. MODEL ──────────────────────────────────────────────────────────────────
class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size, padding=pad
        )
    def forward(self, x, h, c):
        gates    = self.conv(torch.cat([x, h], dim=1))
        i, f, o, g = gates.chunk(4, dim=1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c

class ConvLSTMModel(nn.Module):
    def __init__(self, in_channels, hidden_channels=32, kernel_size=3):
        super().__init__()
        self.cell = ConvLSTMCell(in_channels, hidden_channels, kernel_size)
        self.fc   = nn.Conv2d(hidden_channels, 1, kernel_size=1)
    def forward(self, x):
        B, T, C, H, W = x.shape
        h = torch.zeros(B, self.cell.hidden_channels, H, W, device=x.device)
        c = torch.zeros(B, self.cell.hidden_channels, H, W, device=x.device)
        for t in range(T):
            h, c = self.cell(x[:, t], h, c)
        return self.fc(h).squeeze(1)

# ── 10. TRAIN HELPER ──────────────────────────────────────────────────────────
def run_training(model, X_tr, y_tr, X_v, y_v, lr, max_epochs, patience, trial=None):
    model     = model.to(device)
    tr_loader = loader(X_tr, y_tr, shuffle=True)
    va_loader = loader(X_v,  y_v,  shuffle=False)
    opt       = torch.optim.Adam(model.parameters(), lr=lr)
    sched     = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
    best_val, no_improve = float('inf'), 0

    for epoch in range(max_epochs):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            F.mse_loss(model(xb), yb).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vals = [F.mse_loss(model(xb.to(device)), yb.to(device)).item()
                    for xb, yb in va_loader]
        avg = float(np.mean(vals))
        sched.step(avg)
        if avg < best_val:
            best_val, no_improve = avg, 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
        if trial:
            trial.report(avg, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
    return best_val

# ── 11. OPTUNA ────────────────────────────────────────────────────────────────
def objective(trial):
    hidden = trial.suggest_categorical('hidden', [16, 32, 64])
    lr     = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    kernel = trial.suggest_categorical('kernel', [3, 5])
    model  = ConvLSTMModel(N_FEATURES, hidden_channels=hidden, kernel_size=kernel)
    return run_training(model, X_tr_sc, y_tr_n, X_va_sc, y_va_n,
                        lr=lr, max_epochs=OPTUNA_EPOCHS, patience=5, trial=trial)

study = optuna.create_study(
    direction='minimize',
    pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5)
)
study.optimize(objective, n_trials=N_TRIALS)
bp = study.best_params
print("Best params:", bp)
print("Best val MSE:", f"{study.best_value:.6f}")

# ── 12. FINAL TRAINING ────────────────────────────────────────────────────────
model      = ConvLSTMModel(N_FEATURES, hidden_channels=bp['hidden'], kernel_size=bp['kernel'])
model      = model.to(device)
tr_loader  = loader(X_tr_sc, y_tr_n, shuffle=True)
va_loader  = loader(X_va_sc, y_va_n, shuffle=False)
opt        = torch.optim.Adam(model.parameters(), lr=bp['lr'])
sched      = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=6, factor=0.5)

train_losses, val_losses    = [], []
best_val, no_improve        = float('inf'), 0
best_state                  = copy.deepcopy(model.state_dict())

print(f"\nFinal training — up to {FINAL_EPOCHS} epochs …")
for epoch in range(FINAL_EPOCHS):
    model.train()
    tl = []
    for xb, yb in tr_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = F.mse_loss(model(xb), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tl.append(loss.item())
    train_losses.append(np.mean(tl))

    model.eval()
    with torch.no_grad():
        vl = [F.mse_loss(model(xb.to(device)), yb.to(device)).item()
              for xb, yb in va_loader]
    avg_val = float(np.mean(vl))
    val_losses.append(avg_val)
    print(f"  Epoch {epoch+1:03d} | train {train_losses[-1]:.5f} | val {avg_val:.5f}")

    if avg_val < best_val:
        best_val, no_improve = avg_val, 0
        best_state = copy.deepcopy(model.state_dict())
        torch.save(model.state_dict(), MODEL_PATH)
    else:
        no_improve += 1
        if no_improve >= 12:
            print(f"  Early stop at epoch {epoch+1}")
            break

model.load_state_dict(best_state)

# ── 13. EVALUATE ──────────────────────────────────────────────────────────────
def predict(X_sc):
    model.eval()
    with torch.no_grad():
        preds = []
        for xb in DataLoader(torch.from_numpy(X_sc), batch_size=BATCH):
            preds.append(model(xb.to(device)).cpu().numpy())
    pred = np.concatenate(preds, axis=0)
    pred = pred * y_std + y_mean             # denormalize → log-mm
    return np.clip(np.expm1(pred), 0, None)  # → mm

y_test_pred = predict(X_te_sc)
y_val_pred  = predict(X_va_sc)
y_test_mm   = np.expm1(y_test * y_std + y_mean)
y_val_mm    = np.expm1(y_val  * y_std + y_mean)

# Overall
test_rmse = np.sqrt(mean_squared_error(y_test_mm.flatten(), y_test_pred.flatten()))
test_r2   = r2_score(y_test_mm.flatten(), y_test_pred.flatten())
print(f"=== Overall ===")
print(f"Test RMSE : {test_rmse:.4f} mm")
print(f"Test R²   : {test_r2:.4f}")

# 17:00 mask
test_times = pd.to_datetime(ds0.time.values[train_end + LAG : train_end + LAG + len(y_test)])
mask_17        = test_times.hour == 17
y_test_17      = y_test_mm[mask_17]
y_test_pred_17 = y_test_pred[mask_17]
rmse_17 = np.sqrt(mean_squared_error(y_test_17.flatten(), y_test_pred_17.flatten()))
r2_17   = r2_score(y_test_17.flatten(), y_test_pred_17.flatten())
print(f"\n=== 17:00 only ===")
print(f"Test RMSE : {rmse_17:.4f} mm")
print(f"Test R²   : {r2_17:.4f}")
print(f"Samples   : {mask_17.sum()} windows")

# Loss curves
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
plt.show()

# Spatial maps
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for row, (obs, pred, label) in enumerate([
    (y_test_mm,  y_test_pred,    'All hours'),
    (y_test_17,  y_test_pred_17, '17:00 only')
]):
    im0 = axes[row,0].imshow(obs.mean(axis=0),               cmap='Blues')
    axes[row,0].set_title(f'Observed mean ({label})')
    plt.colorbar(im0, ax=axes[row,0], label='mm')
    im1 = axes[row,1].imshow(pred.mean(axis=0),              cmap='Blues')
    axes[row,1].set_title(f'Predicted mean ({label})')
    plt.colorbar(im1, ax=axes[row,1], label='mm')
    im2 = axes[row,2].imshow(np.abs(obs-pred).mean(axis=0),  cmap='Reds')
    axes[row,2].set_title(f'MAE ({label})')
    plt.colorbar(im2, ax=axes[row,2], label='mm')
plt.tight_layout()
plt.show()