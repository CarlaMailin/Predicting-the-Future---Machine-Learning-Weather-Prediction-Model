# ── IMPORTS ───────────────────────────────────────────────────────────────────
import copy, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from sklearn.metrics import mean_squared_error, r2_score
import xarray as xr
import cfgrib
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
# ── CONSTANTS ─────────────────────────────────────────────────────────────────
MODEL_PATH    = 'best_precip_model.pt'
LAG           = 24
HORIZON       = 24
N_TRIALS      = 3
OPTUNA_EPOCHS = 5
FINAL_EPOCHS  = 15
BATCH         = 512
SEED          = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device('cpu')

# ── 1. LOAD ───────────────────────────────────────────────────────────────────
print("Loading data..."); t0 = time.time()
datasets = cfgrib.open_datasets("data/grid-timeseries_sel_vars_1940.grib")
ds0      = datasets[0]
ds1      = datasets[1]
ref_lat  = ds0.latitude
ref_lon  = ds0.longitude
ref_time = ds0.time

channels, chan_names = [], []
for var in ['sp', 'tcc', 'u10', 'v10', 't2m']:
    channels.append(ds0[var].values.astype(np.float32))
    chan_names.append(var)

for var in ['ssrd', 'tp']:
    arr       = ds1[var].sum(dim='step')
    arr_daily = xr.DataArray(
        arr.values / 24,
        coords={'time': ds1[var].time.values, 'latitude': ref_lat, 'longitude': ref_lon},
        dims=['time', 'latitude', 'longitude']
    )
    arr_h = arr_daily.reindex(time=ref_time, method='ffill').ffill(dim='time').bfill(dim='time')
    channels.append(arr_h.values.astype(np.float32))
    chan_names.append(var)

print(f"Load done in {time.time()-t0:.1f}s")

# ── 2. BUILD X, y ─────────────────────────────────────────────────────────────
print("Building arrays..."); t0 = time.time()
X_raw  = np.stack(channels, axis=1).astype(np.float32)   # (T, 7, 5, 5)
tp_idx = chan_names.index('tp')
y_raw  = channels[tp_idx].copy()

# Fill NaNs/Infs
for i in range(X_raw.shape[1]):
    m = np.nanmean(X_raw[:, i])
    X_raw[:, i] = np.nan_to_num(X_raw[:, i], nan=m, posinf=m, neginf=m)
y_fill = np.nanmean(y_raw)
y_raw  = np.nan_to_num(y_raw, nan=y_fill, posinf=y_fill, neginf=y_fill)

# Log transform
y_raw            = np.log1p(y_raw * 1000)
X_raw[:, tp_idx] = np.log1p(X_raw[:, tp_idx] * 1000)

N_FEATURES = X_raw.shape[1]   # 7
N          = X_raw.shape[0]
print(f"Arrays done in {time.time()-t0:.1f}s | X_raw: {X_raw.shape}")

# ── 3. SPLIT ──────────────────────────────────────────────────────────────────
train_end = int(0.70 * N)
test_end  = int(0.90 * N)
print(f"N={N} | train={train_end} | test={test_end-train_end} | val={N-test_end}")

# ── 4. WINDOWS ────────────────────────────────────────────────────────────────
print("Making windows..."); t0 = time.time()
def make_windows(X, y, lag=LAG, horizon=HORIZON):
    idx = np.arange(lag, len(X) - horizon)
    Xw  = np.stack([X[i-lag:i] for i in idx]).astype(np.float32)
    yw  = np.stack([y[i+horizon] for i in idx]).astype(np.float32)
    return Xw, yw

X_train_w, y_train = make_windows(X_raw[:train_end],         y_raw[:train_end])
X_test_w,  y_test  = make_windows(X_raw[train_end:test_end], y_raw[train_end:test_end])
X_val_w,   y_val   = make_windows(X_raw[test_end:],          y_raw[test_end:])
print(f"Windows done in {time.time()-t0:.1f}s")
print(f"train {X_train_w.shape} | test {X_test_w.shape} | val {X_val_w.shape}")

# ── 5. SCALE (fast numpy) ─────────────────────────────────────────────────────
print("Scaling..."); t0 = time.time()
def fit_scale(Xw):
    N, T, C, H, W = Xw.shape
    flat = Xw.reshape(-1, C)
    return flat.mean(axis=0), flat.std(axis=0) + 1e-8

def transform(Xw, sc):
    mean, std = sc
    N, T, C, H, W = Xw.shape
    return ((Xw.reshape(-1, C) - mean) / std).reshape(N, T, C, H, W).astype(np.float32)

scaler  = fit_scale(X_train_w)
X_tr_sc = transform(X_train_w, scaler)
X_te_sc = transform(X_test_w,  scaler)
X_va_sc = transform(X_val_w,   scaler)
print(f"Scaling done in {time.time()-t0:.1f}s")

# Skip y normalization — use log-mm directly
y_tr_n = y_train
y_te_n = y_test
y_va_n = y_val
y_mean, y_std = 0.0, 1.0

# ── 6. DATASET ────────────────────────────────────────────────────────────────
class SeqDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
    def __len__(self):        return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

def loader(X, y, shuffle):
    return DataLoader(SeqDS(X, y), batch_size=BATCH, shuffle=shuffle, num_workers=0)

# ── 7. MODEL ──────────────────────────────────────────────────────────────────
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

# ── 8. TRAIN HELPER ───────────────────────────────────────────────────────────
def run_training(model, X_tr, y_tr, X_v, y_v, lr, max_epochs, patience, trial=None):
    tr_loader = loader(X_tr, y_tr, shuffle=True)
    va_loader = loader(X_v,  y_v,  shuffle=False)
    opt       = torch.optim.Adam(model.parameters(), lr=lr)
    sched     = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
    best_val, no_improve = float('inf'), 0

    for epoch in range(max_epochs):
        model.train()
        for xb, yb in tr_loader:
            opt.zero_grad()
            F.mse_loss(model(xb), yb).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            avg = np.mean([F.mse_loss(model(xb), yb).item() for xb, yb in va_loader])
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

# ── SKIP OPTUNA — use fixed params ────────────────────────────────────────────
bp = {'hidden': 64, 'kernel': 3, 'lr': 1e-4}
print(f"Using fixed params: {bp}")

# ── 10. FINAL TRAINING ────────────────────────────────────────────────────────
model     = ConvLSTMModel(N_FEATURES, bp['hidden'], bp['kernel'])
tr_loader = loader(X_tr_sc, y_tr_n, shuffle=True)
va_loader = loader(X_va_sc, y_va_n, shuffle=False)
opt       = torch.optim.Adam(model.parameters(), lr=bp['lr'])
sched     = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=6, factor=0.5)

train_losses, val_losses    = [], []
best_val, no_improve        = float('inf'), 0
best_state                  = copy.deepcopy(model.state_dict())

print(f"\nFinal training — up to {FINAL_EPOCHS} epochs …")
for epoch in range(FINAL_EPOCHS):
    t0 = time.time()
    model.train()
    tl = []
    for xb, yb in tr_loader:
        opt.zero_grad()
        loss = F.mse_loss(model(xb), yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tl.append(loss.item())
    train_losses.append(np.mean(tl))

    model.eval()
    with torch.no_grad():
        vl = [F.mse_loss(model(xb), yb).item() for xb, yb in va_loader]
    avg_val = float(np.mean(vl))
    val_losses.append(avg_val)
    print(f"  Epoch {epoch+1:03d} | train {train_losses[-1]:.5f} | val {avg_val:.5f} | {time.time()-t0:.1f}s")

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
print("Done!")

# At end of training script — save losses for evaluate.py
np.save('X_te_sc.npy',  X_te_sc)
np.save('X_va_sc.npy',  X_va_sc)
np.save('y_test.npy',   y_test)
np.save('y_val.npy',    y_val)
np.save('train_losses.npy', np.array(train_losses))
np.save('val_losses.npy',   np.array(val_losses))
print("Saved all evaluation arrays")