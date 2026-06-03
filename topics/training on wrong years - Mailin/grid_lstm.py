# ── IMPORTS ───────────────────────────────────────────────────────────────────
import copy, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import cfgrib
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
MODEL_PATH    = 'best_precip_model_1940.pt'
LAG           = 24
HORIZON       = 24
FINAL_EPOCHS  = 15
BATCH         = 512 (32)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── 1. LOAD ───────────────────────────────────────────────────────────────────
t0 = time.time()

datasets = cfgrib.open_datasets("data/grid-timeseries_sel_vars_1940.grib")
ds0, ds1 = datasets

ref_lat  = ds0.latitude
ref_lon  = ds0.longitude
ref_time = ds0.time

channels, chan_names = [], []

for var in ['sp', 'tcc', 'u10', 'v10', 't2m']:
    channels.append(ds0[var].values.astype(np.float32))
    chan_names.append(var)

for var in ['ssrd', 'tp']:
    arr = ds1[var].sum(dim='step')
    arr_daily = xr.DataArray(
        arr.values / 24,
        coords={'time': ds1[var].time.values,
                'latitude': ref_lat,
                'longitude': ref_lon},
        dims=['time', 'latitude', 'longitude']
    )
    arr_h = arr_daily.reindex(time=ref_time, method='ffill').ffill(dim='time').bfill(dim='time')
    channels.append(arr_h.values.astype(np.float32))
    chan_names.append(var)

print(f"Load done in {time.time()-t0:.1f}s")

# ── 2. BUILD X, y ─────────────────────────────────────────────────────────────
t0 = time.time()

X_raw = np.stack(channels, axis=1).astype(np.float32)
tp_idx = chan_names.index('tp')

y_raw = channels[tp_idx].copy()

# clean NaNs
for i in range(X_raw.shape[1]):
    m = np.nanmean(X_raw[:, i])
    X_raw[:, i] = np.nan_to_num(X_raw[:, i], nan=m)

y_raw = np.nan_to_num(y_raw, nan=np.nanmean(y_raw))

# log transform
y_raw = np.log1p(y_raw * 1000)
X_raw[:, tp_idx] = np.log1p(X_raw[:, tp_idx] * 1000)

N = X_raw.shape[0]
N_FEATURES = X_raw.shape[1]

print(f"Built arrays in {time.time()-t0:.1f}s")

# ── 3. SPLIT ──────────────────────────────────────────────────────────────────
train_end = int(0.70 * N)
test_end  = int(0.90 * N)

# ── 4. WINDOWS ────────────────────────────────────────────────────────────────
def make_windows(X, y, lag=LAG, horizon=HORIZON):
    idx = np.arange(lag, len(X) - horizon)
    Xw = np.stack([X[i-lag:i] for i in idx]).astype(np.float32)
    yw = np.stack([y[i+horizon] for i in idx]).astype(np.float32)
    return Xw, yw

X_train_w, y_train = make_windows(X_raw[:train_end], y_raw[:train_end])
X_test_w,  y_test  = make_windows(X_raw[train_end:test_end], y_raw[train_end:test_end])
X_val_w,   y_val   = make_windows(X_raw[test_end:], y_raw[test_end:])

# ── 5. SCALE ──────────────────────────────────────────────────────────────────
def fit_scale(Xw):
    flat = Xw.reshape(-1, Xw.shape[-3])
    return flat.mean(0), flat.std(0) + 1e-8

def transform(Xw, sc):
    m, s = sc
    return ((Xw.reshape(-1, Xw.shape[-3]) - m) / s).reshape(Xw.shape).astype(np.float32)

scaler = fit_scale(X_train_w)

X_tr_sc = transform(X_train_w, scaler)
X_te_sc = transform(X_test_w, scaler)
X_va_sc = transform(X_val_w, scaler)

y_tr, y_te, y_va = y_train, y_test, y_val

# ── 6. DATASET ────────────────────────────────────────────────────────────────
class SeqDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

def loader(X, y, shuffle):
    return DataLoader(SeqDS(X, y), batch_size=BATCH, shuffle=shuffle)

# ── 7. MODEL ──────────────────────────────────────────────────────────────────
class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hidden, ks=3):
        super().__init__()
        self.hidden = hidden
        self.conv = nn.Conv2d(in_ch + hidden, 4 * hidden, ks, padding=ks//2)

    def forward(self, x, h, c):
        i, f, o, g = self.conv(torch.cat([x, h], dim=1)).chunk(4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c = f * c + i * g
        h = o * torch.tanh(c)

        return h, c


class ConvLSTMModel(nn.Module):
    def __init__(self, in_ch, hidden=64):
        super().__init__()
        self.hidden = hidden
        self.cell = ConvLSTMCell(in_ch, hidden)

        self.norm = nn.GroupNorm(8, hidden)

        self.head = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden // 2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden // 2, 1, 1)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape

        h = torch.zeros(B, self.hidden, H, W, device=x.device)
        c = torch.zeros(B, self.hidden, H, W, device=x.device)

        for t in range(T):
            h, c = self.cell(x[:, t], h, c)

        h = self.norm(h)
        return self.head(h).squeeze(1)

# ── LOSS ──────────────────────────────────────────────────────────────────────
def rain_loss(pred, target):
    weight = 1.0 + 5.0 * (target > 0).float()
    return (weight * (pred - target) ** 2).mean()

# ── 8. HYPERPARAM SEARCH ──────────────────────────────────────────────────────
def train_and_eval(hidden, lr):
    model = ConvLSTMModel(N_FEATURES, hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    tr_loader = loader(X_tr_sc, y_tr, True)
    va_loader = loader(X_va_sc, y_va, False)

    best_val = float("inf")

    for _ in range(6):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)

            opt.zero_grad()
            loss = rain_loss(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        vals = []
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(device), yb.to(device)
                vals.append(rain_loss(model(xb), yb).item())

        best_val = min(best_val, np.mean(vals))

    return best_val

param_grid = {
    "hidden": [32, 64, 128],
    "lr": [1e-3, 3e-4, 1e-4],
}

best_score = float("inf")
best_params = None

for h in param_grid["hidden"]:
    for lr in param_grid["lr"]:
        score = train_and_eval(h, lr)

        print(f"h={h}, lr={lr} → {score:.5f}")

        if score < best_score:
            best_score = score
            best_params = (h, lr)

print("\nBEST:", best_params)

# ── 9. FINAL TRAINING ─────────────────────────────────────────────────────────
best_hidden, best_lr = best_params

model = ConvLSTMModel(N_FEATURES, best_hidden).to(device)
opt = torch.optim.Adam(model.parameters(), lr=best_lr)

tr_loader = loader(X_tr_sc, y_tr, True)
va_loader = loader(X_va_sc, y_va, False)

best_val = float("inf")
best_state = copy.deepcopy(model.state_dict())

train_losses, val_losses = [], []

for epoch in range(FINAL_EPOCHS):
    model.train()
    tl = []

    for xb, yb in tr_loader:
        xb, yb = xb.to(device), yb.to(device)

        opt.zero_grad()
        loss = rain_loss(model(xb), yb)
        loss.backward()
        opt.step()

        tl.append(loss.item())

    train_losses.append(np.mean(tl))

    model.eval()
    vl = []

    with torch.no_grad():
        for xb, yb in va_loader:
            xb, yb = xb.to(device), yb.to(device)
            vl.append(rain_loss(model(xb), yb).item())

    val = np.mean(vl)
    val_losses.append(val)

    print(f"Epoch {epoch+1}: train={train_losses[-1]:.5f}, val={val:.5f}")

    if val < best_val:
        best_val = val
        best_state = copy.deepcopy(model.state_dict())
        torch.save(model.state_dict(), MODEL_PATH)

model.load_state_dict(best_state)
print("Done")

# ── SAVE ──────────────────────────────────────────────────────────────────────
np.save('X_te_sc.npy', X_te_sc)
np.save('X_va_sc.npy', X_va_sc)
np.save('y_test.npy', y_test)
np.save('y_val.npy', y_val)
np.save('train_losses.npy', np.array(train_losses))
np.save('val_losses.npy', np.array(val_losses))

print("Saved all evaluation arrays")