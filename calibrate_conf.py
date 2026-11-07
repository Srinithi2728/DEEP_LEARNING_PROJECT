import json, numpy as np, torch, torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sources6 import EyeC, EMGC, EEGC, crop_logits, device
from train_eeg import zs

rng = np.random.default_rng(0)


def balanced(y, cap):
    idx = []
    for k in np.unique(y):
        i = np.where(y == k)[0]
        idx.extend(rng.choice(i, min(cap, len(i)), replace=False))
    return np.array(sorted(idx))


def eye_val(s):
    labels = np.array(s.ds.targets); idx = np.arange(len(labels))
    _, tmp = train_test_split(idx, test_size=0.3, stratify=labels, random_state=42)
    va, _ = train_test_split(tmp, test_size=0.5, stratify=labels[tmp], random_state=42)
    va = va[balanced(labels[va], 250)]
    zl, yl = [], []
    with torch.no_grad():
        for i in va:
            x, y = s.ds[int(i)]
            zl.append(s.model(x.unsqueeze(0).to(device))[0].cpu()); yl.append(y)
    return torch.stack(zl), torch.tensor(yl)


def emg_val(s):
    ck = torch.load("models/emg_model_cmd.pt", map_location=device, weights_only=False)
    d = np.load("data/emg/windows.npz"); keep = list(ck["classes"])
    m = np.isin(d["y"], keep) & (d["rep"] == 10)                     # validation repetition
    X = ((d["X"][m] - ck["mean"]) / ck["std"]).astype(np.float32)
    y = np.searchsorted(keep, d["y"][m])
    sel = balanced(y, 1500); X = X[sel]
    with torch.no_grad():
        z = torch.cat([s.model(torch.tensor(X[i:i + 1024]).to(device)).cpu() for i in range(0, len(X), 1024)])
    return z, torch.tensor(y[sel])


def eeg_val(s):
    d = np.load("data/eeg/bci2a.npz"); y = d["y"]
    pool = np.where(~d["test"])[0]
    _, va = train_test_split(pool, test_size=0.1, stratify=y[pool], random_state=42)   # same as training
    return crop_logits(s.model, torch.tensor(zs(d["X"][va])), s.T, device), torch.tensor(y[va])


def fit_temp(z, y):
    logT = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=200)
    def closure():
        opt.zero_grad(); loss = F.cross_entropy(z / logT.exp(), y); loss.backward(); return loss
    opt.step(closure)
    return float(logT.exp())


def bins_of(p, y, bins=10):
    conf, pred = p.max(1); acc = (pred == y).float()
    edges = torch.linspace(0, 1, bins + 1)
    return conf, acc, [((conf > lo) & (conf <= hi)) for lo, hi in zip(edges[:-1], edges[1:])]


def ece(p, y):
    conf, acc, ms = bins_of(p, y)
    return sum(m.float().mean().item() * abs(acc[m].mean().item() - conf[m].mean().item()) for m in ms if m.any())


def reliability(p, y):
    conf, acc, ms = bins_of(p, y)
    pts = [(conf[m].mean().item(), acc[m].mean().item()) for m in ms if m.sum() >= 5]
    return [a for a, _ in pts], [b for _, b in pts]


if __name__ == "__main__":
    src = {"eye": EyeC(), "emg": EMGC(), "eeg": EEGC()}
    getters = {"eye": eye_val, "emg": emg_val, "eeg": eeg_val}
    temps = {}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (m, s) in zip(axes, src.items()):
        z, y = getters[m](s)
        t = fit_temp(z, y); temps[m] = t
        p0, p1 = F.softmax(z, 1), F.softmax(z / t, 1)
        print(f"{m}: {len(y)} samples | T = {t:.2f} | ECE {ece(p0, y):.3f} -> {ece(p1, y):.3f} "
              f"| NLL {F.cross_entropy(z, y):.3f} -> {F.cross_entropy(z / t, y):.3f}")
        for p, lab in [(p0, "before"), (p1, "after")]:
            xs, ys = reliability(p, y); ax.plot(xs, ys, "o-", label=lab)
        ax.plot([0, 1], [0, 1], "k--"); ax.set_title(m)
        ax.set_xlabel("confidence"); ax.set_ylabel("accuracy"); ax.legend()
    json.dump(temps, open("models/temps.json", "w"))
    plt.tight_layout(); plt.savefig("results/phase6_reliability.png", dpi=150)
    print("saved models/temps.json and results/phase6_reliability.png")