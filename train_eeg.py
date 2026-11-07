import numpy as np, torch, torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt, seaborn as sns

SPLIT = "session"          # "session" or "subject"
T_CROP, EPOCHS, BATCH, LR = 384, 120, 64, 1e-3     # 3 s crops of the 4 s trial


def zs(a):                  # per-trial, per-channel z-score
    return ((a - a.mean(-1, keepdims=True)) / (a.std(-1, keepdims=True) + 1e-6)).astype(np.float32)


class EEGNet(nn.Module):
    def __init__(self, nc, ch=22, T=384, F1=8, D=2, F2=16, k=64, drop=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, F1, (1, k), padding="same", bias=False), nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (ch, 1), groups=F1, bias=False), nn.BatchNorm2d(F1 * D),
            nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(drop),
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding="same", groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, 1, bias=False), nn.BatchNorm2d(F2),
            nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(drop), nn.Flatten())
        with torch.no_grad():
            n = self.net(torch.zeros(1, 1, ch, T)).shape[1]
        self.fc = nn.Linear(n, nc)

    def forward(self, x):                       # (B, ch, T)
        return self.fc(self.net(x.unsqueeze(1)))


def crop_probs(model, Xs, T, device):
    """Softmax averaged over 3 crops (start, middle, end) of each trial."""
    model.eval(); L = Xs.shape[2]
    offs = sorted({0, (L - T) // 2, L - T}); out = []
    with torch.no_grad():
        for i in range(0, len(Xs), 256):
            xb = Xs[i:i + 256].to(device)
            p = sum(torch.softmax(model(xb[:, :, o:o + T]), 1) for o in offs) / len(offs)
            out.append(p.cpu())
    return torch.cat(out).numpy()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = np.load("data/eeg/bci2a.npz")
    X, y, subj, test, classes = zs(d["X"]), d["y"], d["subj"], d["test"], [str(c) for c in d["classes"]]
    L, nc = X.shape[2], len(classes)

    if SPLIT == "session":
        pool = np.where(~test)[0]
        tr_idx, va_idx = train_test_split(pool, test_size=0.1, stratify=y[pool], random_state=42)
        te_idx = np.where(test)[0]
    else:
        tr_idx, va_idx, te_idx = np.where(subj <= 6)[0], np.where(subj == 7)[0], np.where(subj >= 8)[0]
    print(f"split: {SPLIT} | trials train/val/test: {len(tr_idx)}/{len(va_idx)}/{len(te_idx)} | chance = {1/nc:.2f}")

    Xtr, ytr = torch.tensor(X[tr_idx]), torch.tensor(y[tr_idx])
    Xva, yva = torch.tensor(X[va_idx]), y[va_idx]
    Xte, yte = torch.tensor(X[te_idx]), y[te_idx]

    model = EEGNet(nc, X.shape[1], T_CROP).to(device)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = torch.optim.AdamW(model.parameters(), LR, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)

    best = 0
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            o = np.random.randint(0, L - T_CROP + 1)            # random time crop
            xb = Xtr[b][:, :, o:o + T_CROP].to(device)
            xb = xb + 0.1 * torch.randn_like(xb)
            loss = crit(model(xb), ytr[b].to(device))
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        pv = crop_probs(model, Xva, T_CROP, device).argmax(1)
        f1 = f1_score(yva, pv, average="macro")
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"Epoch {ep+1:03d} | val acc {(pv == yva).mean():.3f} | val macro-F1 {f1:.3f}", flush=True)
        if f1 > best:
            best = f1
            torch.save({"state": model.state_dict(), "classes": classes, "T": T_CROP, "ch": X.shape[1],
                        "fs": 128, "split": SPLIT, "test_idx": te_idx}, f"models/eeg_model_{SPLIT}.pt")

    ck = torch.load(f"models/eeg_model_{SPLIT}.pt", weights_only=False)
    model.load_state_dict(ck["state"])
    pt = crop_probs(model, Xte, T_CROP, device).argmax(1)
    print(f"\nTEST acc {(pt == yte).mean():.3f} | macro-F1 {f1_score(yte, pt, average='macro'):.3f}")
    print(classification_report(yte, pt, target_names=classes, zero_division=0))
    for s in np.unique(subj[te_idx]):
        m = subj[te_idx] == s
        print(f"  subject {s}: acc {(pt[m] == yte[m]).mean():.3f}")
    sns.heatmap(confusion_matrix(yte, pt), annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.savefig(f"results/eeg_confusion_{SPLIT}.png", dpi=150)


if __name__ == "__main__":
    main()