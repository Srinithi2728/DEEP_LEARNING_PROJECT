import numpy as np, torch, torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt, seaborn as sns

MODEL = "cnn"            # "cnn" or "cnn_lstm"
EPOCHS, BATCH, LR = 30, 256, 1e-3


class EMGNet(nn.Module):
    def __init__(self, ch, nc, kind):
        super().__init__()
        self.kind = kind
        def blk(i, o, k, pool=True):
            L = [nn.Conv1d(i, o, k, padding=k // 2), nn.BatchNorm1d(o), nn.ReLU()]
            return nn.Sequential(*(L + [nn.MaxPool1d(2)] if pool else L))
        if kind == "cnn":
            self.f = nn.Sequential(blk(ch, 64, 5), blk(64, 128, 5), blk(128, 128, 3),
                                   nn.AdaptiveAvgPool1d(1), nn.Flatten())
            feat = 128
        else:
            self.f = nn.Sequential(blk(ch, 64, 5, False), blk(64, 64, 3, False))
            self.lstm = nn.LSTM(64, 128, batch_first=True, bidirectional=True)
            feat = 256
        self.h = nn.Sequential(nn.Dropout(0.4), nn.Linear(feat, nc))

    def forward(self, x):
        z = self.f(x)
        if self.kind == "cnn_lstm":
            o, _ = self.lstm(z.transpose(1, 2))     # (B, T, C)
            z = o.mean(1)
        return self.h(z)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = np.load("data/emg/windows.npz")
    X, y, rep = d["X"], d["y"], d["rep"]
    KEEP = sorted([0, 3, 4, 6, 14, 16])    # 0 = rest; replace the rest with your selected set
    keep_m = np.isin(y, KEEP)
    X, rep = X[keep_m], rep[keep_m]
    y = np.searchsorted(KEEP, y[keep_m])
    classes = np.array(KEEP)
    nc = len(classes)

    tr, va, te = np.isin(rep, [1, 3, 4, 6, 8, 9]), rep == 10, np.isin(rep, [2, 5, 7])
    print("model:", MODEL, "| train/val/test:", tr.sum(), va.sum(), te.sum(), "| classes:", nc)

    mean = X[tr].mean(axis=(0, 2), keepdims=True)
    std = X[tr].std(axis=(0, 2), keepdims=True) + 1e-6
    T = lambda m: (torch.tensor((X[m] - mean) / std).to(device), torch.tensor(y[m]).to(device))
    (Xtr, ytr), (Xva, yva), (Xte, yte) = T(tr), T(va), T(te)

    cnt = np.bincount(y[tr], minlength=nc)
    w = torch.tensor(cnt.sum() / (nc * np.maximum(cnt, 1)), dtype=torch.float32).to(device)
    model = EMGNet(X.shape[1], nc, MODEL).to(device)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), LR, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)

    def predict(Xs):
        model.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(Xs), 1024):
                out.append(model(Xs[i:i + 1024]).argmax(1))
        return torch.cat(out).cpu().numpy()

    best = 0
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(Xtr), device=device)
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            xb = Xtr[b] * (0.8 + 0.4 * torch.rand(len(b), 1, 1, device=device))
            xb = xb + 0.05 * torch.randn_like(xb)
            loss = crit(model(xb), ytr[b])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        pv, gv = predict(Xva), yva.cpu().numpy()
        f1 = f1_score(gv, pv, average="macro")
        print(f"Epoch {ep+1:02d} | val acc {(pv == gv).mean():.3f} | val macro-F1 {f1:.3f}", flush=True)
        if f1 > best:
            best = f1
            torch.save({"state": model.state_dict(), "classes": classes, "kind": MODEL,
                        "mean": mean, "std": std, "fs": 100, "win": 30}, f"models/emg_model_cmd.pt")

    model.load_state_dict(torch.load(f"models/emg_model_cmd.pt", weights_only=False)["state"])
    pt, gt = predict(Xte), yte.cpu().numpy()
    print(f"TEST acc {(pt == gt).mean():.3f} | macro-F1 {f1_score(gt, pt, average='macro'):.3f}")
    print(classification_report(gt, pt, zero_division=0))
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix(gt, pt), annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.savefig(f"results/emg_confusion_cmd.png", dpi=150)


if __name__ == "__main__":
    main()