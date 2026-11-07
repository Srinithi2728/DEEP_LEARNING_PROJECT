import json, copy, numpy as np, torch, torch.nn as nn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from train_eeg import EEGNet, crop_probs, zs

device = "cuda" if torch.cuda.is_available() else "cpu"
FT_EPOCHS, FT_LR, SC_EPOCHS, SC_LR, BATCH = 40, 3e-4, 120, 1e-3, 32
torch.manual_seed(0); np.random.seed(0)


def fit(model, Xtr, ytr, Xva, yva, epochs, lr, T):
    L = Xtr.shape[2]
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best, state = 1e9, None
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            o = np.random.randint(0, L - T + 1)
            xb = Xtr[b][:, :, o:o + T].to(device)
            xb = xb + 0.1 * torch.randn_like(xb)
            loss = crit(model(xb), ytr[b].to(device))
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        p = crop_probs(model, Xva, T, device)
        nll = -np.log(p[np.arange(len(yva)), yva] + 1e-9).mean()
        if nll < best:
            best, state = nll, copy.deepcopy(model.state_dict())
    model.load_state_dict(state)
    return model


def acc(model, X, y, T):
    return float((crop_probs(model, X, T, device).argmax(1) == y).mean())


def main():
    d = np.load("data/eeg/bci2a.npz")
    X, y, subj, test = zs(d["X"]), d["y"], d["subj"], d["test"]
    classes = [str(c) for c in d["classes"]]
    ck = torch.load("models/eeg_model_session.pt", weights_only=False)
    T, nc = ck["T"], len(classes)

    rows = []
    print(f"{'subject':>8}{'pooled':>9}{'fine-tuned':>12}{'scratch':>9}")
    for s in np.unique(subj):
        pool = np.where((subj == s) & ~test)[0]
        te = np.where((subj == s) & test)[0]
        tr, va = train_test_split(pool, test_size=0.15, stratify=y[pool], random_state=0)
        Xtr, ytr = torch.tensor(X[tr]), torch.tensor(y[tr])
        Xva, yva = torch.tensor(X[va]), y[va]
        Xte, yte = torch.tensor(X[te]), y[te]

        pooled = EEGNet(nc, ck["ch"], T).to(device); pooled.load_state_dict(ck["state"])
        a_pool = acc(pooled, Xte, yte, T)
        ft = fit(copy.deepcopy(pooled), Xtr, ytr, Xva, yva, FT_EPOCHS, FT_LR, T)
        a_ft = acc(ft, Xte, yte, T)
        torch.save({"state": ft.state_dict(), "classes": classes, "T": T, "ch": ck["ch"], "fs": 128,
                    "split": f"subject-{s}", "test_idx": te}, f"models/eeg_subject_{int(s)}.pt")
        sc = fit(EEGNet(nc, ck["ch"], T).to(device), Xtr, ytr, Xva, yva, SC_EPOCHS, SC_LR, T)
        a_sc = acc(sc, Xte, yte, T)

        print(f"{int(s):>8}{a_pool:>9.3f}{a_ft:>12.3f}{a_sc:>9.3f}", flush=True)
        rows.append((int(s), a_pool, a_ft, a_sc))

    r = np.array(rows)
    print(f"{'mean':>8}{r[:, 1].mean():>9.3f}{r[:, 2].mean():>12.3f}{r[:, 3].mean():>9.3f}")
    json.dump(rows, open("results/eeg_finetune.json", "w"))
    x = np.arange(len(r)); w = 0.27
    for k, lab in enumerate(["pooled", "fine-tuned", "scratch"]):
        plt.bar(x + (k - 1) * w, r[:, k + 1], w, label=lab)
    plt.axhline(0.25, ls="--", c="gray"); plt.xticks(x, r[:, 0].astype(int))
    plt.xlabel("subject"); plt.ylabel("test accuracy"); plt.legend(); plt.tight_layout()
    plt.savefig("results/eeg_finetune.png", dpi=150)


if __name__ == "__main__":
    main()