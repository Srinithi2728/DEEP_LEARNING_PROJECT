import glob, os, re, copy, numpy as np, torch, torch.nn as nn
from PIL import Image
from torchvision import transforms, models
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt, seaborn as sns

device = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS, BATCH, LR = 15, 32, 3e-4
torch.manual_seed(0); np.random.seed(0)

norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
train_tf = transforms.Compose([
    transforms.Grayscale(3),
    transforms.RandomAffine(5, translate=(0.04, 0.04), scale=(0.95, 1.05)),
    transforms.ColorJitter(0.4, 0.4), transforms.ToTensor(), norm])
eval_tf = transforms.Compose([transforms.Grayscale(3), transforms.ToTensor(), norm])


class DS(torch.utils.data.Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf
        self.imgs = [Image.open(p).convert("L") for p, _ in items]
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.tf(self.imgs[i]), self.items[i][1]


def main():
    base = torch.load("models/eye_model.pt", map_location=device, weights_only=False)
    DROP = {"closed"}                       # blink is detected from eyelid geometry instead
    classes = [c for c in base["classes"] if c not in DROP]
    items = []
    for ci, c in enumerate(classes):
        for f in sorted(glob.glob(f"data/webcam/{c}/r*_*.png")):
            items.append((f, ci, int(re.match(r"r(\d+)_", os.path.basename(f)).group(1))))
    rounds = sorted({r for *_, r in items})
    print("classes:", classes, "| images:", len(items), "| rounds:", rounds)
    assert len(rounds) >= 4, "need at least 4 rounds (train / validation / test)"
    test_r, val_r = rounds[-1], rounds[-2]
    pick = lambda cond: [(p, y) for p, y, r in items if cond(r)]
    tr, va, te = pick(lambda r: r < val_r), pick(lambda r: r == val_r), pick(lambda r: r == test_r)
    for name, s in [("train", tr), ("val", va), ("test", te)]:
        print(f"{name}: {len(s)} images | per class {np.bincount([y for _, y in s], minlength=len(classes))}")

    mk = lambda s, tf, sh: torch.utils.data.DataLoader(DS(s, tf), BATCH, shuffle=sh)
    train_dl, val_dl, test_dl = mk(tr, train_tf, True), mk(va, eval_tf, False), mk(te, eval_tf, False)

    def build(init):
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if init == "imagenet" else None)
        m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, len(classes)))
        if init == "popeye":                # keep the PopEYE backbone, fresh output layer
            m.load_state_dict({k: v for k, v in base["state"].items() if not k.startswith("fc.")}, strict=False)
        return m.to(device)

    def predict(m, dl):
        m.eval(); P, G = [], []
        with torch.no_grad():
            for x, y in dl:
                P += m(x.to(device)).argmax(1).cpu().tolist(); G += y.tolist()
        return np.array(P), np.array(G)

    def train(m):
        opt = torch.optim.AdamW(m.parameters(), LR, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
        crit = nn.CrossEntropyLoss()
        best, state = -1, None
        for _ in range(EPOCHS):
            m.train()
            for x, y in train_dl:
                loss = crit(m(x.to(device)), y.to(device))
                opt.zero_grad(); loss.backward(); opt.step()
            sched.step()
            P, G = predict(m, val_dl); f1 = f1_score(G, P, average="macro")
            if f1 > best: best, state = f1, copy.deepcopy(m.state_dict())
        m.load_state_dict(state)
        return m, best

    print(f"\n{'model':32s}{'val F1':>8s}{'test acc':>10s}{'test F1':>9s}")

    best_m, best_v, best_name = None, -1, ""
    for init, label in [("popeye", "PopEYE + webcam fine-tune"), ("imagenet", "ImageNet + webcam fine-tune")]:
        m, v = train(build(init))
        P, G = predict(m, test_dl)
        print(f"{label:32s}{v:>8.3f}{(P == G).mean():>10.3f}{f1_score(G, P, average='macro'):>9.3f}", flush=True)
        if v > best_v: best_m, best_v, best_name = m, v, label

    P, G = predict(best_m, test_dl)
    print(f"\nSaved model: {best_name} (chosen on validation F1)\n")
    print(classification_report(G, P, target_names=classes, zero_division=0))
    sns.heatmap(confusion_matrix(G, P), annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.savefig("results/eye_webcam_confusion.png", dpi=150)
    torch.save({"state": best_m.state_dict(), "classes": classes, "size": base["size"]}, "models/eye_webcam.pt")


if __name__ == "__main__":
    main()
    