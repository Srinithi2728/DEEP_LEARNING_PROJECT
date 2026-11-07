import torch, torch.nn as nn, numpy as np
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt, seaborn as sns

DATA_DIR, SIZE, BATCH, EPOCHS, LR = "data/eye", (128, 192), 64, 12, 1e-3
WORKERS = 2          # set to 0 if you still get multiprocessing errors


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.Grayscale(3), transforms.Resize(SIZE),
        transforms.ColorJitter(brightness=0.4, contrast=0.4),
        transforms.RandomRotation(5),
        transforms.ToTensor(), norm])
    eval_tf = transforms.Compose([
        transforms.Grayscale(3), transforms.Resize(SIZE),
        transforms.ToTensor(), norm])

    full_train = datasets.ImageFolder(DATA_DIR, transform=train_tf)
    full_eval = datasets.ImageFolder(DATA_DIR, transform=eval_tf)
    classes = full_train.classes
    labels = np.array(full_train.targets)
    idx = np.arange(len(labels))
    print("classes:", classes, "| counts:", np.bincount(labels))
    assert len(classes) == 6, "Expected 6 classes - check data/eye for extra folders"

    tr, tmp = train_test_split(idx, test_size=0.3, stratify=labels, random_state=42)
    va, te = train_test_split(tmp, test_size=0.5, stratify=labels[tmp], random_state=42)

    mk = lambda ds, sh: DataLoader(ds, BATCH, shuffle=sh, num_workers=WORKERS,
                                   persistent_workers=WORKERS > 0)
    train_dl = mk(Subset(full_train, tr), True)
    val_dl = mk(Subset(full_eval, va), False)
    test_dl = mk(Subset(full_eval, te), False)

    counts = np.bincount(labels[tr])
    w = torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float32).to(device)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, len(classes)))
    model = model.to(device)

    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)

    def run(dl, train=False):
        model.train(train)
        loss_sum, n, P, G = 0, 0, [], []
        with torch.set_grad_enabled(train):
            for x, y in dl:
                x, y = x.to(device), y.to(device)
                out = model(x); loss = crit(out, y)
                if train:
                    opt.zero_grad(); loss.backward(); opt.step()
                loss_sum += loss.item() * len(y); n += len(y)
                P += out.argmax(1).cpu().tolist(); G += y.cpu().tolist()
        return loss_sum / n, f1_score(G, P, average="macro"), P, G

    best = 0
    for ep in range(EPOCHS):
        _, tf1, _, _ = run(train_dl, True); sched.step()
        vl, vf1, _, _ = run(val_dl)
        print(f"Epoch {ep+1:02d} | train F1 {tf1:.3f} | val loss {vl:.3f} F1 {vf1:.3f}", flush=True)
        if vf1 > best:
            best = vf1
            torch.save({"state": model.state_dict(), "classes": classes, "size": SIZE},
                       "models/eye_model.pt")

    model.load_state_dict(torch.load("models/eye_model.pt")["state"])
    _, f1, P, G = run(test_dl)
    print(f"TEST macro-F1: {f1:.3f}")
    print(classification_report(G, P, target_names=classes))
    sns.heatmap(confusion_matrix(G, P), annot=True, fmt="d",
                xticklabels=classes, yticklabels=classes, cmap="Blues")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.savefig("results/eye_confusion.png", dpi=150)
    print("saved results/eye_confusion.png")


if __name__ == "__main__":
    main()