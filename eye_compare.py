import glob, os, re, json, joblib, numpy as np, torch, torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt, seaborn as sns

device = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load("models/eye_webcam.pt", map_location=device, weights_only=False)
classes = ck["classes"]
cnn = models.resnet18(); cnn.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, len(classes)))
cnn.load_state_dict(ck["state"]); cnn = cnn.to(device).eval()
tf = transforms.Compose([transforms.Grayscale(3), transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

items = []
for ci, c in enumerate(classes):
    for f in sorted(glob.glob(f"data/webcam/{c}/r*_*.png")):
        if os.path.exists(f[:-4] + ".npy"):
            items.append((f, np.load(f[:-4] + ".npy"), ci, int(re.match(r"r(\d+)_", os.path.basename(f)).group(1))))
rounds = sorted({r for *_, r in items})
assert len(rounds) >= 4, "need at least 4 rounds"
test_r, val_r = rounds[-1], rounds[-2]
sel = lambda cond: [it for it in items if cond(it[3])]
tr, va, te = sel(lambda r: r < val_r), sel(lambda r: r == val_r), sel(lambda r: r == test_r)
print("train/val/test:", len(tr), len(va), len(te))
for n, s in [("train", tr), ("test", te)]:
    print(f"  {n} per class:", dict(zip(classes, np.bincount([x[2] for x in s], minlength=len(classes)))))


def cnn_probs(s):
    out = []
    with torch.no_grad():
        for i in range(0, len(s), 64):
            x = torch.stack([tf(Image.open(p).convert("L")) for p, *_ in s[i:i + 64]]).to(device)
            out.append(torch.softmax(cnn(x), 1).cpu().numpy())
    return np.concatenate(out)


Xf = lambda s: np.nan_to_num(np.stack([x[1] for x in s]))
y = lambda s: np.array([x[2] for x in s])
cands = {"logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
         "mlp": make_pipeline(StandardScaler(), MLPClassifier((32, 32), alpha=1e-2, max_iter=1000, random_state=0))}
best_f, best_v = None, -1
for name, m in cands.items():
    m.fit(Xf(tr), y(tr))
    assert list(m.classes_) == list(range(len(classes))), "a class is missing from the training rounds"
    v = f1_score(y(va), m.predict(Xf(va)), average="macro")
    print(f"feature model {name}: val F1 {v:.3f}")
    if v > best_v: best_f, best_v = m, v

P = {"val": {"cnn": cnn_probs(va), "feat": best_f.predict_proba(Xf(va))},
     "test": {"cnn": cnn_probs(te), "feat": best_f.predict_proba(Xf(te))}}
for k in P: P[k]["fused"] = (P[k]["cnn"] + P[k]["feat"]) / 2

print(f"\n{'model':22s}{'val F1':>8s}{'test acc':>10s}{'test F1':>9s}")
res = {}
for mode, label in [("cnn", "CNN (webcam)"), ("feat", "iris geometry"), ("fused", "CNN + geometry")]:
    v = f1_score(y(va), P["val"][mode].argmax(1), average="macro")
    pt = P["test"][mode].argmax(1)
    res[mode] = v
    print(f"{label:22s}{v:>8.3f}{(pt == y(te)).mean():>10.3f}{f1_score(y(te), pt, average='macro'):>9.3f}")
mode = max(res, key=res.get)
print(f"\nchosen on validation F1: {mode}\n")
pt = P["test"][mode].argmax(1)
print(classification_report(y(te), pt, target_names=classes, zero_division=0))
sns.heatmap(confusion_matrix(y(te), pt), annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
plt.xlabel("Predicted"); plt.ylabel("True"); plt.savefig("results/eye_compare_confusion.png", dpi=150)
joblib.dump(best_f, "models/eye_feat.pkl"); json.dump({"mode": mode}, open("models/eye_mode.json", "w"))