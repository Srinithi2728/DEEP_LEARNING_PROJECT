import itertools, numpy as np, torch
from sklearn.metrics import confusion_matrix
from train_emg import EMGNet

K = 5                                   # number of gestures (rest is always included)
device = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load("models/emg_model_cnn.pt", weights_only=False)
d = np.load("data/emg/windows.npz")
X, y, rep = d["X"], d["y"], d["rep"]
nc = len(ck["classes"])

m = rep == 10                           # validation repetition only
Xv = torch.tensor((X[m] - ck["mean"]) / ck["std"]).to(device)
yv = y[m]

model = EMGNet(X.shape[1], nc, ck["kind"]).to(device)
model.load_state_dict(ck["state"]); model.eval()
with torch.no_grad():
    pred = torch.cat([model(Xv[i:i + 1024]).argmax(1) for i in range(0, len(Xv), 1024)]).cpu().numpy()

cm = confusion_matrix(yv, pred, labels=range(nc)).astype(float)

def score(gest):
    sel = [0] + list(gest)              # restricted problem: only these classes exist
    rec = {i: cm[i, i] / cm[i, sel].sum() for i in sel}
    g = [rec[i] for i in gest]
    return np.mean(g), np.min(g), rec[0]

res = sorted(((score(c), c) for c in itertools.combinations(range(1, nc), K)),
             key=lambda t: -t[0][0])
print(f"Top {K}-gesture sets (mean recall, worst gesture, rest recall):")
for (mean_r, min_r, rest_r), c in res[:5]:
    print(f"  gestures {list(c)} | mean {mean_r:.3f} | worst {min_r:.3f} | rest {rest_r:.3f}")