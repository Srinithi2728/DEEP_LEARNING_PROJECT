import glob, re, numpy as np
from scipy.io import loadmat

WIN, STEP, EX = 30, 5, 2        # 30 samples = 300 ms at 100 Hz, 50 ms stride

X, Y, R, S = [], [], [], []
files = sorted(glob.glob(f"data/emg/S*_A1_E{EX}.mat"))
print("files found:", len(files))

for f in files:
    sid = int(re.search(r"S(\d+)_", f).group(1))
    m = loadmat(f)
    emg = m["emg"].astype(np.float32)
    lab = m["restimulus"].flatten()           # refined gesture labels, 0 = rest
    rep = m["rerepetition"].flatten().copy()  # refined repetition ids, 0 = rest

    # rest has repetition 0, so assign it the repetition it sits next to
    last = rep[rep > 0][0]
    for i in range(len(rep)):
        if rep[i] == 0: rep[i] = last
        else: last = rep[i]

    emg = emg / (emg.max(axis=0, keepdims=True) + 1e-6)   # per-subject, per-channel scaling

    for s in range(0, len(emg) - WIN, STEP):
        seg = lab[s:s + WIN]
        if seg.min() != seg.max():            # skip windows spanning a transition
            continue
        X.append(emg[s:s + WIN].T)            # (10, 30)
        Y.append(seg[0]); R.append(rep[s + WIN - 1]); S.append(sid)

X, Y, R, S = np.array(X), np.array(Y), np.array(R), np.array(S)
classes = np.unique(Y)
Y = np.searchsorted(classes, Y)
print("windows:", X.shape, "| classes:", len(classes))
print("windows per class:", np.bincount(Y))
print("repetitions:", np.unique(R), "| subjects:", len(np.unique(S)))
np.savez_compressed("data/emg/windows.npz", X=X, y=Y, rep=R, subj=S, classes=classes)