import glob, random, numpy as np, cv2
random.seed(0)
rows = []
for c in ["center", "up", "down", "closed"]:
    fs = glob.glob(f"data/webcam/{c}/r*_*.png")
    random.shuffle(fs)
    rows.append(np.hstack([cv2.imread(f) for f in fs[:6]]))
cv2.imwrite("results/eye_peek.png", np.vstack(rows))
print("saved results/eye_peek.png  (rows: center, up, down, closed)")