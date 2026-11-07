import time, json, random, joblib, numpy as np, cv2, tkinter as tk
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from webcam_eye import EyeTracker, open_camera

try:
    import winsound
    beep = lambda f: winsound.Beep(f, 120)
except ImportError:
    beep = lambda f: None

GAZE = ["center", "down", "left", "right", "up"]
TARGET = {"center": (.5, .5), "left": (.05, .5), "right": (.95, .5), "up": (.5, .06), "down": (.5, .94)}
SETTLE, HOLD, PASSES, CLOSE_HOLD = 1.0, 2.0, 2, 2.5

root = tk.Tk(); SW, SH = root.winfo_screenwidth(), root.winfo_screenheight(); root.destroy()
cap, tracker = open_camera(), EyeTracker()
cv2.namedWindow("cal", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("cal", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


def draw(text, sub="", target=None):
    img = np.zeros((SH, SW, 3), np.uint8)
    if target:
        cv2.circle(img, (int(target[0] * SW), int(target[1] * SH)), 24, (0, 255, 255), -1)
    for txt, y, sc in [(text, .30, 1.2), (sub, .38, .8)]:
        (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, sc, 2)
        cv2.putText(img, txt, (max(10, (SW - tw) // 2), int(SH * y)), cv2.FONT_HERSHEY_SIMPLEX, sc, (255, 255, 255), 2)
    cv2.imshow("cal", img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        cap.release(); cv2.destroyAllWindows(); raise SystemExit
    return k


def grab():
    ok, f = cap.read()
    return tracker.process(f) if ok else None


def record(text, target, seconds, drop):
    t0 = time.time()
    while time.time() - t0 < SETTLE:
        grab(); draw(text, "get ready...", target)
    beep(1000)
    feats, total, t0 = [], 0, time.time()
    while time.time() - t0 < seconds:
        r = grab(); draw(text, "HOLD", target)
        if time.time() - t0 >= drop:
            total += 1
            if r is not None: feats.append(r["feat"])
    beep(500)
    return (np.array(feats) if feats else np.zeros((0, 12))), total


while True:
    grab()
    if draw("Session calibration", "Keep your head still, move only your eyes.  SPACE = start, ESC = quit") == 32:
        break

data = {p: {} for p in range(PASSES)}
for p in range(PASSES):
    order = GAZE[:]; random.shuffle(order)
    for c in order:
        f, n = record(f"Pass {p + 1}: look at the dot", TARGET[c], HOLD, 0.4)
        data[p][c] = f
        print(f"pass {p + 1} | {c:7s} | {len(f)} frames with a face out of {n}", flush=True)

closed, closed_total = [], 0
for i in range(2):
    f, n = record("CLOSE your eyes at the beep, open at the 2nd beep", None, CLOSE_HOLD, 0.8)
    closed.append(f); closed_total += n
cap.release(); cv2.destroyAllWindows()

for p in range(PASSES):
    for c in GAZE:
        assert len(data[p][c]) >= 15, f"too few frames for '{c}' (pass {p + 1}): face lost. Improve lighting/position and rerun."


def XY(passes):
    X = np.concatenate([data[p][c] for p in passes for c in GAZE])
    y = np.concatenate([[GAZE.index(c)] * len(data[p][c]) for p in passes for c in GAZE])
    return np.nan_to_num(X), y


mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
P, G = [], []
for tr, te in [([0], [1]), ([1], [0])]:
    X1, y1 = XY(tr); X2, y2 = XY(te)
    P += mk().fit(X1, y1).predict(X2).tolist(); G += y2.tolist()
P, G = np.array(P), np.array(G)
labels = list(range(len(GAZE)))
print(f"\nwithin-session accuracy (train on one pass, test on the other): {(P == G).mean():.3f}")
print(classification_report(G, P, labels=labels, target_names=GAZE, zero_division=0))
print(confusion_matrix(G, P, labels=labels))
pr, rc, _, _ = precision_recall_fscore_support(G, P, labels=labels, zero_division=0)
reliable = [c for i, c in enumerate(GAZE) if c != "center" and pr[i] >= .85 and rc[i] >= .85]
print("directions reliable enough to enable:", reliable)

ear_open = np.concatenate([data[p][c] for p in range(PASSES) for c in GAZE])[:, -1]
ear_closed = np.concatenate([f for f in closed])[:, -1] if sum(len(f) for f in closed) else np.array([])
print(f"\nclosed-eye frames where a face was found: {len(ear_closed)} of {closed_total}")
if len(ear_closed) < 10:
    print("WARNING: the tracker loses your face when your eyes close, so blink detection from landmarks won't work."); raise SystemExit
hi_closed, lo_open = np.percentile(ear_closed, 95), np.percentile(ear_open, 5)
th = float((hi_closed + lo_open) / 2)
print(f"EAR: closed (95th percentile) {hi_closed:.3f} | open incl. looking down (5th percentile) {lo_open:.3f} | threshold {th:.3f}")
if hi_closed >= lo_open:
    print("WARNING: closed and open eyelid values overlap - SELECT may misfire when looking down. Recalibrate with eyes fully shut.")

final = mk().fit(*XY(range(PASSES)))
joblib.dump({"clf": final, "classes": GAZE, "reliable": reliable, "acc": float((P == G).mean())}, "models/session.pkl")
json.dump({"ear_closed_below": th}, open("models/blink.json", "w"))
print("saved models/session.pkl and models/blink.json")