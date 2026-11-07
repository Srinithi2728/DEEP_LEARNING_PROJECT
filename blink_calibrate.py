import time, json, numpy as np, cv2, tkinter as tk
from webcam_eye import EyeTracker, open_camera

try:
    import winsound
    beep = lambda f: winsound.Beep(f, 120)
except ImportError:
    beep = lambda f: None

TARGET = {"center": (.5, .5), "down": (.5, .94), "left": (.05, .5), "right": (.95, .5)}
root = tk.Tk(); SW, SH = root.winfo_screenwidth(), root.winfo_screenheight(); root.destroy()
cap, tracker = open_camera(), EyeTracker()
cv2.namedWindow("blink", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("blink", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


def draw(text, sub="", target=None):
    img = np.zeros((SH, SW, 3), np.uint8)
    if target:
        cv2.circle(img, (int(target[0] * SW), int(target[1] * SH)), 24, (0, 255, 255), -1)
    for txt, y, sc in [(text, .30, 1.1), (sub, .38, .8)]:
        (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, sc, 2)
        cv2.putText(img, txt, (max(10, (SW - tw) // 2), int(SH * y)), cv2.FONT_HERSHEY_SIMPLEX, sc, (255, 255, 255), 2)
    cv2.imshow("blink", img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        cap.release(); cv2.destroyAllWindows(); raise SystemExit
    return k


def grab():
    ok, f = cap.read()
    return tracker.process(f) if ok else None


def record(text, target, seconds, skip, trim_end=0.0):
    t0 = time.time()
    while time.time() - t0 < 1.0:
        grab(); draw(text, "get ready...", target)
    beep(1000)
    vals, t0 = [], time.time()
    while time.time() - t0 < seconds:
        r = grab(); draw(text, "HOLD", target)
        t = time.time() - t0
        if r is not None and skip <= t <= seconds - trim_end:
            vals.append(r["ear"])
    beep(500)
    return np.array(vals)


while draw("Blink calibration", "Keep your head still. SPACE = start, ESC = quit") != 32:
    grab()

open_vals = {c: record("Keep your eyes OPEN, look at the dot", TARGET[c], 2.5, 0.5) for c in TARGET}
closed = np.concatenate([record("CLOSE your eyes at the beep, keep them shut until the 2nd beep",
                                None, 2.2, 0.7, 0.2) for _ in range(4)])
cap.release(); cv2.destroyAllWindows()

q = lambda a: " ".join(f"{np.percentile(a, p):.3f}" for p in (5, 25, 50, 75, 95))
print("EAR percentiles (5 / 25 / 50 / 75 / 95):")
for name, a in list(open_vals.items()) + [("CLOSED", closed)]:
    print(f"  {name:7s} {q(a)}   n={len(a)}")

open_all = np.concatenate(list(open_vals.values()))
med_c, lo_o = float(np.median(closed)), float(np.percentile(open_all, 5))
if len(closed) < 15 or med_c > 0.75 * lo_o:
    print(f"\nNOT SAVED: closed median {med_c:.3f} vs open 5th percentile {lo_o:.3f} do not separate.")
    print("Try: light on your face, camera at eye level, shut your eyes fully and keep them shut until the 2nd beep.")
    raise SystemExit
th = (med_c + lo_o) / 2
print(f"\nthreshold: EAR < {th:.3f}   (closed median {med_c:.3f}, open 5th percentile {lo_o:.3f})")
print(f"closed frames below threshold: {(closed < th).mean():.2f}")
for name, a in open_vals.items():
    print(f"open frames below threshold while looking {name:6s}: {(a < th).mean():.2f}")
json.dump({"ear_closed_below": th, "closed_median": med_c, "open_p5": lo_o}, open("models/blink.json", "w"))
print("saved models/blink.json")