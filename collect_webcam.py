import os, sys, time, glob, random, numpy as np, cv2, tkinter as tk
from webcam_eye import EyeTracker, open_camera

try:
    import winsound
    beep = lambda f: winsound.Beep(f, 150)
except ImportError:
    beep = lambda f: None

CLASSES = ["center", "closed", "down", "left", "right", "up"]
ROUNDS, SETTLE, CAPTURE, KEEP_EVERY, MIN_OPEN_EAR = 5, 1.5, 4.0, 2, 0.10
START = int(sys.argv[1]) if len(sys.argv) > 1 else 1           # first round number
OUT = "data/webcam"

root = tk.Tk(); SW, SH = root.winfo_screenwidth(), root.winfo_screenheight(); root.destroy()
TARGET = {"center": (.5, .5), "left": (.05, .5), "right": (.95, .5), "up": (.5, .06), "down": (.5, .94)}
PROMPT = {"center": "Look at the dot", "left": "Look at the dot (left edge)", "right": "Look at the dot (right edge)",
          "up": "Look at the dot (top edge)", "down": "Look at the dot (bottom edge)",
          "closed": "CLOSE your eyes at the beep, open at the second beep"}


def screen(text, sub="", target=None, crop=None):
    img = np.zeros((SH, SW, 3), np.uint8)
    if target is not None:
        cv2.circle(img, (int(target[0] * SW), int(target[1] * SH)), 24, (0, 255, 255), -1)
    for txt, y, sc in [(text, 0.30, 1.3), (sub, 0.38, 0.8)]:
        (tw, _), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, sc, 2)
        cv2.putText(img, txt, (max(10, (SW - tw) // 2), int(SH * y)), cv2.FONT_HERSHEY_SIMPLEX, sc, (255, 255, 255), 2)
    if crop is not None:                              # the crop the model will actually see
        img[SH - 140:SH - 12, SW - 204:SW - 12] = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    return img


cap, tracker = open_camera(), EyeTracker()
cv2.namedWindow("collect", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("collect", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
for c in CLASSES: os.makedirs(f"{OUT}/{c}", exist_ok=True)


def grab():
    ok, f = cap.read()
    return tracker.process(f) if ok else None


def show(img):
    cv2.imshow("collect", img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27: cap.release(); cv2.destroyAllWindows(); sys.exit()
    return k


for rnd in range(START, START + ROUNDS):
    while True:                                       # wait for SPACE, with a live crop preview
        r = grab()
        k = show(screen(f"Round {rnd}", "Move your head / lighting slightly. SPACE = start, ESC = quit" +
                        ("" if r else "   [NO FACE DETECTED]"), crop=r["crop"] if r else None))
        if k == 32: break
    order = CLASSES[:]; random.shuffle(order)
    for c in order:
        t0 = time.time()
        while time.time() - t0 < SETTLE:
            r = grab(); show(screen(PROMPT[c], "get ready...", TARGET.get(c)))
        beep(1000)
        t0, seen, kept = time.time(), 0, 0
        while time.time() - t0 < CAPTURE:
            r = grab()
            if r is not None:
                seen += 1
                if (c in ("closed", "down") or r["ear"] > MIN_OPEN_EAR) and seen % KEEP_EVERY == 0:
                    n = len(glob.glob(f"{OUT}/{c}/r{rnd}_*.png"))
                    cv2.imwrite(f"{OUT}/{c}/r{rnd}_{n:03d}.png", r["crop"])
                    np.save(f"{OUT}/{c}/r{rnd}_{n:03d}.npy", r["feat"]); kept += 1
            show(screen(PROMPT[c], "HOLD - recording", TARGET.get(c), r["crop"] if r else None))
        beep(500)
        print(f"round {rnd} | {c:7s} | saved {kept}", flush=True)

cap.release(); cv2.destroyAllWindows()
for c in CLASSES: print(c, len(glob.glob(f"{OUT}/{c}/*.png")))