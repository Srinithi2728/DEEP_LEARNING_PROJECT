import json, joblib
import os, time, collections, numpy as np, cv2, torch, torch.nn as nn, tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from torchvision import models
from webcam_eye import EyeTracker, open_camera
from phase7_app import App, HIST
from lm import Predictor

device = "cuda" if torch.cuda.is_available() else "cpu"
DWELL, REPEAT, BLINK_HOLD, GATE = 0.45, 0.80, 1.0, 0.60      # seconds / confidence gate
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)        # add "up" to try it
BLINK_TH = json.load(open("models/blink.json"))["ear_closed_below"] if os.path.exists("models/blink.json") else None

class EyeNet:
    """Per-session gaze model from session_calibrate.py (replaces the CNN for live use)."""
    def __init__(self):
        d = joblib.load("models/session.pkl")
        self.clf, self.classes, self.enabled = d["clf"], d["classes"], set(d["reliable"])
        print("directions enabled:", sorted(self.enabled), "| within-session accuracy:", round(d["acc"], 3))
        self.enabled = {"left", "right", "down"}

    def probs(self, r):
        return self.clf.predict_proba(np.nan_to_num(r["feat"])[None])[0]


class Controller:
    """Turns a stream of gaze labels + eye-closed flags into discrete commands."""
    def __init__(self): self.reset()

    def reset(self):
        self.cur, self.start, self.last = None, 0.0, None
        self.close_t, self.close_fired, self.quiet_until = None, False, 0.0

    def update(self, label, closed, t):
        if closed:                                          # long blink -> SELECT
            self.cur, self.last = None, None
            if self.close_t is None: self.close_t = t
            prog = min(1.0, (t - self.close_t) / BLINK_HOLD)
            if prog >= 1.0 and not self.close_fired:
                self.close_fired = True
                return "SELECT", 1.0
            return None, (0.0 if self.close_fired else prog)
        if self.close_t is not None:                        # eyes just reopened
            self.close_t, self.close_fired, self.quiet_until = None, False, t + 0.5
        if t < self.quiet_until or label is None:
            self.cur, self.last = None, None
            return None, 0.0
        if label != self.cur:
            self.cur, self.start, self.last = label, t, None
            return None, 0.0
        if self.last is None:                               # first move after a dwell
            prog = (t - self.start) / DWELL
            if prog >= 1.0:
                self.last = t
                return label.upper(), 1.0
            return None, prog
        prog = (t - self.last) / REPEAT                     # then auto-repeat while held
        if prog >= 1.0:
            self.last = t
            return label.upper(), 1.0
        return None, prog


class LiveApp(App):
    def __init__(self, lm, tracker, cap, net):
        super().__init__({}, lm)
        self.tracker, self.cap, self.net = tracker, cap, net
        self.ctl, self.sp, self.ear_hist = Controller(), None, collections.deque(maxlen=300)
        self.mode, self.paused = "both", False
        self.root.title("Eye-controlled communication (live webcam)")
        self.gaze = tk.Label(self.root, font=("Consolas", 11)); self.gaze.pack()
        self.prog = ttk.Progressbar(self.root, maximum=100, length=420); self.prog.pack(pady=4)
        tk.Label(self.root, font=("Arial", 9),
                 text="p = pause | m = blink-detection mode | arrows/Enter still work as a backup").pack()
        self.cam_win = tk.Toplevel(self.root); self.cam_win.title("camera")
        self.cam_lbl = tk.Label(self.cam_win); self.cam_lbl.pack()
        self.root.bind("p", lambda e: setattr(self, "paused", not self.paused))
        self.root.bind("m", self.cycle_mode)
        self.root.attributes("-fullscreen", True)
        self.root.bind("<Escape>", lambda e: self.root.attributes("-fullscreen", False))
        for txt, kw in [("◀", dict(relx=0.01, rely=0.5, anchor="w")), ("▶", dict(relx=0.99, rely=0.5, anchor="e")),
                        ("▲", dict(relx=0.5, rely=0.01, anchor="n")), ("▼", dict(relx=0.5, rely=0.99, anchor="s"))]:
            tk.Label(self.root, text=txt, font=("Arial", 48), fg="#888").place(**kw)
        self.cam_win.geometry("+10+10")
        self.cam_win.attributes("-topmost", True)
        self.root.after(100, self.tick)

    def cycle_mode(self, _=None):
        self.mode = {"both": "cnn", "cnn": "ear", "ear": "both"}[self.mode]

    def intent(self, cmd):                                  # keyboard backup
        self.apply(cmd)

    def tick(self):
        ok, frame = self.cap.read()
        if ok:
            now, r = time.time(), self.tracker.process(frame)
            view, info, prog = frame.copy(), "no face detected - sit in front of the camera", 0.0
            if r is None:
                self.sp = None; self.ctl.reset()
            else:
                p = self.net.probs(r)
                self.sp = p if self.sp is None else 0.5 * self.sp + 0.5 * p     # light smoothing
                k = int(self.sp.argmax()); name, conf = self.net.classes[k], float(self.sp[k])
                self.ear_hist.append(r["ear"])
                recent = float(np.median(list(self.ear_hist)[-3:]))        # 3-frame median against flicker
                if BLINK_TH is not None:
                    closed = recent < BLINK_TH
                else:                                                      # fallback if blink.json is missing
                    base = np.percentile(self.ear_hist, 90) if len(self.ear_hist) >= 30 else None
                    closed = base is not None and recent < 0.65 * base
                label = name if (conf >= GATE and name in self.net.enabled) else None
                if self.paused:
                    self.ctl.reset()
                else:
                    cmd, prog = self.ctl.update(label, closed, now)
                    if cmd:
                        self.apply(cmd)
                        self.status.config(text=f"last command: {cmd}")
                info = (f"{'PAUSED  ' if self.paused else ''}gaze {name} ({conf:.2f}) | "
                        f"EAR {r['ear']:.2f} | blink mode {self.mode}")
                x0, y0, x1, y1 = r["box"]
                cv2.rectangle(view, (x0, y0), (x1, y1), (0, 255, 0), 2)
                for ix, iy in r["iris"]:
                    cv2.circle(view, (int(ix), int(iy)), 3, (0, 0, 255), -1)
            small = cv2.resize(view, (480, int(480 * view.shape[0] / view.shape[1])))
            if r is not None:
                small[0:128, 0:192] = cv2.cvtColor(r["crop"], cv2.COLOR_GRAY2BGR)   # what the model sees
            img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
            self.cam_lbl.config(image=img); self._img = img
            self.gaze.config(text=info); self.prog["value"] = prog * 100
        self.root.after(5, self.tick)


if __name__ == "__main__":
    cap, tracker, net, lm = open_camera(), EyeTracker(), EyeNet(), Predictor()
    if os.path.exists(HIST):
        for line in open(HIST, encoding="utf-8"): lm.learn(line)
    LiveApp(lm, tracker, cap, net)
    try:
        tk.mainloop()
    finally:
        cap.release()