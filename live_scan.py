import os, json, time, collections, numpy as np, cv2, tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from webcam_eye import EyeTracker, open_camera
from phase7_app import App, HIST, ROWS, FLAT
from lm import Predictor

SCAN_T = 1.0        # seconds the highlight stays on each row / key
BLINK_HOLD = 0.7    # seconds of continuous eye closure that count as a selection
REFRACT = 0.8       # pause after a selection
LAPS = 2            # unanswered passes along a row before returning to row scanning


class ScanApp(App):
    def __init__(self, lm, tracker, cap):
        self.state, self.r, self.c, self.laps = "row", 1, 0, 0     # set before App builds the window
        self.t_next, self.paused = time.time() + SCAN_T, False
        self.closed_since, self.fired, self.rearm_at = None, False, 0.0
        self.ear_hist = collections.deque(maxlen=3)
        self.th = json.load(open("models/blink.json"))["ear_closed_below"]
        super().__init__({}, lm)
        self.tracker, self.cap = tracker, cap
        self.root.title("Blink-scanning communication (live webcam)")
        self.info2 = tk.Label(self.root, font=("Consolas", 11)); self.info2.pack()
        self.bar = ttk.Progressbar(self.root, maximum=100, length=420); self.bar.pack(pady=4)
        tk.Label(self.root, font=("Arial", 9),
                 text="Hold a blink to select the highlighted row / key  |  SPACE or Enter = simulated blink  |  p = pause  |  q = quit").pack()
        self.cam_win = tk.Toplevel(self.root); self.cam_win.title("camera")
        self.cam_lbl = tk.Label(self.cam_win); self.cam_lbl.pack()
        self.root.bind("<space>", lambda e: self.on_blink())
        self.root.bind("p", lambda e: setattr(self, "paused", not self.paused))
        self.refresh()
        self.root.after(100, self.tick)

    # ---- scanning logic ----
    def vcols(self, r):
        return [c for c in range(3) if self.sugg[c]] if r == 0 else list(range(len(ROWS[r])))

    def vrows(self):
        return [r for r in range(len(ROWS)) if self.vcols(r)]

    def paint(self):
        for (r, c), lb in self.keys.items():
            if self.state == "row":
                bg = "#ffd54f" if r == self.r else "#f0f0f0"
            else:
                bg = "#ffd54f" if (r, c) == (self.r, self.c) else "#bbdefb" if r == self.r else "#f0f0f0"
            lb.config(bg=bg)

    def refresh(self):
        super().refresh()
        self.paint()

    def scan_step(self, now):
        if self.paused or now < self.t_next:
            return
        self.t_next = now + SCAN_T
        if self.state == "row":
            rows = self.vrows()
            self.r = rows[(rows.index(self.r) + 1) % len(rows)] if self.r in rows else rows[0]
        else:
            cols = self.vcols(self.r)
            if self.c in cols and cols.index(self.c) == len(cols) - 1:
                self.laps += 1
                if self.laps >= LAPS:                      # nobody selected: back to rows
                    self.state, self.r = "row", self.vrows()[0]
                    self.paint(); return
            self.c = cols[(cols.index(self.c) + 1) % len(cols)] if self.c in cols else cols[0]
        self.paint()

    def on_blink(self):
        if self.state == "row":
            if not self.vcols(self.r): return
            self.state, self.c, self.laps = "col", self.vcols(self.r)[0], 0
        else:
            self.pos = FLAT.index((self.r, self.c))
            self.apply("SELECT")                           # types the key / picks the suggestion / speaks
            self.state, self.r = "row", self.vrows()[0]
        self.t_next = time.time() + 1.5 * SCAN_T           # a little extra time to re-orient
        self.paint()

    def intent(self, cmd):                                 # Enter = simulated blink (backup)
        if cmd == "SELECT": self.on_blink()

    # ---- camera loop ----
    def tick(self):
        ok, frame = self.cap.read()
        now = time.time()
        if ok:
            r, prog, view = self.tracker.process(frame), 0.0, frame.copy()
            if r is None:
                self.closed_since, self.fired = None, False
                self.ear_hist.clear()
                msg = "no face detected"
            else:
                self.ear_hist.append(r["ear"])
                ear = float(np.median(self.ear_hist))
                closed = ear < self.th
                if not closed:
                    self.closed_since, self.fired = None, False
                elif now >= self.rearm_at and not self.fired and not self.paused:
                    if self.closed_since is None: self.closed_since = now
                    prog = min(1.0, (now - self.closed_since) / BLINK_HOLD)
                    if prog >= 1.0:
                        self.fired, self.rearm_at = True, now + REFRACT
                        self.on_blink()
                msg = f"EAR {ear:.3f} (closed below {self.th:.3f}) | {'CLOSED' if closed else 'open'}"
                x0, y0, x1, y1 = r["box"]
                cv2.rectangle(view, (x0, y0), (x1, y1), (0, 255, 0), 2)
            small = cv2.resize(view, (480, int(480 * view.shape[0] / view.shape[1])))
            if r is not None:
                small[0:128, 0:192] = cv2.cvtColor(r["crop"], cv2.COLOR_GRAY2BGR)
            img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
            self.cam_lbl.config(image=img); self._img = img
            self.info2.config(text=f"{'PAUSED | ' if self.paused else ''}scan: {self.state} | {msg}")
            self.bar["value"] = prog * 100
        self.scan_step(now)
        self.root.after(5, self.tick)


if __name__ == "__main__":
    assert os.path.exists("models/blink.json"), "run src\\blink_calibrate.py first"
    cap, tracker, lm = open_camera(), EyeTracker(), Predictor()
    if os.path.exists(HIST):
        for line in open(HIST, encoding="utf-8"): lm.learn(line)
    ScanApp(lm, tracker, cap)
    try:
        tk.mainloop()
    finally:
        cap.release()
        