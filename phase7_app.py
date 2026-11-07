import os, threading, tkinter as tk
from lm import Predictor, tokenize
from sources6 import EyeC, EMGC, EEGC
from phase5_switching import apply_scale
from phase6_adaptive import choose

HIST = "data/personal_history.txt"
ROWS = [["", "", ""], list("QWERTYUIOP"), list("ASDFGHJKL"), list("ZXCVBNM"), ["SPACE", "DEL", "SPEAK"]]
FLAT = [(r, c) for r, row in enumerate(ROWS) for c in range(len(row))]


def speak(text):
    def run():
        try:
            import pyttsx3
            e = pyttsx3.init(); e.say(text); e.runAndWait()
        except Exception as ex:
            print("TTS failed:", ex)
    threading.Thread(target=run, daemon=True).start()


class App:
    def __init__(self, sources, lm, tau=0.6):
        self.src, self.lm, self.tau = sources, lm, tau
        self.pos, self.text, self.sugg, self.n, self.ok = FLAT.index((1, 0)), "", ["", "", ""], 0, 0
        self.root = tk.Tk(); self.root.title("Phase 7 - personalized prediction + speech")
        self.out = tk.Label(self.root, font=("Consolas", 20), anchor="nw", justify="left",
                            wraplength=760, width=44, height=2)
        self.out.pack(padx=10, pady=6)
        self.info = tk.Label(self.root, font=("Arial", 11, "bold")); self.info.pack()
        frame = tk.Frame(self.root); frame.pack(pady=6)
        self.keys = {}
        for r, row in enumerate(ROWS):
            f = tk.Frame(frame); f.pack(pady=2)
            for c, k in enumerate(row):
                lb = tk.Label(f, text=k, width=16 if r in (0, 4) else 4, height=2,
                              relief="raised", font=("Arial", 14))
                lb.pack(side="left", padx=2); self.keys[(r, c)] = lb
        bar = tk.Frame(self.root); bar.pack(pady=4)
        self.level = {}
        for m in sources:
            self.level[m] = tk.DoubleVar(value=0.0)
            sc = tk.Scale(bar, from_=0, to=2.0 if m == "eye" else 1.0, resolution=0.05,
                          orient="horizontal", length=170, variable=self.level[m],
                          label=f"degrade {m.upper()}" + (" (1.5+ = covered)" if m == "eye" else ""))
            sc.pack(side="left", padx=6)
            sc.bind("<ButtonRelease-1>", lambda e: self.root.focus_set())   # arrows must not move sliders
        self.status = tk.Label(self.root, font=("Arial", 10), fg="#444"); self.status.pack(pady=4)
        tk.Label(self.root, font=("Arial", 9),
                 text="Arrows / Enter = intended command  |  q = quit  |  SPEAK key reads the text aloud and learns it").pack()
        for key, cmd in [("<Left>", "LEFT"), ("<Right>", "RIGHT"), ("<Up>", "UP"),
                         ("<Down>", "DOWN"), ("<Return>", "SELECT")]:
            self.root.bind(key, lambda e, c=cmd: self.intent(c))
        self.root.bind("q", lambda e: self.root.destroy())
        self.refresh()

    def parts(self):
        prefix = "" if (not self.text or self.text.endswith(" ")) else self.text.split(" ")[-1]
        return tokenize(self.text[:len(self.text) - len(prefix)]), prefix.lower()

    def refresh(self):
        ctx, prefix = self.parts()
        self.sugg = (self.lm.suggest(ctx, prefix, 3) + ["", "", ""])[:3]
        for c, w in enumerate(self.sugg):
            self.keys[(0, c)].config(text=w or "-")
        self.out.config(text=self.text + "|")
        for pos, rc in enumerate(FLAT):
            self.keys[rc].config(bg="#ffd54f" if pos == self.pos else "#f0f0f0")

    def say(self):
        t = self.text.strip()
        if not t: return
        speak(t); self.lm.learn(t)
        with open(HIST, "a", encoding="utf-8") as f: f.write(t.lower() + "\n")
        self.text = ""

    def apply(self, cmd):
        r, c = FLAT[self.pos]
        if cmd == "LEFT": self.pos = (self.pos - 1) % len(FLAT)
        elif cmd == "RIGHT": self.pos = (self.pos + 1) % len(FLAT)
        elif cmd in ("UP", "DOWN"):
            nr = (r + (1 if cmd == "DOWN" else -1)) % len(ROWS)
            nc = round(c * (len(ROWS[nr]) - 1) / max(1, len(ROWS[r]) - 1))
            self.pos = FLAT.index((nr, nc))
        elif cmd == "SELECT":
            k = ROWS[r][c]
            if r == 0:
                if self.sugg[c]:
                    _, prefix = self.parts()
                    self.text = self.text[:len(self.text) - len(prefix)] + self.sugg[c] + " "
            elif k == "SPACE": self.text += " "
            elif k == "DEL": self.text = self.text[:-1]
            elif k == "SPEAK": self.say()
            else: self.text += k.lower()
        self.refresh()

    def intent(self, cmd):
        reads = {m: s.read(cmd, self.level[m].get()) for m, s in self.src.items()}
        m, pred, _ = choose(reads, self.tau)
        self.n += 1; self.ok += int(pred == cmd)
        self.info.config(text=f"modality used: {(m or 'none').upper()}")
        self.status.config(text=f"intended {cmd} -> {pred} -> {'APPLIED' if m else 'rejected'} "
                                f"| correct commands: {self.ok}/{self.n}")
        if m: self.apply(pred)
        else: self.refresh()


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    src = {"eye": EyeC(), "emg": EMGC(), "eeg": EEGC()}
    apply_scale(src)
    lm = Predictor()
    if os.path.exists(HIST):
        for line in open(HIST, encoding="utf-8"): lm.learn(line)
    App(src, lm); tk.mainloop()