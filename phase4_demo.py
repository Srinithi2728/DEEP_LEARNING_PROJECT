import sys, tkinter as tk
from sources import EyeSource, EMGSource, EEGSource, CMDS

THR = {"eye": 0.6, "emg": 0.6, "eeg": 0.5}      # confidence gate per modality
ROWS = [list("QWERTYUIOP"), list("ASDFGHJKL"), list("ZXCVBNM"), ["SPACE", "DEL"]]
FLAT = [(r, c) for r, row in enumerate(ROWS) for c in range(len(row))]


class App:
    def __init__(self, sources):
        self.src, self.active, self.pos, self.text, self.n, self.ok = sources, "eye", 0, "", 0, 0
        self.root = tk.Tk(); self.root.title("Phase 4 - rule-based switching")
        self.out = tk.Label(self.root, font=("Consolas", 22), anchor="w", width=36); self.out.pack(padx=10, pady=8)
        self.mode = tk.Label(self.root, font=("Arial", 14, "bold")); self.mode.pack()
        frame = tk.Frame(self.root); frame.pack(pady=8)
        self.keys = {}
        for r, row in enumerate(ROWS):
            for c, k in enumerate(row):
                lb = tk.Label(frame, text=k, width=7 if len(k) > 1 else 4, height=2, relief="raised", font=("Arial", 16))
                lb.grid(row=r, column=c, padx=2, pady=2); self.keys[(r, c)] = lb
        self.status = tk.Label(self.root, font=("Arial", 11), fg="#444"); self.status.pack(pady=4)
        tk.Label(self.root, text="Arrows / Enter = intended command   |   1 eye, 2 EMG, 3 EEG = choose modality   |   q = quit",
                 font=("Arial", 9)).pack(pady=4)
        for key, cmd in [("<Left>", "LEFT"), ("<Right>", "RIGHT"), ("<Up>", "UP"), ("<Down>", "DOWN"), ("<Return>", "SELECT")]:
            self.root.bind(key, lambda e, c=cmd: self.intent(c))
        for key, name in [("1", "eye"), ("2", "emg"), ("3", "eeg")]:
            self.root.bind(key, lambda e, n=name: self.switch(n))
        self.root.bind("q", lambda e: self.root.destroy())
        self.draw()

    def switch(self, name): self.active = name; self.draw()

    def intent(self, cmd):
        src = self.src[self.active]
        if cmd not in src.by_cmd:
            self.status.config(text=f"[{self.active}] cannot produce {cmd}"); return
        pred, conf = src.sample(cmd)
        accepted = pred is not None and conf >= THR[self.active]
        self.n += 1; self.ok += int(accepted and pred == cmd)
        self.status.config(text=f"[{self.active}] intended {cmd} -> model says {pred} ({conf:.2f}) -> "
                                f"{'APPLIED' if accepted else 'rejected'} | correct so far: {self.ok}/{self.n}")
        if accepted: self.apply(pred)

    def apply(self, cmd):
        r, c = FLAT[self.pos]
        if cmd == "LEFT": self.pos = (self.pos - 1) % len(FLAT)
        elif cmd == "RIGHT": self.pos = (self.pos + 1) % len(FLAT)
        elif cmd in ("UP", "DOWN"):
            nr = (r + (1 if cmd == "DOWN" else -1)) % len(ROWS)
            self.pos = FLAT.index((nr, min(c, len(ROWS[nr]) - 1)))
        elif cmd == "SELECT":
            k = ROWS[r][c]
            self.text = self.text + " " if k == "SPACE" else self.text[:-1] if k == "DEL" else self.text + k
        self.draw()

    def draw(self):
        self.out.config(text=self.text + "|")
        self.mode.config(text=f"Active modality: {self.active.upper()}")
        for pos, (r, c) in enumerate(FLAT):
            self.keys[(r, c)].config(bg="#ffd54f" if pos == self.pos else "#f0f0f0")


def evaluate(sources, n=100):
    for name, src in sources.items():
        print(f"\n{name.upper()} (confidence gate {THR[name]}, {n} trials per intent)")
        print(f"{'intent':8s}{'correct':>9s}{'wrong':>9s}{'missed':>9s}")
        for intent in CMDS + [None]:
            if intent not in src.by_cmd: continue
            out = []
            for _ in range(n):
                cmd, conf = src.sample(intent)
                out.append(cmd if (cmd is not None and conf >= THR[name]) else None)
            if intent is None:
                print(f"{'rest':8s}{'':>9s}{sum(o is not None for o in out) / n:>9.2f}{'':>9s}   <- false-command rate")
            else:
                ok = sum(o == intent for o in out) / n; miss = sum(o is None for o in out) / n
                print(f"{intent:8s}{ok:>9.2f}{1 - ok - miss:>9.2f}{miss:>9.2f}")


if __name__ == "__main__":
    sources = {"eye": EyeSource(), "emg": EMGSource(), "eeg": EEGSource()}
    if len(sys.argv) > 1 and sys.argv[1] == "eval": evaluate(sources)
    else: tk.mainloop() if App(sources) else None