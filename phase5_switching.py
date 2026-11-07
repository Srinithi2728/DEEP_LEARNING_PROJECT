import sys, random, numpy as np, tkinter as tk
import matplotlib.pyplot as plt
from sources5 import EyeQ, EMGQ, EEGQ
from phase4_demo import App, THR

MODS = ["eye", "emg", "eeg"]
COMMON = ["LEFT", "RIGHT", "SELECT"]          # the three commands every modality can produce


class QualitySwitcher:
    PRIORITY = ["eye", "emg", "eeg"]          # most to least reliable, from the Phase 4 table

    def __init__(self, low=0.6, high=0.7, patience=3):
        self.low, self.high, self.patience = low, high, patience
        self.active = "eye"
        self.bad = dict.fromkeys(self.PRIORITY, 0)
        self.good = dict.fromkeys(self.PRIORITY, 0)

    def update(self, q):
        for m in self.PRIORITY:
            self.bad[m] = self.bad[m] + 1 if q[m] < self.low else 0
            self.good[m] = self.good[m] + 1 if q[m] >= self.high else 0
        ready = [m for m in self.PRIORITY if self.good[m] >= self.patience]
        if self.active is not None and self.bad[self.active] < self.patience:
            better = [m for m in ready if self.PRIORITY.index(m) < self.PRIORITY.index(self.active)]
            if better: self.active = better[0]          # a preferred modality has recovered
        else:
            self.active = ready[0] if ready else None   # None = paused
        return self.active


# ---------- 1. does quality actually track reliability? ----------
def check(src, n=150):
    for m, s in src.items():
        print(f"\n{m.upper()}")
        print(f"{'level':>7s}{'quality':>9s}{'correct':>9s}{'wrong':>8s}{'missed':>8s}")
        for lv in [0, 0.25, 0.5, 0.75, 1.0]:
            qs, out = [], []
            for _ in range(n):
                intent = random.choice(COMMON)
                pred, conf, q = s.read(intent, lv)
                acc = pred is not None and conf >= THR[m]
                qs.append(q); out.append("c" if acc and pred == intent else "w" if acc else "m")
            f = lambda k: out.count(k) / n
            print(f"{lv:>7.2f}{np.mean(qs):>9.2f}{f('c'):>9.2f}{f('w'):>8.2f}{f('m'):>8.2f}")


# ---------- 2. scenario simulation ----------
T = 600
BAD = {"eye": [(150, 350), (520, 560)], "emg": [(300, 450), (520, 560)], "eeg": [(400, 560)]}

def level(m, t):
    if any(a <= t < b for a, b in BAD[m]):
        return 2.0 if m == "eye" else random.uniform(0.85, 1.0)   # eye fault = camera covered
    return 0.0

def make_trace(src, seed=0):
    random.seed(seed); np.random.seed(seed)
    trace = []
    for t in range(T):
        intent = random.choice(COMMON)
        trace.append((intent, {m: src[m].read(intent, level(m, t)) for m in MODS}))
    return trace

def run_policy(trace, policy):
    sw, rec = QualitySwitcher(), []
    for intent, reads in trace:
        act = sw.update({m: reads[m][2] for m in MODS}) if policy == "auto" else policy
        if act is None:
            out = "paused"
        else:
            pred, conf, _ = reads[act]
            ok = pred is not None and conf >= THR[act]
            out = "correct" if ok and pred == intent else "wrong" if ok else "missed"
        rec.append((act, out))
    return rec

def run_sim(src):
    trace = make_trace(src)          # all policies see exactly the same signals and intents
    fault = np.array([any(a <= t < b for iv in BAD.values() for a, b in iv) for t in range(T)])
    for label, mask in [("ALL STEPS", np.ones(T, bool)), ("STEPS WITH A FAULT", fault)]:
        print(f"\n{label} ({mask.sum()} steps)")
        print(f"{'policy':10s}{'correct':>9s}{'wrong':>8s}{'missed':>8s}{'paused':>8s}")
        for pol in ["eye", "emg", "eeg", "auto"]:
            out = np.array([r[1] for r in run_policy(trace, pol)])[mask]
            print(f"{pol:10s}" + "".join(f"{(out == k).mean():>{w}.2f}"
                  for k, w in [("correct", 9), ("wrong", 8), ("missed", 8), ("paused", 8)]))
    rec = run_policy(trace, "auto")
    t = np.arange(T)
    fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    for m in MODS: ax[0].plot(t, [r[1][m][2] for r in trace], label=m, alpha=0.8)
    ax[0].axhline(0.6, ls="--", c="gray"); ax[0].set_ylabel("signal quality"); ax[0].legend(ncol=3)
    ax[1].step(t, [MODS.index(r[0]) if r[0] else -1 for r in rec], where="post")
    ax[1].set_yticks([-1, 0, 1, 2]); ax[1].set_yticklabels(["paused", "eye", "emg", "eeg"])
    ax[1].set_xlabel("time step"); plt.tight_layout()
    plt.savefig("results/phase5_timeline.png", dpi=150)
    print("\nsaved results/phase5_timeline.png")


# ---------- 3. interactive demo: sliders inject faults, switching is automatic ----------
class AutoApp(App):
    def __init__(self, sources):
        super().__init__(sources)
        self.root.title("Phase 5 - quality-based switching")
        self.sw = QualitySwitcher()
        self.level = {m: tk.DoubleVar(value=0.0) for m in sources}
        bar = tk.Frame(self.root); bar.pack(pady=4)
        for m in sources:
            tk.Scale(bar, from_=0, to=1, resolution=0.05, orient="horizontal", length=170,
                     label=f"degrade {m.upper()}", variable=self.level[m]).pack(side="left", padx=6)
        self.draw()

    def switch(self, name): pass                    # manual switching is disabled in auto mode

    def intent(self, cmd):
        reads = {m: s.read(cmd, self.level[m].get()) for m, s in self.src.items()}
        act = self.sw.update({m: r[2] for m, r in reads.items()})
        self.active = act or "paused"
        qtxt = "  ".join(f"{m} {reads[m][2]:.2f}" for m in reads)
        if act is None:
            self.status.config(text=f"quality: {qtxt} | no reliable signal, input paused")
            self.draw(); return
        pred, conf, _ = reads[act]
        ok = pred is not None and conf >= THR[act]
        self.n += 1; self.ok += int(ok and pred == cmd)
        self.status.config(text=f"quality: {qtxt} | [{act}] intended {cmd} -> {pred} ({conf:.2f}) "
                                f"-> {'APPLIED' if ok else 'rejected'} | correct: {self.ok}/{self.n}")
        if ok: self.apply(pred)
        self.draw()

import json, os

def calibrate(src, n=200):
    """Find the raw quality at which each model starts to fail, and rescale quality to that point."""
    scale = {}
    for m, s in src.items():
        rows = []
        for lv in [float(v) for v in np.linspace(0, 1, 11)]:
            qs, out = [], []
            for _ in range(n):
                intent = random.choice(COMMON)
                pred, conf, q = s.read(intent, lv)
                acc = pred is not None and conf >= THR[m]
                qs.append(q); out.append("c" if acc and pred == intent else "w" if acc else "m")
            rows.append((lv, float(np.mean(qs)), out.count("c") / n, out.count("w") / n))
        _, q_star, c0, w0 = rows[0]; lv_ok = 0.0
        for lv, q, c, w in rows:                     # walk up until the model starts to fail
            if c >= 0.9 * c0 and w <= w0 + 0.05: q_star, lv_ok = q, lv
            else: break
        scale[m] = max(q_star, 0.05)
        print(f"{m}: model holds up to degrade level {lv_ok:.1f} (raw quality there {q_star:.2f})")
    json.dump(scale, open("models/quality_scale.json", "w"))

def apply_scale(src):
    if not os.path.exists("models/quality_scale.json"): return
    sc = json.load(open("models/quality_scale.json"))
    for m, s in src.items():
        def read(intent, level=0.0, _raw=s.read, _k=sc[m]):
            p, c, q = _raw(intent, level)
            return p, c, min(1.0, q / _k)
        s.read = read

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "demo"
    src = {"eye": EyeQ(), "emg": EMGQ(), "eeg": EEGQ()}
    if mode == "calibrate":
        calibrate(src); sys.exit()
    apply_scale(src)
    if mode == "check": check(src)
    elif mode == "sim": run_sim(src)
    else:
        AutoApp(src); tk.mainloop()