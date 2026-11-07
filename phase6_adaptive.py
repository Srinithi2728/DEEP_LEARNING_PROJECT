import sys, numpy as np, tkinter as tk
import matplotlib.pyplot as plt
from sources6 import EyeC, EMGC, EEGC
from phase5_switching import QualitySwitcher, AutoApp, MODS, T, BAD, make_trace, apply_scale

SEEDS = [0, 1, 2]


def choose(reads, tau, qmin=0.3):
    """reads: {modality: (command, calibrated confidence, quality)} -> (modality, command, score)."""
    best, best_s = None, 0.0
    for m, (pred, conf, q) in reads.items():
        if pred is None or q < qmin: continue
        s = conf * q
        if s > best_s: best, best_s = m, s
    if best is None or best_s < tau: return None, None, best_s
    return best, reads[best][0], best_s


def run6(trace, policy, tau):
    sw, rec = QualitySwitcher(), []
    for intent, reads in trace:
        if policy == "adaptive":
            m, pred, _ = choose(reads, tau)
        else:
            m = sw.update({k: reads[k][2] for k in MODS}) if policy == "priority" else policy
            pred = None
            if m is not None:
                p, c, _ = reads[m]; pred = p if (p is not None and c >= tau) else None
        rec.append((m, "none" if pred is None else "correct" if pred == intent else "wrong"))
    return rec


def pooled(traces, policy, tau, mask):
    out = np.concatenate([[r[1] for r in run6(tr, policy, tau)] for tr in traces])
    out = out[np.tile(mask, len(traces))]
    return (out == "correct").mean(), (out == "wrong").mean(), (out == "none").mean()


def run_sim(src):
    traces = [make_trace(src, sd) for sd in SEEDS]       # every policy sees identical signals
    fault = np.array([any(a <= t < b for iv in BAD.values() for a, b in iv) for t in range(T)])
    policies = ["eye", "emg", "eeg", "priority", "adaptive"]
    for tau in (0.6, 0.8):
        for label, mask in [("ALL STEPS", np.ones(T, bool)), ("STEPS WITH A FAULT", fault)]:
            print(f"\ntau = {tau} | {label} ({mask.sum()} steps x {len(SEEDS)} runs)")
            print(f"{'policy':10s}{'correct':>9s}{'wrong':>8s}{'none':>8s}")
            for pol in policies:
                c, w, n = pooled(traces, pol, tau, mask)
                print(f"{pol:10s}{c:>9.2f}{w:>8.2f}{n:>8.2f}")
    plt.figure(figsize=(6, 5))
    for pol in policies:
        pts = [pooled(traces, pol, tau, np.ones(T, bool)) for tau in np.linspace(0.05, 0.97, 24)]
        plt.plot([p[1] for p in pts], [p[0] for p in pts], "o-", ms=3, label=pol)
    plt.xlabel("wrong-command rate"); plt.ylabel("correct-command rate")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig("results/phase6_tradeoff.png", dpi=150)
    print("\nsaved results/phase6_tradeoff.png")


class AdaptiveApp(AutoApp):
    def __init__(self, sources, tau=0.6):
        super().__init__(sources); self.tau = tau
        self.root.title("Phase 6 - confidence-based adaptive selection")

    def intent(self, cmd):
        reads = {m: s.read(cmd, self.level[m].get()) for m, s in self.src.items()}
        m, pred, _ = choose(reads, self.tau)
        info = "  ".join(f"{k} {r[1] * r[2]:.2f}" for k, r in reads.items())
        self.active = m or "none"
        self.n += 1; self.ok += int(pred == cmd)
        self.status.config(text=f"conf x quality: {info} | chose {m} -> {pred} -> "
                                f"{'APPLIED' if m else 'rejected'} | correct: {self.ok}/{self.n}")
        if m: self.apply(pred)
        self.draw()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "demo"
    src = {"eye": EyeC(), "emg": EMGC(), "eeg": EEGC()}
    apply_scale(src)
    if mode == "sim": run_sim(src)
    else: AdaptiveApp(src); tk.mainloop()