import glob, os, re, json, numpy as np

rows = []                                              # (ear, class, round)
for f in glob.glob("data/webcam/*/r*_*.npy"):
    rows.append((float(np.load(f)[-1]), os.path.basename(os.path.dirname(f)),
                 int(re.match(r"r(\d+)_", os.path.basename(f)).group(1))))
assert rows, "no .npy feature files found in data/webcam"
test_r = max(r for *_, r in rows)
tr = [x for x in rows if x[2] < test_r]
te = [x for x in rows if x[2] == test_r]


def bal(data, th):                                     # balanced accuracy of "closed" vs "open"
    e = np.array([x[0] for x in data]); closed = np.array([x[1] for x in data]) == "closed"
    pred = e < th
    return 0.5 * (pred[closed].mean() + (~pred[~closed]).mean())


print("EAR per class (all rounds):")
for c in sorted({x[1] for x in rows}):
    v = np.array([x[0] for x in rows if x[1] == c])
    print(f"  {c:7s} median {np.median(v):.3f} | 5th-95th percentile {np.percentile(v, 5):.3f} - {np.percentile(v, 95):.3f}")

th = float(max(np.linspace(0.02, 0.40, 190), key=lambda t: bal(tr, t)))
print(f"\nchosen threshold (best balanced accuracy on rounds before {test_r}): EAR < {th:.3f}")
for lab, t in [("chosen", th), ("stricter (x0.85)", th * 0.85)]:
    print(f"\n{lab}: EAR < {t:.3f} -> share of held-out round {test_r} frames flagged 'closed'")
    for c in sorted({x[1] for x in te}):
        v = np.array([x[0] for x in te if x[1] == c])
        print(f"  {c:7s} {(v < t).mean():.2f}")
json.dump({"ear_closed_below": th}, open("models/blink.json", "w"))
print("\nsaved models/blink.json")