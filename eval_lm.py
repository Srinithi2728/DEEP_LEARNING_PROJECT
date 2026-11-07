from lm import Predictor, tokenize

lines = [l.strip() for l in open("data/personal.txt", encoding="utf-8") if tokenize(l)]
cut = int(0.6 * len(lines)); hist, test = lines[:cut], lines[cut:]
print(f"personal lines: {len(lines)} | history {len(hist)} | test {len(test)}")
pr = Predictor()


def ksr(sents, k):
    base = used = 0
    for s in sents:
        ctx = []
        for w in tokenize(s):
            base += len(w) + 1
            cost = len(w) + 1
            for j in range(len(w) + 1):
                if w in pr.suggest(ctx, w[:j], k):
                    cost = j + 1; break
            used += cost; ctx.append(w)
        pr.learn(s)                                   # learn only after predicting
    return 1 - used / base


print(f"\n{'lam_max':>8}   k=1     k=3     k=5")
for lam in (0.0, 0.3, 0.6, 0.9):
    row = []
    for k in (1, 3, 5):
        pr.reset(lam)
        for s in hist: pr.learn(s)
        row.append(ksr(test, k))
    print(f"{lam:>8.1f}" + "".join(f"{v:>8.1%}" for v in row))
print("(lam_max 0.0 = base model only, no personalization)")