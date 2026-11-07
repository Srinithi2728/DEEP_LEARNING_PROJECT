import random, numpy as np, torch, torch.nn as nn
from collections import Counter
from lm import WordLM, SEED, load_corpus, tokenize, device

VMAX, EPOCHS, BATCH, LR = 10000, 8, 128, 2e-3
random.seed(0); torch.manual_seed(0)


def main():
    sents = load_corpus(); random.shuffle(sents)
    nval = len(sents) // 50
    val, train = sents[:nval], sents[nval:]
    seed = [tokenize(s) for s in SEED]

    cnt = Counter(w for s in train for w in s)
    itos = ["<pad>", "<s>", "<unk>"] + [w for w, _ in cnt.most_common(VMAX - 3)]
    have = set(itos)
    for s in seed:
        for w in s:
            if w not in have:
                itos.append(w); have.add(w)
    stoi = {w: i for i, w in enumerate(itos)}; V = len(itos)
    enc = lambda s: [stoi.get(w, 2) for w in s]
    data = [enc(s) for s in train] + [enc(s) for s in seed] * 30       # domain phrases upweighted
    vdata = [enc(s) for s in val]
    print(f"vocab {V} | train sentences {len(data)} | val sentences {len(vdata)}")

    def batch(ids):
        L = max(len(s) for s in ids)
        x = torch.zeros(len(ids), L, dtype=torch.long); y = torch.zeros_like(x)
        for i, s in enumerate(ids):
            x[i, :len(s)] = torch.tensor([1] + s[:-1]); y[i, :len(s)] = torch.tensor(s)
        return x.to(device), y.to(device)

    model = WordLM(V).to(device)
    opt = torch.optim.AdamW(model.parameters(), LR, weight_decay=1e-2)
    steps = EPOCHS * ((len(data) + BATCH - 1) // BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=steps)

    def perplexity():
        model.eval(); tot, n = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(vdata), 256):
                x, y = batch(vdata[i:i + 256])
                tot += nn.functional.cross_entropy(model(x).reshape(-1, V), y.reshape(-1),
                                                   ignore_index=0, reduction="sum").item()
                n += (y != 0).sum().item()
        return float(np.exp(tot / n))

    for ep in range(EPOCHS):
        model.train(); random.shuffle(data)
        for i in range(0, len(data), BATCH):
            x, y = batch(data[i:i + BATCH])
            loss = nn.functional.cross_entropy(model(x).reshape(-1, V), y.reshape(-1), ignore_index=0)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
        print(f"epoch {ep+1} | Brown validation perplexity {perplexity():.1f}", flush=True)
    torch.save({"state": model.state_dict(), "vocab": itos, "E": 128, "H": 256}, "models/lm.pt")


if __name__ == "__main__":
    main()