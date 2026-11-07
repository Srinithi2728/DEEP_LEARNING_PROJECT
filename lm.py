import re, torch, torch.nn as nn
from collections import Counter

device = "cuda" if torch.cuda.is_available() else "cpu"
TOK = re.compile(r"[a-z']+")


def tokenize(s):
    return TOK.findall(s.lower())


SEED = """i need water|i am thirsty|i am hungry|i am in pain|please call the nurse|please call my family
i need help|i feel cold|i feel hot|thank you very much|i am tired|i want to rest
please turn on the light|please turn off the light|i need the bathroom|can you adjust my pillow
my head hurts|i love you|how are you today|good morning|good night|yes please|no thank you
please open the window|i want to watch television|what time is it|i am feeling better today
please stay with me|i need my medicine|please call the doctor|see you tomorrow|i am okay
i am scared|i am happy to see you|please repeat that|wait a moment please|i do not understand
can i have some food|i want to sleep now|please bring me some water|i feel dizzy""".replace("\n", "|").split("|")


def load_corpus():
    import nltk
    nltk.download("brown", quiet=True)
    from nltk.corpus import brown
    out = []
    for s in brown.sents():
        w = [x.lower() for x in s if re.fullmatch(r"[a-z']+", x.lower())]
        if 3 <= len(w) <= 30:
            out.append(w)
    return out


class WordLM(nn.Module):
    def __init__(self, V, E=128, H=256):
        super().__init__()
        self.emb = nn.Embedding(V, E)
        self.rnn = nn.GRU(E, H, batch_first=True)
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(H, V)

    def forward(self, x):
        o, _ = self.rnn(self.drop(self.emb(x)))
        return self.out(self.drop(o))


class Predictor:
    """Base GRU language model + online personal cache."""
    def __init__(self, path="models/lm.pt", lam_max=0.6):
        ck = torch.load(path, map_location=device, weights_only=False)
        self.itos = ck["vocab"]
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.model = WordLM(len(self.itos), ck["E"], ck["H"]).to(device)
        self.model.load_state_dict(ck["state"]); self.model.eval()
        self._cache = {}
        self.reset(lam_max)

    def reset(self, lam_max=None):
        if lam_max is not None:
            self.lam_max = lam_max
        self.uni, self.big, self.big_tot = Counter(), Counter(), Counter()
        self.uni_tot = 0

    def learn(self, text):
        prev = "<s>"
        for w in tokenize(text):
            self.uni[w] += 1; self.big[(prev, w)] += 1; self.big_tot[prev] += 1
            self.uni_tot += 1; prev = w

    def _lm_probs(self, ctx):
        key = tuple(ctx[-8:])
        if key not in self._cache:
            if len(self._cache) > 5000:
                self._cache.clear()
            ids = [1] + [self.stoi.get(w, 2) for w in key]
            with torch.no_grad():
                lo = self.model(torch.tensor([ids], device=device))
            self._cache[key] = torch.softmax(lo[0, -1], 0).cpu().numpy()
        return self._cache[key]

    def suggest(self, ctx, prefix="", k=3):
        p = self._lm_probs(ctx)
        prev = ctx[-1] if ctx else "<s>"
        lam = self.lam_max * self.uni_tot / (self.uni_tot + 100.0)     # more history -> more trust
        cand = {}
        for i in range(3, len(self.itos)):
            w = self.itos[i]
            if w.startswith(prefix):
                cand[w] = (1 - lam) * float(p[i])
        if lam > 0:
            tb = self.big_tot.get(prev, 0)
            wb = tb / (tb + 2.0)                                       # bigram evidence vs unigram
            for w, c in self.uni.items():
                if w.startswith(prefix):
                    pb = self.big.get((prev, w), 0) / tb if tb else 0.0
                    pu = wb * pb + (1 - wb) * c / self.uni_tot
                    cand[w] = cand.get(w, (1 - lam) * 1e-7) + lam * pu  # words the LM has never seen
        return sorted(cand, key=cand.get, reverse=True)[:k]