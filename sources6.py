import json, os, random, numpy as np, torch
import quality as Q
from sources5 import EyeQ, EMGQ, EEGQ
from sources import device, EYE_MAP, EMG_MAP, EEG_MAP
from train_eeg import zs

TEMP = json.load(open("models/temps.json")) if os.path.exists("models/temps.json") \
    else {"eye": 1.0, "emg": 1.0, "eeg": 1.0}


def crop_logits(model, X, T, dev):
    """Logits averaged over 3 crops (start, middle, end) of each trial."""
    model.eval(); L = X.shape[2]; offs = sorted({0, (L - T) // 2, L - T}); out = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            xb = X[i:i + 256].to(dev)
            out.append((sum(model(xb[:, :, o:o + T]) for o in offs) / len(offs)).cpu())
    return torch.cat(out)


class EyeC(EyeQ):
    def read(self, intent, level=0.0):
        key, ok = self._pick(intent)
        path, _ = self.ds.samples[random.choice(self.by_cmd[key])]
        img = Q.eye_degrade(self.ds.loader(path), level)
        q = Q.eye_quality(img, self.ref)
        with torch.no_grad():
            z = self.model(self.ds.transform(img).unsqueeze(0).to(device))[0].cpu()
        p = torch.softmax(z / TEMP["eye"], 0).numpy(); k = int(p.argmax())
        return (EYE_MAP[self.classes[k]] if ok else None), float(p[k]), q


class EMGC(EMGQ):
    def read(self, intent, level=0.0):
        key, ok = self._pick(intent)
        w = Q.emg_degrade(self.raw[np.random.choice(self.by_cmd[key])], level, self.rng)
        q = Q.emg_quality(w, self.ref)
        x = torch.tensor(((w - self.mean) / self.std).astype(np.float32))[None].to(device)
        with torch.no_grad():
            z = self.model(x)[0].cpu()
        p = torch.softmax(z / TEMP["emg"], 0).numpy(); k = int(p.argmax())
        return (EMG_MAP.get(k) if ok else None), float(p[k]), q


class EEGC(EEGQ):
    def read(self, intent, level=0.0):
        key, ok = self._pick(intent)
        w = Q.eeg_degrade(self.raw[random.choice(self.by_cmd[key])], level, self.rng)
        q = Q.eeg_quality(w, self.ref)
        z = crop_logits(self.model, torch.tensor(zs(w))[None], self.T, device)[0]
        p = torch.softmax(z / TEMP["eeg"], 0).numpy(); k = int(p.argmax())
        return (EEG_MAP[self.classes[k]] if ok else None), float(p[k]), q