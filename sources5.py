import random, numpy as np, torch
from sources import EyeSource, EMGSource, EEGSource, device, EYE_MAP, EMG_MAP, EEG_MAP
from train_eeg import crop_probs, zs
import quality as Q


class _Mixin:
    def _pick(self, intent):
        """Intents a modality cannot express still give a trial, so quality is always measured."""
        if intent in self.by_cmd: return intent, True
        return random.choice(list(self.by_cmd)), False


class EyeQ(EyeSource, _Mixin):
    def __init__(self):
        super().__init__()
        idx = random.sample(range(len(self.ds.samples)), 300)
        m = np.array([Q.eye_metrics(self.ds.loader(self.ds.samples[i][0])) for i in idx])
        self.ref = np.percentile(m, 10, axis=0)

    def read(self, intent, level=0.0):
        key, ok = self._pick(intent)
        path, _ = self.ds.samples[random.choice(self.by_cmd[key])]
        img = Q.eye_degrade(self.ds.loader(path), level)
        q = Q.eye_quality(img, self.ref)
        with torch.no_grad():
            p = torch.softmax(self.model(self.ds.transform(img).unsqueeze(0).to(device)), 1)[0].cpu().numpy()
        k = int(p.argmax())
        return (EYE_MAP[self.classes[k]] if ok else None), float(p[k]), q


class EMGQ(EMGSource, _Mixin):
    def __init__(self):
        super().__init__()
        ck = torch.load("models/emg_model_cmd.pt", map_location=device, weights_only=False)
        self.mean, self.std = ck["mean"][0], ck["std"][0]            # (10, 1)
        self.raw = self.X * self.std + self.mean                      # undo standardization
        self.rng = np.random.default_rng(0)
        self.ref = np.percentile([Q.emg_metrics(w)[1] for w in self.raw[::50]], 95)

    def read(self, intent, level=0.0):
        key, ok = self._pick(intent)
        w = Q.emg_degrade(self.raw[np.random.choice(self.by_cmd[key])], level, self.rng)
        q = Q.emg_quality(w, self.ref)
        x = torch.tensor(((w - self.mean) / self.std).astype(np.float32))[None].to(device)
        with torch.no_grad():
            p = torch.softmax(self.model(x), 1)[0].cpu().numpy()
        k = int(p.argmax())
        return (EMG_MAP.get(k) if ok else None), float(p[k]), q


class EEGQ(EEGSource, _Mixin):
    def __init__(self):
        super().__init__()
        ck = torch.load("models/eeg_model_session.pt", map_location=device, weights_only=False)
        self.raw = np.load("data/eeg/bci2a.npz")["X"][ck["test_idx"]]
        self.rng = np.random.default_rng(0)
        self.ref = float(np.median(self.raw[::10].std(axis=2)))

    def read(self, intent, level=0.0):
        key, ok = self._pick(intent)
        w = Q.eeg_degrade(self.raw[random.choice(self.by_cmd[key])], level, self.rng)
        q = Q.eeg_quality(w, self.ref)
        p = crop_probs(self.model, torch.tensor(zs(w))[None], self.T, device)[0]
        k = int(p.argmax())
        return (EEG_MAP[self.classes[k]] if ok else None), float(p[k]), q