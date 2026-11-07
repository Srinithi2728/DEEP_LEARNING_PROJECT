import numpy as np, cv2
from PIL import ImageEnhance, ImageFilter

# ---------------- EYE ----------------
def eye_metrics(img):                         # measured at the size the model sees
    g = np.asarray(img.convert("L").resize((192, 128)), dtype=np.float32)
    return np.array([g.mean(), g.std(), np.sqrt(cv2.Laplacian(g, cv2.CV_32F).var())])

def eye_degrade(img, level):                  # darker and blurrier as level -> 1
    level = float(level)
    if level <= 0: return img
    if level >= 1.5:                          # camera covered / lights off
        return ImageEnhance.Brightness(img).enhance(0.02)
    img = ImageEnhance.Brightness(img).enhance(1 - 0.9 * level)
    return img.filter(ImageFilter.GaussianBlur(float(6 * level)))

def eye_quality(img, ref):                    # ref = 10th percentile of clean metrics
    return float(np.clip(eye_metrics(img) / ref, 0, 1).min())

# ---------------- EMG ----------------
def emg_metrics(w):                           # w: (channels, time), raw envelope in [0, 1]
    return np.mean(np.ptp(w, axis=1) < 1e-6), np.abs(np.diff(w, axis=1)).mean()

def emg_degrade(w, level, rng):               # noise first, then dead electrodes
    if level <= 0: return w
    w = w + rng.normal(0, 0.2 * level, w.shape).astype(np.float32)
    w[rng.choice(w.shape[0], int(round(level * 8)), replace=False)] = 0.0
    return w

def emg_quality(w, ref_jitter):               # ref_jitter = 95th percentile of clean jitter
    dead, jit = emg_metrics(w)
    return float((1 - dead) * min(1.0, ref_jitter / (jit + 1e-9)))

# ---------------- EEG ----------------
def eeg_quality(w, ref):                      # ref = median clean channel std
    s = w.std(axis=1)
    return float(np.mean((s > 0.1 * ref) & (s < 5 * ref)))

def eeg_degrade(w, level, rng):               # half the bad channels go flat, half get 10x noise
    if level <= 0: return w
    w = w.copy(); sc = np.median(w.std(axis=1))
    for ch in rng.choice(w.shape[0], int(round(level * 16)), replace=False):
        w[ch] = w[ch].mean() if rng.random() < 0.5 else w[ch] + rng.normal(0, 10 * sc, w.shape[1])
    return w