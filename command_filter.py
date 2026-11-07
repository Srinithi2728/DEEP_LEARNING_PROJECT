import numpy as np
from collections import deque

class CommandFilter:
    """Turns noisy per-window probabilities into stable commands."""
    def __init__(self, conf_thresh=0.8, votes=5, need=4, cooldown=10):
        self.conf, self.buf = conf_thresh, deque(maxlen=votes)
        self.need, self.cooldown, self.wait = need, cooldown, 0

    def update(self, probs):
        """probs: softmax vector, index 0 = rest. Returns a class id (>0) or None."""
        k = int(np.argmax(probs))
        self.buf.append(k if probs[k] >= self.conf else 0)   # low confidence counts as rest
        if self.wait > 0:
            self.wait -= 1
            return None
        vals, cnts = np.unique(self.buf, return_counts=True)
        top = vals[cnts.argmax()]
        if top != 0 and cnts.max() >= self.need:
            self.wait = self.cooldown                          # avoid repeated firing
            self.buf.clear()
            return int(top)
        return None