import numpy as np, warnings
from moabb.datasets import BNCI2014_001
from moabb.paradigms import MotorImagery
warnings.filterwarnings("ignore")

FS = 128
paradigm = MotorImagery(n_classes=4, fmin=4, fmax=38, resample=FS)
X, y, meta = paradigm.get_data(BNCI2014_001(), subjects=list(range(1, 10)))

X = X.astype(np.float32)
classes = np.unique(y).astype(str)
y = np.searchsorted(classes, y.astype(str))
subj = meta["subject"].values
sess = meta["session"].astype(str).values

# the dataset has two recording days; identify the second one as the test day
is_test = np.array([("test" in s.lower()) or s.lower().endswith("_e") for s in sess])
print("session names:", np.unique(sess), "| test trials:", is_test.sum(), "| train trials:", (~is_test).sum())
print("X:", X.shape, "| classes:", classes, "| per class:", np.bincount(y))

np.savez_compressed("data/eeg/bci2a.npz", X=X, y=y, subj=subj, test=is_test, classes=classes)