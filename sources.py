import random, numpy as np, torch, torch.nn as nn
from torchvision import datasets, transforms, models
from sklearn.model_selection import train_test_split
from train_emg import EMGNet
from train_eeg import EEGNet, crop_probs, zs

device = "cuda" if torch.cuda.is_available() else "cpu"
CMDS = ["LEFT", "RIGHT", "UP", "DOWN", "SELECT"]

# modality output -> command (None = no command). EDIT THESE.
EYE_MAP = {"left": "LEFT", "right": "RIGHT", "up": "UP", "down": "DOWN", "closed": "SELECT", "center": None}
EMG_MAP = {1: "LEFT", 2: "RIGHT", 3: "DOWN", 4: "UP", 5: "SELECT"}      # model class index, 0 = rest
EEG_MAP = {"left_hand": "LEFT", "right_hand": "RIGHT", "feet": "SELECT", "tongue": None}


class EyeSource:
    name = "eye"
    def __init__(self):
        ck = torch.load("models/eye_model.pt", map_location=device, weights_only=False)
        self.classes = ck["classes"]
        m = models.resnet18()
        m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, len(self.classes)))
        m.load_state_dict(ck["state"]); self.model = m.to(device).eval()
        tf = transforms.Compose([transforms.Grayscale(3), transforms.Resize(ck["size"]), transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.ds = datasets.ImageFolder("data/eye", transform=tf)
        labels = np.array(self.ds.targets); idx = np.arange(len(labels))
        _, tmp = train_test_split(idx, test_size=0.3, stratify=labels, random_state=42)   # same split as training
        _, te = train_test_split(tmp, test_size=0.5, stratify=labels[tmp], random_state=42)
        self.by_cmd = {}
        for i in te:
            self.by_cmd.setdefault(EYE_MAP[self.classes[labels[i]]], []).append(int(i))

    def sample(self, intent):
        x, _ = self.ds[random.choice(self.by_cmd[intent])]
        with torch.no_grad():
            p = torch.softmax(self.model(x.unsqueeze(0).to(device)), 1)[0].cpu().numpy()
        k = int(p.argmax()); return EYE_MAP[self.classes[k]], float(p[k])


class EMGSource:
    name = "emg"
    def __init__(self):
        ck = torch.load("models/emg_model_cmd.pt", map_location=device, weights_only=False)
        d = np.load("data/emg/windows.npz")
        keep = list(ck["classes"])
        m = np.isin(d["y"], keep) & np.isin(d["rep"], [2, 5, 7])                       # test repetitions only
        self.X = ((d["X"][m] - ck["mean"]) / ck["std"]).astype(np.float32)
        self.y = np.searchsorted(keep, d["y"][m])
        self.model = EMGNet(self.X.shape[1], len(keep), ck["kind"]).to(device)
        self.model.load_state_dict(ck["state"]); self.model.eval()
        self.by_cmd = {EMG_MAP.get(int(k)): np.where(self.y == k)[0] for k in np.unique(self.y)}

    def sample(self, intent):
        i = np.random.choice(self.by_cmd[intent])
        with torch.no_grad():
            p = torch.softmax(self.model(torch.tensor(self.X[i:i + 1]).to(device)), 1)[0].cpu().numpy()
        k = int(p.argmax()); return EMG_MAP.get(k), float(p[k])


class EEGSource:
    name = "eeg"
    def __init__(self):
        ck = torch.load("models/eeg_model_session.pt", map_location=device, weights_only=False)
        d = np.load("data/eeg/bci2a.npz")
        te = ck["test_idx"]
        self.X, self.y, self.T, self.classes = torch.tensor(zs(d["X"][te])), d["y"][te], ck["T"], ck["classes"]
        self.model = EEGNet(len(self.classes), ck["ch"], self.T).to(device)
        self.model.load_state_dict(ck["state"]); self.model.eval()
        self.by_cmd = {}      # note: intent None = tongue trials (no real EEG "rest" class in this dataset)
        for k, name in enumerate(self.classes):
            self.by_cmd.setdefault(EEG_MAP[name], []).extend(np.where(self.y == k)[0].tolist())

    def sample(self, intent):
        i = random.choice(self.by_cmd[intent])
        p = crop_probs(self.model, self.X[i:i + 1], self.T, device)[0]
        k = int(p.argmax()); return EEG_MAP[self.classes[k]], float(p[k])