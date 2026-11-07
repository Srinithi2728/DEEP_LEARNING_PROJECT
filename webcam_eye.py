import cv2, numpy as np, mediapipe as mp

RIGHT_EAR = [33, 160, 158, 133, 153, 144]     # eye on the left of the image
LEFT_EAR = [362, 385, 387, 263, 373, 380]     # eye on the right of the image
EYE_A = dict(c1=33, c2=133, up=159, lo=145, iris=468)     # left in the image
EYE_B = dict(c1=362, c2=263, up=386, lo=374, iris=473)    # right in the image
CROP_SCALE = 1.9                              # tighter than before (was 2.6)
OUT_W, OUT_H = 192, 128


def open_camera(index=0, w=1280, h=720):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    if not cap.isOpened():
        raise RuntimeError("camera not found - close other apps using it, or try index=1")
    return cap


def _ear(p, i):
    a = np.linalg.norm(p[i[1]] - p[i[5]]) + np.linalg.norm(p[i[2]] - p[i[4]])
    return a / (2 * np.linalg.norm(p[i[0]] - p[i[3]]) + 1e-6)


def _eye_feats(p, e):
    c1, c2 = p[e["c1"]], p[e["c2"]]
    w = np.linalg.norm(c2 - c1) + 1e-6
    u = (c2 - c1) / w
    n = np.array([-u[1], u[0]])                   # perpendicular to the eye axis (points down)
    iris, up, lo = p[e["iris"]], p[e["up"]], p[e["lo"]]
    h = np.dot(iris - c1, u) / w                  # iris position across the eye
    v = np.dot(iris - (c1 + c2) / 2, n) / w       # iris offset from the eye's centre line
    vr = np.dot(iris - up, n) / (np.dot(lo - up, n) + 1e-6)   # iris position between the lids
    return [h, v, vr, np.dot(lo - up, n) / w], (c1 + c2) / 2


class EyeTracker:
    def __init__(self):
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def process(self, frame_bgr):
        """None if no usable face, else dict(crop, ear, box, brightness, feat, iris)."""
        h, w = frame_bgr.shape[:2]
        res = self.mesh.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return None
        p = np.array([[l.x * w, l.y * h] for l in res.multi_face_landmarks[0].landmark])
        ear = (_ear(p, RIGHT_EAR) + _ear(p, LEFT_EAR)) / 2
        fa, ma = _eye_feats(p, EYE_A); fb, mb = _eye_feats(p, EYE_B)
        d = np.linalg.norm(ma - mb) + 1e-6            # distance between the eyes
        ec = (ma + mb) / 2
        nose = (p[1] - ec) / d                        # head turn / tilt proxy
        roll = np.arctan2(mb[1] - ma[1], mb[0] - ma[0])
        feat = np.nan_to_num(np.array(fa + fb + [nose[0], nose[1], roll, ear], np.float32))
        c1, c2 = p[33], p[133]
        cx, cy = (c1 + c2) / 2
        cw = np.linalg.norm(c1 - c2) * CROP_SCALE
        ch = cw * OUT_H / OUT_W
        x0, y0, x1, y1 = int(cx - cw / 2), int(cy - ch / 2), int(cx + cw / 2), int(cy + ch / 2)
        if x0 < 0 or y0 < 0 or x1 > w or y1 > h or x1 - x0 < 20:
            return None
        gray = cv2.cvtColor(frame_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        crop = cv2.resize(gray, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
        return {"crop": crop, "ear": float(ear), "box": (x0, y0, x1, y1), "brightness": float(gray.mean()),
                "feat": feat, "iris": [p[468], p[473]]}