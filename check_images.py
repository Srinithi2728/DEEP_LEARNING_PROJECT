import os
from PIL import Image

path = "data/eye"
classes = ["center", "closed", "down", "left", "right", "up"]

for c in classes:
    folder = os.path.join(path, c)

    files = [
        f for f in os.listdir(folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    # Take the first image
    image_path = os.path.join(folder, files[0])

    try:
        img = Image.open(image_path)

        print(f"\nClass: {c}")
        print(f"Image: {files[0]}")
        print(f"Size: {img.size}")
        print(f"Mode: {img.mode}")

    except Exception as e:
        print(f"{c}: ERROR -> {e}")