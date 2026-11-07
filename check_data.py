import os

path = "data/eye"

for c in sorted(os.listdir(path)):
    if os.path.isdir(os.path.join(path, c)):
        print(c, len(os.listdir(os.path.join(path, c))))
        

import os

path = "data/eye"

classes = ["center", "closed", "down", "left", "right", "up"]

for c in classes:
    folder = os.path.join(path, c)

    images = [
        f for f in os.listdir(folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    print(f"{c:8} : {len(images)} images")
    
        
#cd bci_project
#venv\Scripts\activate