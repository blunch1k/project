import os
import torch
from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection.ssd import SSDHead
from torchvision.transforms import ToTensor
from PIL import Image
import cv2
import numpy as np

# ================= НАСТРОЙКИ =================
INPUT_FOLDER = "data/test/images"     # папка с картинками
OUTPUT_FOLDER = "results"        # куда сохранять
MODEL_PATH = "ssd_trained.pth"
CONF_THRESHOLD = 0.5

CLASS_NAMES = ["background", "Airplane", "Birds", "Drones"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ================= МОДЕЛЬ =================
model = ssd300_vgg16(weights=None)
model.head = SSDHead(
    [512,1024,512,256,256,256],
    [4,6,6,6,4,4],
    num_classes=4
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE).eval()

to_tensor = ToTensor()

# ================= ИНФЕРЕНС ПО ПАПКЕ =================
files = [f for f in os.listdir(INPUT_FOLDER)
         if f.lower().endswith((".jpg", ".png", ".jpeg"))]

print(f"Найдено изображений: {len(files)}")

for name in files:
    path = os.path.join(INPUT_FOLDER, name)
    img = Image.open(path).convert("RGB")
    tensor = to_tensor(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model(tensor)[0]

    np_img = np.array(img)
    detections = 0

    for box, lbl, scr in zip(out["boxes"], out["labels"], out["scores"]):
        if scr < CONF_THRESHOLD or lbl.item() == 0:
            continue

        x1, y1, x2, y2 = map(int, box.tolist())
        cv2.rectangle(np_img, (x1, y1), (x2, y2), (0,255,0), 2)

        text = f"{CLASS_NAMES[lbl]} {scr:.2f}"
        cv2.putText(np_img, text, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        detections += 1

    save_path = os.path.join(OUTPUT_FOLDER, name)
    cv2.imwrite(save_path, cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR))
    print(f"{name} → объектов: {detections}")

print("Готово. Результаты в папке:", OUTPUT_FOLDER)