import os
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import ToTensor
from PIL import Image
import cv2
import numpy as np
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

INPUT_FOLDER = "data/test/images"
OUTPUT_FOLDER = "results_frcnn"
MODEL_PATH = "fasterrcnn_trained.pth"
CONF = 0.7

CLASS_NAMES = ["background","Airplane","Birds","Drones"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

model = fasterrcnn_resnet50_fpn(weights=None)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 4)

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE).eval()

files = [f for f in os.listdir(INPUT_FOLDER)
         if f.lower().endswith((".jpg",".png",".jpeg"))]

for name in files:
    img = Image.open(os.path.join(INPUT_FOLDER, name)).convert("RGB")
    tensor = ToTensor()(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model(tensor)[0]

    np_img = np.array(img)
    count = 0

    for box, lbl, scr in zip(out["boxes"], out["labels"], out["scores"]):
        if scr < CONF or lbl.item()==0:
            continue
        x1,y1,x2,y2 = map(int, box.tolist())
        cv2.rectangle(np_img,(x1,y1),(x2,y2),(0,255,0),2)
        text = f"{CLASS_NAMES[lbl]} {scr:.2f}"
        cv2.putText(np_img,text,(x1,y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
        count += 1

    cv2.imwrite(os.path.join(OUTPUT_FOLDER,name),
                cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR))
    print(name, "objects:", count)