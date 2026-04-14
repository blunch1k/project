import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import ToTensor

from torchvision.models.detection import ssd300_vgg16, fasterrcnn_resnet50_fpn
from torchvision.models.detection.ssd import SSDHead
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from ultralytics import YOLO

# ====== НАСТРОЙКИ ======
TEST_IMAGES = "data/test/images"
TEST_LABELS = "data/test/labels"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IOU_THRESH = 0.5
CONF = 0.5

# ====== Загрузка моделей ======
def load_ssd():
    m = ssd300_vgg16(weights=None)
    m.head = SSDHead([512,1024,512,256,256,256],[4,6,6,6,4,4],4)
    m.load_state_dict(torch.load("ssd_trained.pth", map_location=DEVICE))
    return m.to(DEVICE).eval()

def load_frcnn():
    m = fasterrcnn_resnet50_fpn(weights=None)
    in_f = m.roi_heads.box_predictor.cls_score.in_features
    m.roi_heads.box_predictor = FastRCNNPredictor(in_f,4)
    m.load_state_dict(torch.load("fasterrcnn_trained.pth", map_location=DEVICE))
    return m.to(DEVICE).eval()


yolo = YOLO("runs/detect/yolo_drone_project/weights/best.pt")
ssd = load_ssd()
frcnn = load_frcnn()
to_tensor = ToTensor()

# ====== IoU ======
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB-xA) * max(0, yB-yA)
    areaA = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    return inter / (areaA + areaB - inter + 1e-6)

def compute_ap(recalls, precisions):
    recalls = np.concatenate(([0.], recalls, [1.]))
    precisions = np.concatenate(([0.], precisions, [0.]))

    for i in range(len(precisions)-1, 0, -1):
        precisions[i-1] = np.maximum(precisions[i-1], precisions[i])

    indices = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[indices+1] - recalls[indices]) * precisions[indices+1])
    return ap

# ====== Чтение YOLO разметки ======
def read_yolo(label_path, w, h):
    boxes = []
    with open(label_path) as f:
        for line in f:
            cls, x, y, bw, bh = map(float, line.split())
            x1 = (x-bw/2)*w
            y1 = (y-bh/2)*h
            x2 = (x+bw/2)*w
            y2 = (y+bh/2)*h
            boxes.append((int(cls)+1, [x1,y1,x2,y2]))
    return boxes

# ====== Оценка модели ======
def yolo_predict(img_path):
    results = yolo(img_path, verbose=False)[0]

    preds = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = box.conf.item()
        cls = int(box.cls.item()) + 1

        if conf > CONF:
            preds.append((cls, [x1, y1, x2, y2], conf))  # ← ВАЖНО

    return preds

def evaluate(model):
    TP=FP=FN=0

    for name in os.listdir(TEST_IMAGES):
        if not name.endswith((".jpg",".png",".jpeg")): continue
        img = Image.open(os.path.join(TEST_IMAGES,name)).convert("RGB")
        w,h = img.size
        tensor = to_tensor(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            out = model(tensor)[0]

        gt = read_yolo(os.path.join(TEST_LABELS,name.rsplit(".",1)[0]+".txt"),w,h)
        used = set()

        for box,lbl,scr in zip(out["boxes"],out["labels"],out["scores"]):
            if scr < CONF or lbl.item()==0: continue
            box = box.tolist()
            matched=False

            for i,(gt_lbl,gt_box) in enumerate(gt):
                if i in used: continue
                if lbl.item()==gt_lbl and iou(box,gt_box)>IOU_THRESH:
                    TP+=1
                    used.add(i)
                    matched=True
                    break
            if not matched:
                FP+=1

        FN += len(gt)-len(used)

    precision = TP/(TP+FP+1e-6)
    recall = TP/(TP+FN+1e-6)
    f1 = 2*precision*recall/(precision+recall+1e-6)

    return precision, recall, f1
def evaluate_yolo():
    TP=FP=FN=0

    for name in os.listdir(TEST_IMAGES):
        if not name.endswith((".jpg",".png",".jpeg")): continue

        img_path = os.path.join(TEST_IMAGES, name)
        img = Image.open(img_path).convert("RGB")
        w,h = img.size

        preds = yolo_predict(img_path)
        gt = read_yolo(os.path.join(TEST_LABELS,
                        name.rsplit(".",1)[0]+".txt"), w, h)

        used = set()

        for lbl, box, conf in preds:
            matched=False
            for i,(gt_lbl,gt_box) in enumerate(gt):
                if i in used: continue
                if lbl==gt_lbl and iou(box,gt_box)>IOU_THRESH:
                    TP+=1
                    used.add(i)
                    matched=True
                    break
            if not matched:
                FP+=1

        FN += len(gt)-len(used)

    precision = TP/(TP+FP+1e-6)
    recall = TP/(TP+FN+1e-6)
    f1 = 2*precision*recall/(precision+recall+1e-6)

    return precision, recall, f1
def evaluate_map(model, is_yolo=False):
    all_detections = {1: [], 2: [], 3: []}
    all_gts = {1: 0, 2: 0, 3: 0}

    for name in os.listdir(TEST_IMAGES):
        if not name.endswith((".jpg",".png",".jpeg")):
            continue

        img_path = os.path.join(TEST_IMAGES, name)
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        gt = read_yolo(os.path.join(
            TEST_LABELS, name.rsplit(".",1)[0]+".txt"), w, h)

        for cls,_ in gt:
            all_gts[cls] += 1

        if is_yolo:
            preds = yolo_predict(img_path)
        else:
            tensor = to_tensor(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                out = model(tensor)[0]
            preds = []
            for b,l,s in zip(out["boxes"],out["labels"],out["scores"]):
                if l.item()!=0:
                    preds.append((l.item(), b.tolist(), s.item()))

        for cls,box,conf in preds:
            all_detections[cls].append((conf, box, gt))

    APs = []

    for cls in [1,2,3]:
        detections = sorted(all_detections[cls], key=lambda x: -x[0])
        TP = np.zeros(len(detections))
        FP = np.zeros(len(detections))

        for i,(conf,box,gt_boxes) in enumerate(detections):
            matched = False
            for gt_cls,gt_box in gt_boxes:
                if gt_cls==cls and iou(box,gt_box)>0.5:
                    matched=True
                    break
            if matched:
                TP[i]=1
            else:
                FP[i]=1

        TP = np.cumsum(TP)
        FP = np.cumsum(FP)

        recalls = TP / (all_gts[cls] + 1e-6)
        precisions = TP / (TP + FP + 1e-6)

        APs.append(compute_ap(recalls, precisions))

    return np.mean(APs)
# ====== Считаем ======
ssd_metrics = evaluate(ssd)
frcnn_metrics = evaluate(frcnn)
yolo_metrics = evaluate_yolo()
ssd_map = evaluate_map(ssd)
frcnn_map = evaluate_map(frcnn)
yolo_map = evaluate_map(None, is_yolo=True)

print("\nmAP@0.5")
print("SSD:", ssd_map)
print("FasterRCNN:", frcnn_map)
print("YOLO:", yolo_map)

print("SSD:", ssd_metrics)
print("FasterRCNN:", frcnn_metrics)
print("YOLO:", yolo_metrics)


# ====== График ======
labels = ["Precision","Recall","F1"]
x = np.arange(len(labels))

plt.figure(figsize=(9,6))
plt.plot(x, ssd_metrics, marker='o')
plt.plot(x, frcnn_metrics, marker='o')
plt.plot(x, yolo_metrics, marker='o')

plt.xticks(x, labels)
plt.grid(True)
plt.legend(["SSD","Faster R-CNN","YOLO"])
plt.title("SSD vs Faster R-CNN vs YOLO")

plt.savefig("model_comparison_metrics.png", dpi=300, bbox_inches="tight")
plt.show()

report = f"""
========== MODEL COMPARISON REPORT ==========

Precision / Recall / F1
--------------------------------------------
SSD:
  Precision: {ssd_metrics[0]:.4f}
  Recall:    {ssd_metrics[1]:.4f}
  F1-score:  {ssd_metrics[2]:.4f}

Faster R-CNN:
  Precision: {frcnn_metrics[0]:.4f}
  Recall:    {frcnn_metrics[1]:.4f}
  F1-score:  {frcnn_metrics[2]:.4f}

YOLO:
  Precision: {yolo_metrics[0]:.4f}
  Recall:    {yolo_metrics[1]:.4f}
  F1-score:  {yolo_metrics[2]:.4f}

mAP@0.5
--------------------------------------------
SSD:         {ssd_map:.4f}
Faster R-CNN:{frcnn_map:.4f}
YOLO:        {yolo_map:.4f}

============================================
"""

print(report)

with open("comparison_report.txt", "w", encoding="utf-8") as f:
    f.write(report)