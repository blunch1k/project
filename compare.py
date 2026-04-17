import os
import json
import torch
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

IOU_THRESH = 0.5
CONF_THRESH = 0.4

# ---------- IOU ----------
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB-xA) * max(0, yB-yA)
    areaA = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])

    return inter / (areaA + areaB - inter + 1e-6)


# ---------- GT from COCO ----------
def load_coco_gt(ann_file):
    with open(ann_file) as f:
        coco = json.load(f)

    id_to_name = {img["id"]: img["file_name"] for img in coco["images"]}
    gt = {name: [] for name in id_to_name.values()}

    for ann in coco["annotations"]:
        img_name = id_to_name[ann["image_id"]]
        x, y, w, h = ann["bbox"]
        box = [x, y, x+w, y+h]
        gt[img_name].append((ann["category_id"], box))

    return gt


def load_yolo_gt(labels_dir, images_dir):
    gt = {}

    # создаём словарь: имя без .txt -> реальное имя картинки
    image_files = os.listdir(images_dir)
    name_map = {}

    for img in image_files:
        key = os.path.splitext(img)[0]
        name_map[key] = img

    for file in os.listdir(labels_dir):
        if not file.endswith(".txt"):
            continue

        key = os.path.splitext(file)[0]

        if key not in name_map:
            continue

        img_name = name_map[key]
        img_path = os.path.join(images_dir, img_name)

        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        boxes = []
        with open(os.path.join(labels_dir, file)) as f:
            for line in f:
                cls, xc, yc, bw, bh = map(float, line.split())
                x1 = (xc - bw/2) * w
                y1 = (yc - bh/2) * h
                x2 = (xc + bw/2) * w
                y2 = (yc + bh/2) * h
                boxes.append((int(cls)+1, [x1,y1,x2,y2]))

        gt[img_name] = boxes

    return gt


# ---------- METRICS ----------
def evaluate(preds, gts):
    TP, FP, FN = 0, 0, 0

    for img in gts:
        gt_boxes = gts[img]
        pred_boxes = preds.get(img, [])

        matched = set()

        for pc, pb in pred_boxes:
            ok = False
            for i, (gc, gb) in enumerate(gt_boxes):
                if i in matched:
                    continue
                if pc == gc and iou(pb, gb) > IOU_THRESH:
                    TP += 1
                    matched.add(i)
                    ok = True
                    break
            if not ok:
                FP += 1

        FN += len(gt_boxes) - len(matched)

    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return precision, recall, f1


# ---------- mAP@0.5 ----------
def compute_map(preds, gts):
    aps = []

    for img in gts:
        gt_boxes = gts[img]
        pred_boxes = preds.get(img, [])

        for pc, pb in pred_boxes:
            best_iou = 0
            for gc, gb in gt_boxes:
                if pc == gc:
                    best_iou = max(best_iou, iou(pb, gb))
            aps.append(1 if best_iou > IOU_THRESH else 0)

    return np.mean(aps) if aps else 0


# ---------- MODEL PREDICTIONS ----------
def get_preds_torch(model, images_dir, device):
    model.eval()
    preds = {}

    for img_name in tqdm(os.listdir(images_dir)):
        img_path = os.path.join(images_dir, img_name)
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img_rgb/255.).permute(2,0,1).float().to(device)

        with torch.no_grad():
            out = model([tensor])[0]

        boxes = out["boxes"].cpu().numpy()
        labels = out["labels"].cpu().numpy()
        scores = out["scores"].cpu().numpy()

        img_preds = []
        for b,l,s in zip(boxes, labels, scores):
            if s > CONF_THRESH:
                img_preds.append((int(l), b.tolist()))

        preds[img_name] = img_preds

    return preds


# ---------- YOLO PREDICTIONS ----------
def get_preds_yolo(yolo_model, images_dir):
    preds = {}

    for img_name in tqdm(os.listdir(images_dir)):
        img_path = os.path.join(images_dir, img_name)
        results = yolo_model(img_path, verbose=False)[0]

        img_preds = []
        for b, l, s in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
            if s > CONF_THRESH:
                img_preds.append((int(l)+1, b.cpu().numpy().tolist()))

        preds[img_name] = img_preds

    return preds


# ================= MAIN =================

def run_compare(
    ssd_model, rcnn_model, yolo_model,
    coco_ann, yolo_labels,
    images_dir, device
):

    print("Loading GT...")
    coco_gt = load_coco_gt(coco_ann)
    yolo_gt = load_yolo_gt(yolo_labels, images_dir)

    print("Getting predictions...")
    ssd_preds = get_preds_torch(ssd_model, images_dir, device)
    rcnn_preds = get_preds_torch(rcnn_model, images_dir, device)
    yolo_preds = get_preds_yolo(yolo_model, images_dir)

    print("Evaluating...")

    ssd_metrics = evaluate(ssd_preds, coco_gt)
    rcnn_metrics = evaluate(rcnn_preds, coco_gt)
    yolo_metrics = evaluate(yolo_preds, yolo_gt)

    ssd_map = compute_map(ssd_preds, coco_gt)
    rcnn_map = compute_map(rcnn_preds, coco_gt)
    yolo_map = compute_map(yolo_preds, yolo_gt)

    report = f"""
========== MODEL COMPARISON REPORT ==========

Precision / Recall / F1
--------------------------------------------
SSD:
  Precision: {ssd_metrics[0]:.4f}
  Recall:    {ssd_metrics[1]:.4f}
  F1-score:  {ssd_metrics[2]:.4f}

Faster R-CNN:
  Precision: {rcnn_metrics[0]:.4f}
  Recall:    {rcnn_metrics[1]:.4f}
  F1-score:  {rcnn_metrics[2]:.4f}

YOLO:
  Precision: {yolo_metrics[0]:.4f}
  Recall:    {yolo_metrics[1]:.4f}
  F1-score:  {yolo_metrics[2]:.4f}

mAP@0.5
--------------------------------------------
SSD:          {ssd_map:.4f}
Faster R-CNN: {rcnn_map:.4f}
YOLO:         {yolo_map:.4f}
"""

    print(report)

    with open("comparison_report.txt", "w") as f:
        f.write(report)

    # ---------- Graph ----------
    labels = ["SSD", "Faster R-CNN", "YOLO"]
    maps = [ssd_map, rcnn_map, yolo_map]

    plt.figure()
    plt.bar(labels, maps)
    plt.title("mAP@0.5 Comparison")
    plt.savefig("comparison_map.png")
    plt.close()