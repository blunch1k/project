from graph.compare import run_compare

import torch
from torchvision.models.detection import ssd300_vgg16
from ultralytics import YOLO
from torchvision.models.detection import fasterrcnn_resnet50_fpn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ssd_model = ssd300_vgg16(num_classes=4)  # 3 класса + background
ssd_model.load_state_dict(torch.load("ssd_trained.pth", map_location=device))
ssd_model.to(device)

yolo_model = YOLO("runs/detect/train/weights/best.pt")

rcnn_model = fasterrcnn_resnet50_fpn(num_classes=4)
rcnn_model.load_state_dict(torch.load("fasterrcnn_trained.pth", map_location=device))
rcnn_model.to(device)

run_compare(
    ssd_model,
    rcnn_model,
    yolo_model,
    coco_ann="dataset/test/_annotations.coco.json",
    yolo_labels="data_yolo/test/labels",
    images_dir="data_yolo/test/images",
    device=device
)