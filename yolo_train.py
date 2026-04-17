from ultralytics import YOLO

model = YOLO("yolo26s.pt") 

model.train(
    data="project/data_yolo/data.yaml",
    epochs=200,
    imgsz=640,
    batch=16,
    device=0,  
    workers=0,
    patience=5
)