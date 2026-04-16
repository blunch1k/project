from ultralytics import YOLO

# Загружаем предобученную маленькую модель
model = YOLO("yolo26s.pt")  # nano версия — быстро учится даже на CPU

model.train(
    data="data_yolo\data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    device=0,   # если есть видеокарта — поставь 0
    workers=0,
    name="yolo_drone_project",
    patience=5
)