import torch
from torchvision.models.detection import ssd300_vgg16
from torchvision.transforms import ToTensor
from torchvision.models.detection.ssd import SSDHead
from PIL import Image
import os

DATASET_ROOT = "data"
IMAGES = f"{DATASET_ROOT}/train/images"
LABELS = f"{DATASET_ROOT}/train/labels"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 4  # background + 3 класса

class YOLODataset(torch.utils.data.Dataset):
    def __init__(self, images_dir, labels_dir):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.files = [f for f in os.listdir(images_dir) if f.endswith((".jpg",".png"))]
        self.t = ToTensor()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        img = Image.open(os.path.join(self.images_dir, name)).convert("RGB")
        w, h = img.size

        boxes, labels = [], []
        label_path = os.path.join(self.labels_dir, name.rsplit(".",1)[0] + ".txt")

        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f:
                    cls, x, y, bw, bh = map(float, line.split())
                    x1 = (x - bw/2) * w
                    y1 = (y - bh/2) * h
                    x2 = (x + bw/2) * w
                    y2 = (y + bh/2) * h
                    boxes.append([x1,y1,x2,y2])
                    labels.append(int(cls)+1)

        if not boxes:
            boxes = torch.zeros((0,4),dtype=torch.float32)
            labels = torch.zeros((0,),dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes,dtype=torch.float32)
            labels = torch.tensor(labels,dtype=torch.int64)

        return self.t(img), {"boxes": boxes, "labels": labels}

def collate_fn(b): return tuple(zip(*b))

ds = YOLODataset(IMAGES, LABELS)
dl = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True,
                                 collate_fn=collate_fn, num_workers=0)

model = ssd300_vgg16(weights="DEFAULT")
model.head = SSDHead(
    [512,1024,512,256,256,256],
    [4,6,6,6,4,4],
    num_classes=NUM_CLASSES
)

model.to(DEVICE).train()
opt = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

for epoch in range(30):
    total = 0
    for imgs, targs in dl:
        imgs = [i.to(DEVICE) for i in imgs]
        targs = [{k:v.to(DEVICE) for k,v in t.items()} for t in targs]
        loss = sum(model(imgs,targs).values())
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item()
    print(f"Epoch {epoch+1}: {total/len(dl):.3f}")

torch.save(model.state_dict(),"ssd_trained.pth")