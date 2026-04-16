import torch
from torchvision.datasets import CocoDetection
from torchvision.transforms import ToTensor
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from coco_dataset import collate_fn
import matplotlib.pyplot as plt
from torchvision.datasets import CocoDetection
from torchvision.transforms import ToTensor
from coco_dataset import collate_fn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_ds = CocoDetection(
    root="dataset/train/images",
    annFile="dataset/train/instances_train.json",
    transform=ToTensor()
)
val_ds = CocoDetection(
    root="dataset/val/images",
    annFile="dataset/val/instances_val.json",
    transform=ToTensor()
)

val_dl = torch.utils.data.DataLoader(
    val_ds,
    batch_size=4,
    shuffle=False,
    collate_fn=collate_fn
)

train_dl = torch.utils.data.DataLoader(train_ds, 4, True, collate_fn=collate_fn)

model = fasterrcnn_resnet50_fpn(weights=None)
in_f = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_f, 4)
model.to(DEVICE)

opt = torch.optim.Adam(model.parameters(), 1e-4)

train_losses = []
val_losses = []

best_val = 1e9
patience = 5
wait = 0
for epoch in range(200):
    model.train()
    train_loss = 0

    for imgs, targs in train_dl:
        imgs = [i.to(DEVICE) for i in imgs]
        targs = [{k:v.to(DEVICE) for k,v in t.items()} for t in targs]

        loss = sum(model(imgs, targs).values())

        opt.zero_grad()
        loss.backward()
        opt.step()

        train_loss += loss.item()

    train_loss /= len(train_dl)

    # -------- VALIDATION --------
    model.train()
    val_loss = 0
    with torch.no_grad():
        for imgs, targs in val_dl:
            imgs = [i.to(DEVICE) for i in imgs]
            targs = [{k:v.to(DEVICE) for k,v in t.items()} for t in targs]
            val_loss += sum(model(imgs, targs).values()).item()

    val_loss /= len(val_dl)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1}: Train={train_loss:.3f} Val={val_loss:.3f}")

    # -------- EARLY STOPPING --------
    if val_loss < best_val:
        best_val = val_loss
        wait = 0
        torch.save(model.state_dict(), "ssd_trained.pth")
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping triggered")
            break

plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.savefig("frcnn_loss_curve.png", dpi=300)
plt.show()
torch.save(model.state_dict(), "fasterrcnn_trained.pth")