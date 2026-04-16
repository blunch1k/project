import torch

def coco_to_target(anns):
    boxes, labels = [], []

    for a in anns:
        x, y, w, h = a['bbox']
        boxes.append([x, y, x+w, y+h])
        labels.append(a['category_id'])

    target = {
        "boxes": torch.tensor(boxes, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.int64),
    }
    return target


def collate_fn(batch):
    imgs, targets = [], []
    for img, anns in batch:
        target = coco_to_target(anns)
        imgs.append(img)
        targets.append(target)
    return imgs, targets