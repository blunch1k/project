import torch

def coco_to_target(anns):
    boxes, labels = [], []

    for a in anns:
        x,y,w,h = a['bbox']
        boxes.append([x, y, x+w, y+h])
        labels.append(a['category_id'])

    target = {
        "boxes": torch.tensor(boxes, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.int64),
    }
    return target


class CocoWrapper:
    def __init__(self, dataset, transforms=None):
        self.dataset = dataset
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, anns = self.dataset[idx]
        target = coco_to_target(anns)

        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))