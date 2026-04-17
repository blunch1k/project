import random
import torch
import torchvision.transforms.functional as F
from torchvision.transforms import ColorJitter as CJ



class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target
    
class ToTensor:
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target
    
class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            w, _ = image.size
            image = F.hflip(image)
            boxes = target["boxes"]
            boxes[:, [0,2]] = w - boxes[:, [2,0]]
            target["boxes"] = boxes
        return image, target
    


class ColorJitter:
    def __init__(self, *args, **kwargs):
        self.cj = CJ(*args, **kwargs)

    def __call__(self, image, target):
        image = self.cj(image)  # меняем только картинку
        return image, target

class RandomResize:
    def __call__(self, image, target):
        import random
        from torchvision.transforms import functional as F

        size = random.choice([300, 350, 400, 450])
        w, h = image.size
        image = F.resize(image, (size, size))

        scale_x = size / w
        scale_y = size / h

        boxes = target["boxes"]
        boxes[:, [0,2]] *= scale_x
        boxes[:, [1,3]] *= scale_y
        target["boxes"] = boxes

        return image, target
    

class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            w, _ = image.size
            image = F.hflip(image)

            boxes = target["boxes"]
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
            target["boxes"] = boxes

        return image, target


class RandomBrightness:
    def __call__(self, image, target):
        factor = 0.7 + random.random()*0.6
        image = F.adjust_brightness(image, factor)
        return image, target