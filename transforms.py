import numpy as np
import torchvision.transforms.functional as F
from torchvision import transforms


class NonZeroCrop(object):
    """Cut out black regions.
    """

    def __call__(self, img):
        arr = np.asarray(img)
        pixels = np.transpose(arr.nonzero())
        if len(arr.shape) > 2:
            pixels = pixels[:, :-1]
        top = pixels.min(axis=0)
        h, w = pixels.max(axis=0) - top
        return F.crop(img, top[0], top[1], h, w)


class ValidationTransform(transforms.Compose):
    def __init__(self):
        super(ValidationTransform, self).__init__(
            [NonZeroCrop(), transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])


class TrainTransform(transforms.Compose):
    def __init__(self):
        super(TrainTransform, self).__init__(
            [NonZeroCrop(), transforms.Resize(256), transforms.RandomCrop(224), transforms.RandomHorizontalFlip(),
             transforms.ToTensor()])
