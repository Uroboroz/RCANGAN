from scipy import misc
from scipy import ndimage
from torch.utils.data import Dataset
import numpy as np
import torch
import glob


class Loader(Dataset):
    @staticmethod
    def rescale(image):
        # [n, m, 3] -> [3, n, m]
        return image.transpose((2, 0, 1))

    @staticmethod
    def crop(image, x1, x2, y1, y2):
        if image.shape[1] > image.shape[2]:
            image.transpose((0, 2, 1))
        return image[y1:y2, x1:x2, :]

    @staticmethod
    def flip(image, rotate=0):
        return np.array([ndimage.rotate(i, rotate) for i in image])

    def get_image(self, path_image, x1=0, x2=1080, y1=0, y2=1080, rotate=0):
        image = misc.imread(path_image)
        image = self.crop(image, x1, x2, y1, y2)
        image = self.rescale(image)
        return self.flip(image, rotate)

    def __init__(self, path='./Dataset4K', crop_size=1080):
        self.list_image = list()
        self.crop_size = crop_size
        for file in glob.glob(path + '/*'):
            self.list_image += [{"path_image": file,
                                 "x1": (self.crop_size * x),
                                 "x2": (self.crop_size * (x + 1)),
                                 "y1": (self.crop_size * y),
                                 "y2": (self.crop_size * (y + 1)),
                                 "rotate": j * 90}
                                for x in range(3840 // self.crop_size)
                                for y in range(2160 // self.crop_size)
                                for j in range(4)]

    def __getitem__(self, i):
        return torch.Tensor(self.get_image(**self.list_image[i])), torch.Tensor([0])

    def __len__(self):
        return len(self.list_image)



