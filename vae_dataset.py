import glob
import os
import cv2
import random
import numbers
import torch
import numpy as np

from torch.utils.data import Dataset

class VAEDataset(Dataset):

    def __init__(self, dir_, mode, transform):
        self.mode = mode

        self.file_list = glob.glob(os.path.join(dir_, mode, '*.jpg'))
        self.dir = dir_
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = self.file_list[idx]

        bgr_img = cv2.imread(img_name, -1)
        try:
            b, g, r = cv2.split(bgr_img)  # get b,g,r
        except:
            print(img_name)
        image = cv2.merge([r, g, b])  # switch it to rgb

        # if bgr_img.ndim == 2:
        #     image = cv2.merge([bgr_img, bgr_img, bgr_img])
        # elif bgr_img.ndim == 3:
        #     if bgr_img.shape[2] == 3:
        #         b, g, r = cv2.split(bgr_img)  # get b,g,r
        #         image = cv2.merge([r, g, b])  # switch it to rgb
        #     else:
        #         image = cv2.merge([bgr_img, bgr_img, bgr_img])
        # image = cv2.imread(img_name, -1)

        sample = self.transform(image)

        return sample

class Scale(object):
    """Rescales the input np.ndarray to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: cv.INTER_CUBIC
    """
    def __init__(self, size, interpolation=cv2.INTER_CUBIC):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        w, h = img.shape[1], img.shape[0]
        if (w <= h and w == self.size) or (h <= w and h == self.size):
            return img
        if w < h:
            ow = self.size
            oh = int(float(self.size) * h / w)
        else:
            oh = self.size
            ow = int(float(self.size) * w / h)
        return cv2.resize(img, dsize=(ow, oh),
                          interpolation=self.interpolation)



class CenterCrop(object):
    """Crops the given np.ndarray at the center to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape
    (size, size)
    """
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        w, h = img.shape[1], img.shape[0]
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))

        return img[y1:y1+th, x1:x1+tw, :]


class RandomScale(object):

    def __init__(self, interpolation=cv2.INTER_CUBIC):
        self.interpolation = interpolation

    def __call__(self, img):
        # random_scale = random.sample([0.25, 0.5, 1.0], 1)
        random_scale = [1.0]
        w, h = img.shape[1], img.shape[0]
        w = int(w * random_scale[0])
        h = int(h * random_scale[0])

        return cv2.resize(img, dsize=(w, h),
                          interpolation=self.interpolation)

class RandomCrop(object):
    """Crops the given np.ndarray at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape
    (size, size)
    """
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        w, h = img.shape[1], img.shape[0]
        th, tw = self.size
        if w == tw and h == th:
            return img

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        if img.ndim == 3:
            img = img[y1:y1+th, x1:x1+tw, :]
        else:
            img = img[y1:y1 + th, x1:x1 + tw]
        return img

class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given np.ndarray with a probability of 0.5
    """
    def __call__(self, img):
        if random.random() < 0.5:
            img = cv2.flip(img, 1).reshape(img.shape)
        return img

class RandomVerticalFlip(object):
    """Randomly vertically flips the given np.ndarray with a probability of 0.5
    """
    def __call__(self, img):
        if random.random() < 0.5:
            img = cv2.flip(img, 0).reshape(img.shape)
        return img

class RandomTransposeFlip(object):
    """Randomly horizontally and vertically flips the given np.ndarray with a probability of 0.5
    """
    def __call__(self, img):
        if random.random() < 0.5:
            img = cv2.flip(img, -1).reshape(img.shape)
        return img


class RandomBlur(object):
    def __call__(self, img):
        label = img
        if random.random() < 0.8:
            # kernel_size = random.randrange(1, 19 + 1, 2)
            kernel_size = 19
            img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

        return {'image': img, 'label': label}


class Convert(object):
    """Randomly horizontally and vertically flips the given np.ndarray with a probability of 0.5
    """
    def __call__(self, img):
        if img.ndim < 3:
            img = np.expand_dims(img, axis=2)
        img = img.transpose(2, 0, 1)

        dtype = torch.FloatTensor
        img = torch.from_numpy(img).type(dtype)/255.0

        return img
