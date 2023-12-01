import glob
import logging
import random
import cv2
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
from utils.augmentations import augment_arc, augment_gaussian, augment_hsv, copy_paste, cutout, random_perspective


def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(
            f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, hyp, img_size: int = 224, mask_suffix: str = '', augment=True):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < img_size, 'img_size must be bigger than zero'
        self.img_size = img_size
        self.mask_suffix = mask_suffix
        self.hyp = hyp
        self.augment = augment

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(
            join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(
                f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')
        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir,
                       mask_suffix=self.mask_suffix), self.ids),
                total=len(self.ids)
            ))

        self.mask_values = list(
            sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, img_size, is_mask):
        newW, newH = img_size, img_size
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize(
            (newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img

    def load_mosaic(self, index):
        # loads images in a 4-mosaic
        s = self.img_size
        yc, xc = [int(random.uniform(-x, 2 * s + x))
                  for x in self.mosaic_border]  # mosaic center x, y
        # 3 additional image indices
        indices = [index] + random.choices(self.indices, k=3)
        random.shuffle(indices)
        for i, index in enumerate(indices):
            # Load image
            img, mask = self.load_images(index)
            h, w = img.shape[:2]

            # place img in img4
            if i == 0:  # top left
                # base image with 4 tiles
                img4 = np.full((s * 2, s * 2, img.shape[2]), 0, dtype=np.uint8)
                # base mask with 4 tiles
                mask4 = np.full((s * 2, s * 2), 0, dtype=np.uint8)
                # xmin, ymin, xmax, ymax (large image)
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                # xmin, ymin, xmax, ymax (small image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(
                    xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            # img4[ymin:ymax, xmin:xmax]
            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            # mask4[ymin:ymax, xmin:xmax]
            mask4[y1a:y2a, x1a:x2a] = mask[y1b:y2b, x1b:x2b]

        # Augment
        if random.random() < self.hyp['copy_paste']:
            img4, mask4 = copy_paste(mask4, img4, mask4, img4)
        img4, mask4 = random_perspective(img4, mask4,
                                         degrees=self.hyp['degrees'],
                                         translate=self.hyp['translate'],
                                         scale=self.hyp['scale'],
                                         shear=self.hyp['shear'],
                                         perspective=self.hyp['perspective'],
                                         border=self.mosaic_border)  # border to remove
        return img4, mask4

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(
            img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(
            mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = load_image(mask_file[0])
        img = load_image(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        if self.augment:
            img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
            mask = np.asarray(mask)
            self.mosaic_border = [-img.shape[0] // 2, -img.shape[0] // 2]
            # Load mosaic
            # if random.random() < self.hyp['mosaic']:
            # img, mask = self.load_mosaic(idx)

            if random.random() < self.hyp['arc']:
                img = augment_arc(img)

            # HSV color-space
            augment_hsv(
                img, hgain=self.hyp['hsv_h'], sgain=self.hyp['hsv_s'], vgain=self.hyp['hsv_v'])

            # Flip up-down
            if random.random() < self.hyp['flipud']:
                img = np.flipud(img).copy()
                mask = np.flipud(mask).copy()

            # Flip left-right
            if random.random() < self.hyp['fliplr']:
                img = np.fliplr(img).copy()
                mask = np.fliplr(mask).copy()

            # GaussianBur
            if random.random() < self.hyp['gaussianblur']:
                img = augment_gaussian(img)

            # Cutouts
            if random.random() < self.hyp['cutouts']:
                img = cutout(img)

            # Augment
            if random.random() < self.hyp['copy_paste']:
                img, mask = copy_paste(mask, img, mask, img, lsj=True)
            img, mask = random_perspective(img, mask,
                                           degrees=self.hyp['degrees'],
                                           translate=self.hyp['translate'],
                                           scale=self.hyp['scale'],
                                           shear=self.hyp['shear'],
                                           perspective=self.hyp['perspective'],
                                           border=self.mosaic_border)  # border to remove

            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            mask = Image.fromarray(mask)

        img = self.preprocess(self.mask_values, img, self.img_size, is_mask=False)
        mask = self.preprocess(self.mask_values, mask,
                               self.img_size, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, hyp, img_size=224, mask_suffix='', augment=True):
        super().__init__(images_dir, mask_dir, hyp, img_size, mask_suffix, augment=True)
