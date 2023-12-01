
"""
Image augmentation functions
"""

import math
import random
import cv2
import numpy as np
from skimage import img_as_float, img_as_ubyte, io


def hist_equalize(im, clahe=True, bgr=False):
    # Equalize histogram on BGR image 'im' with im.shape(n,m,3) and range 0-255
    yuv = cv2.cvtColor(im, cv2.COLOR_BGR2YUV if bgr else cv2.COLOR_RGB2YUV)
    if clahe:
        c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        yuv[:, :, 0] = c.apply(yuv[:, :, 0])
    else:
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])  # equalize Y channel histogram
    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR if bgr else cv2.COLOR_YUV2RGB)  # convert YUV image to RGB

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

def random_flip_horizontal(mask, img, p=0.8):
    if np.random.random() < p:
        img = img[:, ::-1, :]
        mask = mask[:, ::-1]
    return mask, img

def rescale_src(mask_src, img_src, h, w):
    if len(mask_src.shape) == 3:
        h_src, w_src, c = mask_src.shape
    elif len(mask_src.shape) == 2:
        h_src, w_src = mask_src.shape
    max_reshape_ratio = min(h / h_src, w / w_src)
    rescale_ratio = np.random.uniform(0.2, max_reshape_ratio)

    # reshape src img and mask
    rescale_h, rescale_w = int(h_src * rescale_ratio), int(w_src * rescale_ratio)
    mask_src = cv2.resize(mask_src, (rescale_w, rescale_h),
                          interpolation=cv2.INTER_NEAREST)
    # mask_src = mask_src.resize((rescale_w, rescale_h), Image.NEAREST)
    img_src = cv2.resize(img_src, (rescale_w, rescale_h),
                         interpolation=cv2.INTER_LINEAR)

    # set paste coord
    py = int(np.random.random() * (h - rescale_h))
    px = int(np.random.random() * (w - rescale_w))

    # paste src img and mask to a zeros background
    img_pad = np.zeros((h, w, 3), dtype=np.uint8)
    mask_pad = np.zeros((h, w), dtype=np.uint8)
    img_pad[py:int(py + h_src * rescale_ratio), px:int(px + w_src * rescale_ratio), :] = img_src
    mask_pad[py:int(py + h_src * rescale_ratio), px:int(px + w_src * rescale_ratio)] = mask_src

    return mask_pad, img_pad

def copy_paste(mask_src, img_src, mask_main, img_main, lsj=False):
    # Implement Copy-Paste augmentation https://arxiv.org/abs/2012.07177, labels as nx5 np.array(cls, xyxy)
    mask_src, img_src = random_flip_horizontal(mask_src, img_src)
    mask_main, img_main = random_flip_horizontal(mask_main, img_main)

    # LSJ, Large_Scale_Jittering
    if lsj:
        mask_src, img_src = Large_Scale_Jittering(mask_src, img_src)
        mask_main, img_main = Large_Scale_Jittering(mask_main, img_main)
    else:
        # rescale mask_src/img_src to less than mask_main/img_main's size
        h, w, _ = img_main.shape
        mask_src, img_src = rescale_src(mask_src, img_src, h, w)

    img = img_add(img_src, img_main, mask_src)
    mask = img_add(mask_src, mask_main, mask_src)

    return img, mask

def img_add(img_src, img_main, mask_src):
    if len(img_main.shape) == 3:
        h, w, c = img_main.shape
    elif len(img_main.shape) == 2:
        h, w = img_main.shape
    mask = np.asarray(mask_src, dtype=np.uint8)
    sub_img01 = cv2.add(img_src, np.zeros(np.shape(img_src), dtype=np.uint8), mask=mask)
    mask_02 = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    mask_02 = np.asarray(mask_02, dtype=np.uint8)
    sub_img02 = cv2.add(img_main, np.zeros(np.shape(img_main), dtype=np.uint8),
                        mask=mask_02)
    img_main = img_main - sub_img02 + cv2.resize(sub_img01, (img_main.shape[1], img_main.shape[0]),
                                                 interpolation=cv2.INTER_NEAREST)
    return img_main


def cutout(im):
    # Applies image cutout augmentation https://arxiv.org/abs/1708.04552
    h, w = im.shape[:2]
    scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + \
             [0.0625] * 8 + [0.03125] * 16  # image size fraction
    for s in scales:
        mask_h = random.randint(1, int(h * s))  # create random masks
        mask_w = random.randint(1, int(w * s))

        # box
        xmin = max(0, random.randint(0, w) - mask_w // 2)
        ymin = max(0, random.randint(0, h) - mask_h // 2)
        xmax = min(w, xmin + mask_w)
        ymax = min(h, ymin + mask_h)

        # apply random color mask
        im[ymin:ymax, xmin:xmax] = [
            random.randint(64, 191) for _ in range(3)]
    return im


def augment_hsv(im, hgain=0.5, sgain=0.5, vgain=0.5):
    # HSV color-space augmentation
    if hgain or sgain or vgain:
        r = np.random.uniform(-1, 1, 3) * \
            [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
        dtype = im.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(
            sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=im)  # no return needed


def augment_gaussian(im):
    ksizes = [3, 5, 7, 9]
    ksize = random.sample(ksizes, 1)
    im = cv2.GaussianBlur(im, (ksize[0], ksize[0]), 0)
    return im


def random_perspective(im, mask, degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0,
                       border=(0, 0)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))

    height = im.shape[0] + border[0] * 2  # shape(h,w,c)
    width = im.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -im.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -im.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    # x perspective (about y)
    P[2, 0] = random.uniform(-perspective, perspective)
    # y perspective (about x)
    P[2, 1] = random.uniform(-perspective, perspective)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) *
                       math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) *
                       math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 +
                             translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 +
                             translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
        if perspective:
            im2 = cv2.warpPerspective(im, M, dsize=(
                width, height), borderValue=(0, 0, 0))
            mask2 = cv2.warpPerspective(mask, M, dsize=(
                width, height), borderValue=(0, 0, 0))
        else:  # affine
            im2 = cv2.warpAffine(im, M[:2], dsize=(
                width, height), borderValue=(0, 0, 0))
            mask2 = cv2.warpAffine(mask, M[:2], dsize=(
                width, height), borderValue=(0, 0, 0))

    # # Visualize
    # import matplotlib.pyplot as plt
    # ax = plt.subplots(1, 2, figsize=(12, 6))[1].ravel()
    # ax[0].imshow(im[:, :, ::-1])  # base
    # ax[1].imshow(im2[:, :, ::-1])  # warped
    return im2, mask2

def mixup(im, im2,):
    # Applies MixUp augmentation https://arxiv.org/pdf/1710.09412.pdf
    r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
    im = (im * r + im2 * (1 - r)).astype(np.uint8)
    return im

def Large_Scale_Jittering(mask, img, min_scale=0.1, max_scale=2.0):
    rescale_ratio = np.random.uniform(min_scale, max_scale)
    h, w, _ = img.shape

    # rescale
    h_new, w_new = int(h * rescale_ratio), int(w * rescale_ratio)
    img = cv2.resize(img, (w_new, h_new), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (w_new, h_new), interpolation=cv2.INTER_NEAREST)
    # mask = mask.resize((w_new, h_new), Image.NEAREST)

    # crop or padding
    x, y = int(np.random.uniform(0, abs(w_new - w))), int(np.random.uniform(0, abs(h_new - h)))
    if rescale_ratio <= 1.0:  # padding
        img_pad = np.ones((h, w, 3), dtype=np.uint8) * 168
        mask_pad = np.zeros((h, w), dtype=np.uint8)
        img_pad[y:y + h_new, x:x + w_new, :] = img
        mask_pad[y:y + h_new, x:x + w_new] = mask
        return mask_pad, img_pad
    else:  # crop
        img_crop = img[y:y + h, x:x + w, :]
        mask_crop = mask[y:y + h, x:x + w]
        return mask_crop, img_crop

def augment_arc(img):
    img_new = np.zeros(
        (img.shape[0] * 2, img.shape[1] * 2, img.shape[2]), np.uint8)
    for _ in range(4):
        pt1 = (random.randint(0, img.shape[0] * 2), 0)
        for _ in range(10):
            pt2 = (random.randint(1, img.shape[0] * 2),
                   random.randint(0, img.shape[1] * 2))
            value = random.random() * 255
            cv2.line(img_new, pt1, pt2, (value, value, value), random.randint(1, 4))
    img_new = img_new[int(img_new.shape[0] / 2):img_new.shape[0], :]
    ptLeftTop = (random.randint(
        0, img_new.shape[0] - img.shape[0]), random.randint(0, img_new.shape[1] - img.shape[1]))
    for i in range(img.shape[0]):
        for k in range(img.shape[1]):
            for p in range(img.shape[2]):
                if int(img_new[i + ptLeftTop[0]][k + ptLeftTop[1]][p]) + int(img[i][k][p]) > 255:
                    img[i][k][p] = 255
                else:
                    img[i][k][p] += img_new[i + ptLeftTop[0]][k + ptLeftTop[1]][p]
    return img

def augment_blur(img):
    img = img_as_float(img)
    img_out = img.copy()

    row, col, channel = img.shape
    xx = np.arange(col)
    yy = np.arange(row)

    x_mask = np.matlib.repmat(xx, row, 1)
    y_mask = np.matlib.repmat(yy, col, 1)
    y_mask = np.transpose(y_mask)

    center_y = (row - 1)
    center_x = (col - 1) / 2.0

    R = np.sqrt((x_mask - center_x) ** 2 + (y_mask - center_y) ** 2)

    angle = np.arctan2(y_mask - center_y, x_mask - center_x)

    Num = 35
    arr = np.arange(Num)

    for i in range(row):
        for j in range(col):
            R_arr = R[i, j] - arr
            R_arr[R_arr < 0] = 0

            new_x = R_arr * np.cos(angle[i, j]) + center_x
            new_y = R_arr * np.sin(angle[i, j]) + center_y

            int_x = new_x.astype(int)
            int_y = new_y.astype(int)

            int_x[int_x > col - 1] = col - 1
            int_x[int_x < 0] = 0
            int_y[int_y < 0] = 0
            int_y[int_y > row - 1] = row - 1

            img_out[i, j, 0] = img[int_y, int_x, 0].sum() / Num
            img_out[i, j, 1] = img[int_y, int_x, 1].sum() / Num
            img_out[i, j, 2] = img[int_y, int_x, 2].sum() / Num

    return img_as_ubyte(img_out)

if __name__ == '__main__':
    img = cv2.imread("/home/lisen/Desktop/UNet3plus_pth/data/train/images/0452.jpg")
    mask = cv2.imread("/home/lisen/Desktop/UNet3plus_pth/data/train/masks/0452_matte.png", cv2.IMREAD_GRAYSCALE)
    img1 = cv2.imread("/home/lisen/Desktop/UNet3plus_pth/data/train/images/0082.jpg")
    mask1 = cv2.imread("/home/lisen/Desktop/UNet3plus_pth/data/train/masks/0082_matte.png", cv2.IMREAD_GRAYSCALE)
    _, img_out = copy_paste(mask, img, mask1, img1, False)
    cv2.imshow("dasd", img_out)
    cv2.imwrite("../copy_paste_8.jpg", img_out)
    cv2.waitKey(0)