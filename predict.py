import argparse
import logging
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from unet.Atten_Unet import AttU_Net
from unet.Atten_Unet_puls import AttU_Net_plus
from utils.data_loading import BasicDataset
from unet import UNet
from utils.utils import plot_img_and_mask


def predict_img(net,
                full_img,
                device,
                img_size=320,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(
        None, full_img, img_size=img_size, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(
            output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()


def get_args():
    parser = argparse.ArgumentParser(
        description='Predict masks from input images')
    parser.add_argument('--model', '-m',
                        default='/home/lisen/Desktop/Pytorch-UNet/checkpoints_original/best_core_0.86204.pth',
                        metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--imgs_root', type=str, default='unet', help='test images root')
    parser.add_argument('--save_dir', type=str, default='./results', help='save result dir')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--type', type=str, default='atten_UNet+')
    parser.add_argument('--no-save', '-n', action='store_true',
                        default=False, help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--img_size', '-s', type=int,
                        default=320, help='Image size')
    parser.add_argument('--bilinear', action='store_true',
                        default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int,
                        default=1, help='Number of classes')

    return parser.parse_args()


def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1],
                        len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(message)s')

    imgs_root = args.imgs_root
    imgs = os.listdir(imgs_root)
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    if args.type == 'U-Net':
        model = UNet(n_channels=3, n_classes=args.classes,
                     bilinear=args.bilinear)
        model = model.to(memory_format=torch.channels_last)
        logging.info(f'Network:\n'
                     f'\t{model.n_channels} input channels\n'
                     f'\t{model.n_classes} output channels (classes)\n'
                     f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')
    elif args.type == 'atten_UNet':
        model = AttU_Net(n_channels=3, n_classes=args.classes)
        logging.info(f'Network:\n'
                     f'\t{model.n_channels} input channels\n'
                     f'\t{model.n_classes} output channels (classes)')
    elif args.type == 'atten_UNet+':
        model = AttU_Net_plus(n_channels=3, n_classes=args.classes)
        logging.info(f'Network:\n'
                     f'\t{model.n_channels} input channels\n'
                     f'\t{model.n_classes} output channels (classes)')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model {args.model}')
    logging.info(f'Using device {device}')

    model.to(device=device)
    state_dict = torch.load(args.model, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    model.load_state_dict(state_dict)

    logging.info('Model loaded!')

    t = time.perf_counter()

    for img in imgs:
        filename = os.path.join(imgs_root, img)
        logging.info(f'Predicting image {filename} ...')
        img = Image.open(filename)

        mask = predict_img(net=model,
                           full_img=img,
                           img_size=args.img_size,
                           out_threshold=args.mask_threshold,
                           device=device)

        if not args.no_save:
            out_filename = os.path.join(save_dir, img)
            result = mask_to_image(mask, mask_values)
            result.save(out_filename)
            logging.info(f'Mask saved to {out_filename}')

        if args.viz:
            logging.info(
                f'Visualizing results for image {filename}, close to continue...')
            plot_img_and_mask(img, mask)

    logging.info(f'coast per image:{(time.perf_counter() - t) / len(imgs):.8f}s')
