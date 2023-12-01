import argparse
import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm

import wandb
import yaml
from evaluate import evaluate
from unet import UNet
from unet.Atten_Unet import AttU_Net
from unet.Atten_Unet_puls import AttU_Net_plus
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss
from losses.tversky_loss import tversky_loss
from utils.Dice_loss import PixelContrastCrossEntropyLoss, OHEMLoss
from utils.utils import EarlyStopping
from apex import amp

dir_img = Path('./data/imgs/')
dir_mask = Path('./data/masks/')
dir_checkpoint = Path('./checkpoints+tversky_loss/')


def train_model(
        model,
        device,
        hyp,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_size: int = 224,
        amp_: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        mask_suffix='',
        patience=100,
        data_augment=True
):
    if isinstance(hyp, str):
        with open(hyp) as f:
            hyp = yaml.safe_load(f)  # load hyps dict

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_size, amp=amp_)
    )

    logging.info('hyperparameters: ' +
                 ', '.join(f'{k}={v}' for k, v in hyp.items()))

    # 1. Create dataset
    try:
        dataset = CarvanaDataset(
            images_dir=dir_img,
            mask_dir=dir_mask,
            hyp=hyp,
            img_size=img_size, mask_suffix=str(mask_suffix), augment=data_augment)
    except (AssertionError, RuntimeError, IndexError):
        dataset = BasicDataset(images_dir=dir_img,
                               mask_dir=dir_mask,
                               hyp=hyp,
                               img_size=img_size, mask_suffix=str(mask_suffix), augment=data_augment)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size,
                       num_workers=8, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False,
                            drop_last=True, **loader_args)

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images size:  {img_size}
        Mixed Precision: {amp_}
        Use Data Augment: {data_augment}
    ''')

    global_step = 0
    best_val_score = 0
    val_score = 0

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    # optimizer = optim.RMSprop(model.parameters(),
    #                           lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf

    optimizer = optim.RMSprop(
        model.parameters(), lr=learning_rate, weight_decay=1e-8)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O2")

    def lf(x):
        return (((1 + math.cos(x * math.pi / epochs)) / 2)
                ** 1.0) * 0.95 + 0.05  # cosine

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler.last_epoch = global_step
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    # optimizer, 'max', patience=5)  # goal: maximize Dice score
    # grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    stopper = EarlyStopping(patience=patience)

    pixcel_loss = PixelContrastCrossEntropyLoss()
    ohem_loss = OHEMLoss()

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(
                    device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp_):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1),
                                         true_masks.float())
                        # dice_loss_ = dice_loss(F.sigmoid(masks_pred.squeeze(1)),
                        #    true_masks.float(), multiclass=False)
                        # pixcel_loss_ = pixcel_loss(masks_pred,true_masks)
                        # ohem_loss_=ohem_loss(masks_pred,true_masks)
                        # loss = loss + dice_loss_ * 0.8 + pixcel_loss_ * 0.2
                        tversky_loss_ = tversky_loss(masks_pred, true_masks)
                        loss = loss + tversky_loss_

                    else:
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(
                                0, 3, 1, 2).float(),
                            multiclass=True
                        )

                # optimizer.zero_grad(set_to_none=True)
                # grad_scaler.scale(loss).backward()
                # torch.nn.utils.clip_grad_norm_(
                #     model.parameters(), gradient_clipping)
                # grad_scaler.step(optimizer)
                # grad_scaler.update()
                optimizer.zero_grad()
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                optimizer.step()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        # for tag, value in model.named_parameters():
                        # tag = tag.replace('/', '.')
                        # if not (torch.isinf(value) | torch.isnan(value)).any():
                        # histograms['Weights/' +
                        #    tag] = wandb.Histogram(value.data.cpu())
                        # if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                        # histograms['Gradients/' +
                        #    tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score, miou, acc, kappa = evaluate(
                            model, val_loader, device, amp_)
                        # scheduler.step()

                        logging.info(
                            'Validation Dice score: {}'.format(val_score))

                        try:
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation Dice': val_score,
                                'images': wandb.Image(images[0].cpu()),
                                'masks': {
                                    'true': wandb.Image(true_masks[0].float().cpu()*255),
                                    'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()*255),
                                },
                                'step': global_step,
                                'epoch': epoch,
                                **histograms
                            })
                        except:
                            pass

        scheduler.step()

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict['mask_values'] = dataset.mask_values
            if val_score > best_val_score:
                best_val_score = val_score
                torch.save(state_dict, str(dir_checkpoint /
                           'best_core_{:.05}.pth'.format(best_val_score)))
                logging.info(f'Checkpoint {epoch} saved!')
            # torch.save(state_dict, str(dir_checkpoint /
                #    'checkpoint_epoch{}.pth'.format(epoch)))
            # logging.info(f'Checkpoint {epoch} saved!')

        stopper(epoch=epoch, fitness=val_score)
        if stopper.possible_stop:
            break


def get_args():
    parser = argparse.ArgumentParser(
        description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E',
                        type=int, default=150, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size',
                        metavar='B', type=int, default=14, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1.5e-4,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str,
                        default=False, help='Load model from a .pth file')
    parser.add_argument('--img_size', '-s', type=int,
                        default=320, help='Image size')
    parser.add_argument('--type', type=str, default='atten_UNet+')
    parser.add_argument(
        '--hyp', type=str, default='hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--mask-suffix', type=str, default='_matte')
    parser.add_argument('--amp', action='store_true',
                        default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true',
                        default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int,
                        default=1, help='Number of classes')
    parser.add_argument('--patience', type=int, default=100,
                        help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--data_augment', action='store_true',
                        default=True, help='Use data_augment')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
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

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    # try:
    train_model(
        model=model,
        hyp=args.hyp,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=device,
        img_size=args.img_size,
        val_percent=args.val / 100,
        amp_=args.amp,
        mask_suffix=args.mask_suffix,
        patience=args.patience,
        data_augment=args.data_augment
    )
    # except torch.cuda.OutOfMemoryError:
    #     logging.error('Detected OutOfMemoryError! '
    #                   'Enabling checkpointing to reduce memory usage, but this slows down training. '
    #                   'Consider enabling AMP (--amp) for fast and memory efficient training')
    #     torch.cuda.empty_cache()
    #     model.use_checkpointing()
    #     train_model(
    #         model=model,
    #         epochs=args.epochs,
    #         batch_size=args.batch_size,
    #         learning_rate=args.lr,
    #         device=device,
    #         img_size=args.img_size,
    #         val_percent=args.val / 100,
    #         amp=args.amp
    #         mask_suffix=args.mask_suffix,
    #         patience=args.patience
    #     )
