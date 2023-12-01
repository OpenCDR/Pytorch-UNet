import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils.metrics import *

from utils.dice_score import multiclass_dice_coeff, dice_coeff


@torch.inference_mode()
# @torch.no_grad()
def evaluate(net, dataloader, device, amp, auc_roc=False):
    nranks = 1
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    intersect_area_all = torch.zeros([1], dtype=torch.int64).to(device=device)
    pred_area_all = torch.zeros([1], dtype=torch.int64).to(device=device)
    label_area_all = torch.zeros([1], dtype=torch.int64).to(device=device)
    logits_all = None
    label_all = None

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32,
                             memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)

            intersect_area, pred_area, label_area = calculate_area(
                mask_pred,
                mask_true,
                net.n_classes)

            # Gather from all ranks
            if nranks > 1:
                intersect_area_list = []
                pred_area_list = []
                label_area_list = []
                torch.distributed.all_gather(intersect_area_list,
                                             intersect_area)
                torch.distributed.all_gather(pred_area_list, pred_area)
                torch.distributed.all_gather(label_area_list, label_area)

                # Some image has been evaluated and should be eliminated in last iter
                if (iter + 1) * nranks > len(num_val_batches):
                    valid = len(num_val_batches) - iter * nranks
                    intersect_area_list = intersect_area_list[:valid]
                    pred_area_list = pred_area_list[:valid]
                    label_area_list = label_area_list[:valid]

                for i in range(len(intersect_area_list)):
                    intersect_area_all = intersect_area_all + intersect_area_list[
                        i]
                    pred_area_all = pred_area_all + pred_area_list[i]
                    label_area_all = label_area_all + label_area_list[i]
            else:
                intersect_area_all = intersect_area_all + intersect_area
                pred_area_all = pred_area_all + pred_area
                label_area_all = label_area_all + label_area

                if auc_roc:
                    logits = F.softmax(logits, axis=1)
                    if logits_all is None:
                        logits_all = logits.numpy()
                        label_all = mask_true.numpy()
                    else:
                        logits_all = np.concatenate(
                            [logits_all, logits.numpy()])  # (KN, C, H, W)
                        label_all = np.concatenate([label_all, mask_true.numpy()])

            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max(
                ) <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred,
                                         mask_true, reduce_batch_first=False)
            else:
                assert mask_true.min() >= 0 and mask_true.max(
                ) < net.n_classes, 'True mask indices should be in [0, n_classes]'
                # convert to one-hot format
                mask_true = F.one_hot(mask_true, net.n_classes).permute(
                    0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(
                    dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(
                    mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

    metrics_input = (intersect_area_all, pred_area_all, label_area_all)
    class_iou, miou = mean_iou(*metrics_input)
    acc, class_precision, class_recall = class_measurement(
        *metrics_input)
    kappa_ = kappa(*metrics_input)
    class_dice, mdice = dice(*metrics_input)

    if auc_roc:
        auc_roc = metrics.auc_roc(
            logits_all, label_all, num_classes=net.n_classes)
        auc_infor = ' Auc_roc: {:.4f}'.format(auc_roc)

    net.train()
    return dice_score / max(num_val_batches, 1), miou, acc, kappa_
