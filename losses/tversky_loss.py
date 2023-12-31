import torch

epsilon = 1e-5
smooth = 1


def tversky(y_true, y_pred):
    y_true_pos = torch.flatten(y_true)
    y_pred_pos = torch.flatten(y_pred)
    true_pos = torch.sum(y_true_pos * y_pred_pos)
    false_neg = torch.sum(y_true_pos * (1-y_pred_pos))
    false_pos = torch.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)


def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)


def focal_tversky(y_true, y_pred):
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return torch.pow((1-pt_1), gamma)


if __name__ == '__main__':
    batch_size = 4
    C = 1
    input = torch.randn(batch_size, C, 320, 320)
    mask = torch.randn(batch_size, 320, 320)
    loss = focal_tversky(input, mask)
    print(loss)
