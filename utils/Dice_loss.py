import torch
from torch import nn


class PixelContrastCrossEntropyLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(PixelContrastCrossEntropyLoss, self).__init__()
        self.temperature = temperature

    def forward(self, inputs, targets):
        batch_size, _, height, width = inputs.size()
        print(inputs.shape,targets.shape)

        inputs = inputs.view(batch_size, -1)  # 将输入展平为 [batch_size, num_pixels]
        targets = targets.view(-1)  # 将目标展平为 [batch_size, num_pixels]

        # 计算相似度矩阵
        similarity_matrix = torch.matmul(inputs, inputs.t())  # [batch_size, num_pixels] x [num_pixels, batch_size] = [batch_size, batch_size]

        # 使用温度参数进行缩放
        similarity_matrix /= self.temperature
        
        # 计算对比损失
        contrastive_loss = nn.CrossEntropyLoss()
        
        loss = contrastive_loss(similarity_matrix, targets)

        return loss


class DiceTopK10Loss(nn.Module):
    def __init__(self, topk=10):
        super(DiceTopK10Loss, self).__init__()
        self.topk = topk

    def forward(self, inputs, targets):
        smooth = 1e-5

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = torch.sum(inputs * targets)
        union = torch.sum(inputs) + torch.sum(targets)

        dice_loss = (2.0 * intersection + smooth) / (union + smooth)
        topk_loss = torch.topk(1 - dice_loss, self.topk)[0]

        loss = torch.mean(topk_loss)

        return loss


class OHEMLoss(nn.Module):
    def __init__(self, topk_ratio=0.3):
        super(OHEMLoss, self).__init__()
        self.topk_ratio = topk_ratio

    def forward(self, inputs, targets):
        batch_size = inputs.size(0)
        num_pixels = inputs.size(2) * inputs.size(3)

        inputs = inputs.view(batch_size,-1)
        targets = targets.view(batch_size,-1).float()

        loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)  # 计算交叉熵损失

        k = int(self.topk_ratio * num_pixels)  # 选择前k个像素
        
        loss=torch.flatten(loss)
        
        if k>loss.numel():
            k=loss.numel()

        _, indices = loss.topk(k, largest=True, sorted=False)  # 获取top-k个像素的索引

        loss = loss[indices]  # 从原始损失中选择top-k个像素的损失

        return loss.mean()
