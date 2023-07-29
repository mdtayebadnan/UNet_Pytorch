import torch
import torchvision
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
class FocalLossIgnoreBackground(nn.Module):
    def __init__(self, alpha=1, gamma=2, ignore_index=0):
        super(FocalLossIgnoreBackground, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.loss_fn = nn.BCELoss()

    def forward(self, inputs, targets):
        N, C, H, W = inputs.shape
        one_hot_targets = F.one_hot(targets, num_classes=C).permute(0, 3, 1, 2)
        input_softmax = F.softmax(inputs, dim=1)
        targets_softmax = one_hot_targets * input_softmax + (1 - one_hot_targets) * (1 - input_softmax)
        targets_softmax = targets_softmax.clamp(min=1e-7, max=1 - 1e-7)
        logpt = -torch.log(targets_softmax)
        loss = self.alpha * (1 - targets_softmax)**self.gamma * logpt
        mask = targets != self.ignore_index
        mask = mask.float()
        loss = loss * mask.unsqueeze(1)
        loss = loss.mean()
        return loss
      
class TverskyLoss(nn.Module):
    def __init__(self, alpha: float, beta: float) -> None:
        super(TverskyLoss, self).__init__()
        self.alpha: float = alpha
        self.beta: float = beta
        self.eps: float = 1e-6

    def forward(
            self,
            input: torch.Tensor,
            target: torch.Tensor) -> torch.Tensor:
        # compute softmax over the classes axis
        input_soft = F.softmax(input, dim=1)
        
        target_one_hot = F.one_hot(target, num_classes=input.shape[1]).permute(0,3,1,2)

        # compute the actual dice score
        dims = (1, 2, 3)
        intersection = torch.sum(input_soft * target_one_hot, dims)
        fps = torch.sum(input_soft * (1. - target_one_hot), dims)
        fns = torch.sum((1. - input_soft) * target_one_hot, dims)

        numerator = intersection
        denominator = intersection + self.alpha * fps + self.beta * fns
        tversky_loss = numerator / (denominator + self.eps)  
        return torch.mean(1. - tversky_loss)
