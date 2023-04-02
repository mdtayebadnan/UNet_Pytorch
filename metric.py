import numpy as np

def dice_score_ignore_bg(y_true, y_pred, num_classes):
    # ignore the background class (class 0)
    class_indices = np.arange(1, num_classes)

    dices = []
    weights = []
    for cls in class_indices:
        true_cls = y_true == cls
        pred_cls = y_pred == cls

        intersection = np.logical_and(true_cls, pred_cls).sum()
        denominator = true_cls.sum() + pred_cls.sum()

        # handle division by zero case
        if denominator == 0:
            dice = 0
        else:
            dice = 2 * intersection / denominator

        dices.append(dice)
        weights.append(np.sum(true_cls))
        
    ##calculate weighted mean
    weights = np.asarray(weights)
    total_pixels = np.sum(weights)
    weights = weights / total_pixels
    weighted_mean_dice = np.sum(np.asarray(dices) * weights)
    
    ##calculate arithmatic mean
    arithmatic_mean_dice = np.mean(dices)

    return weighted_mean_dice,arithmatic_mean_dice

def mean_iou_ignore_bg(y_true, y_pred, num_classes):
    # ignore the background class (class 0)
    class_indices = np.arange(1, num_classes)

    ious = []
    weights = []
    for cls in class_indices:
        true_cls = y_true == cls
        pred_cls = y_pred == cls

        intersection = np.logical_and(true_cls, pred_cls).sum()
        union = np.logical_or(true_cls, pred_cls).sum()

        # handle division by zero case
        if union == 0:
            iou = 0
        else:
            iou = intersection / union

        ious.append(iou)
        weights.append(np.sum(true_cls))
        
    ##calculate the weighted mean
    weights = np.asarray(weights)
    total_pixels = np.sum(weights)
    weights = weights / total_pixels
    weighted_mean_iou = np.sum(np.asarray(ious) * weights)
    
    ##calculate arithmatic mean
    arithmatic_mean_iou = np.mean(ious)

    return weighted_mean_iou,arithmatic_mean_iou
