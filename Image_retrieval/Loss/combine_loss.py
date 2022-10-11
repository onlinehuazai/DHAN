import torch.nn as nn
from Loss.triplet_loss import TripletLoss, CrossEntropyLabelSmooth


# 初始化
tri = TripletLoss(margin=0.5)
cross_entropy = nn.CrossEntropyLoss()
cross_entropy_smooth = CrossEntropyLabelSmooth()

center_weight = 0.0001


def triplet_and_cross(feat, targets, labels):
    loss = tri(feat, labels) + cross_entropy(targets, labels)
    return loss


def triplet_and_cross_smooth(feat, targets, labels):
    loss = tri(feat, labels) + cross_entropy_smooth(targets, labels)
    return loss


def triplet_and_center_and_cross_smooth(feat, targets, labels, center):
    loss = tri(feat, labels) + cross_entropy_smooth(targets, labels) + center_weight * center(feat, labels)
    return loss

