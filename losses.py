import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations
import numpy as np
import time

class HardNegativeLoss(nn.Module):

    def __init__(self):
        super(HardNegativeLoss, self).__init__()

    def forward(self, feats, labels, image_idxs, normalize_feats):

        # Select hard pairs
        #if self.cpu:
            #feats = feats.cpu()

        # Find matrix of all pairwise distances
        # Uncomment below for l2 distance (distance_matrix[i][j] = -2(xi'xj) + xi'xi + xj'xj (which is {xi-xj}'{xi-xj}))
        # distance_matrix = -2 * feats.mm(torch.t(feats)) + feats.pow(2).sum(dim=1).view(1, -1) + feats.pow(2).sum(dim=1).view(-1, 1)
        # Uncomment below for l_inf distance
        feats = feats.view(feats.shape[0], -1)  # convert feats to be [batch_size, feat_size]
        # feats = (feats - feats.min(dim=0)[0]) / (feats.max(dim=0)[0] - feats.min(dim=0)[0])  # normalize to [0,1]
        if normalize_feats:
            feats = feats / feats.max(dim=1)[0].unsqueeze(1)  # normalize to have a max of 1
        feats_temp = feats.unsqueeze(1)
        distance_matrix = torch.zeros(feats.shape[0], feats.shape[0])
        closest_idx = torch.zeros(feats.shape[0]).long()
        for i in range(feats.shape[0]):
            tmp = torch.max(torch.abs((feats_temp - feats[i]).squeeze()), 1)[0]  # l_inf diff of a given feat i with every feat
            # find the closest feature to feat i
            distance_matrix[:, i] = tmp  # inf norm of each difference
            tmp[(labels == labels[i]).nonzero()] = float("inf")  # ignore the ones with same label
            min_dist, closest_idx[i] = torch.min(tmp, 0)

        # Index of each image and its closest non-similar image in this batch
        top_negative_pairs = torch.stack((torch.arange(0, feats.shape[0]), closest_idx), 1).long()

        # closest_pairs information
        closest_pairs = image_idxs[closest_idx]

        # Define loss
        # loss = (feats[top_negative_pairs[:, 0]] - feats[top_negative_pairs[:, 1]]).pow(2).sum(1)  # L2 loss
        # loss = torch.max(torch.abs(feats[top_negative_pairs[:, 0]] - feats[top_negative_pairs[:, 1]]), 1)[0]  # l_inf loss by recalculating differences
        loss = distance_matrix[top_negative_pairs[:, 0], top_negative_pairs[:, 1]]  # l_inf loss
        return loss.mean(), closest_pairs#, distance_matrix, top_negative_pairs, feats, feats_temp
