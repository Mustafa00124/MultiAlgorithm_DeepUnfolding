import numpy as np
import torch


def gaussian_3D(ksize, sigma, normalized = True):
    x = np.linspace(-ksize[0]/2, ksize[0]/2, ksize[0])
    y = np.linspace(-ksize[1]/2, ksize[1]/2, ksize[1])
    z = np.linspace(-ksize[2]/2, ksize[2]/2, ksize[2])
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    kernel = np.exp(-(xx ** 2 + yy ** 2 + zz ** 2) / (2 * sigma ** 2))

    if normalized:
        kernel = kernel/np.sum(kernel)

    return kernel


def stats_per_frame(masks, masks_target, rois=None):
    # roi_main = rois
    # mask_gt = masks_target
    if masks_target.max()==255:
        mask_gt = (masks_target == 255).astype(int) # F1 score computed for WHITE pixels (motion)
    else:
        mask_gt = (masks_target == 1).astype(int)
    roi_main = np.logical_and(masks_target != 85, masks_target != 170).astype(int) # ROI does not accounts for ROI and NON-UNKNOWN pixels
    smask = np.around(masks)
    TP = np.count_nonzero(roi_main * mask_gt * smask)
    TN = np.count_nonzero(roi_main * (1 - mask_gt) * (1 - smask))
    FP = np.count_nonzero(roi_main * (1 - mask_gt) * smask)
    FN = np.count_nonzero(roi_main * mask_gt * (1 - smask))
    return (TP, TN, FP, FN)


def stats_per_frame_v2(masks, masks_target):
    mask_gt = (masks_target == 255).astype(int)  # F1 score computed for WHITE pixels (motion)
    to_include = np.count_nonzero(mask_gt, axis=(-1, -2)) != 0
    if np.count_nonzero(to_include.astype(int)) == 0:
        return None
    indexes_to_retain = to_include.nonzero()
    roi_main = np.logical_and(masks_target != 85, masks_target != 170).astype(
        np.int)  # ROI does not accounts for ROI and NON-UNKNOWN pixels
    smask = np.around(masks)
    TP = np.count_nonzero(roi_main * mask_gt * smask, axis=(-1, -2))[indexes_to_retain]
    TN = np.count_nonzero(roi_main * (1 - mask_gt) * (1 - smask), axis=(-1, -2))[indexes_to_retain]
    FP = np.count_nonzero(roi_main * (1 - mask_gt) * smask, axis=(-1, -2))[indexes_to_retain]
    FN = np.count_nonzero(roi_main * mask_gt * (1 - smask), axis=(-1, -2))[indexes_to_retain]
    return (TP, TN, FP, FN)


def compute_F1(stats):
    # F1 score according to the CDNET2014 dataset
    TP_tot = sum(stat[0] for stat in stats)
    TN_tot = sum(stat[1] for stat in stats)
    FP_tot = sum(stat[2] for stat in stats)
    FN_tot = sum(stat[3] for stat in stats)
    epsilon = 1e-7
    precision = TP_tot / (TP_tot + FP_tot + epsilon)
    recall = TP_tot / (TP_tot + FN_tot + epsilon)
    return 2 * (precision * recall) / (precision + recall + epsilon)

def compute_pre_rec(stats):
    # precision, recall according to the CDNET2014 dataset
    TP_tot = sum(stat[0] for stat in stats)
    TN_tot = sum(stat[1] for stat in stats)
    FP_tot = sum(stat[2] for stat in stats)
    FN_tot = sum(stat[3] for stat in stats)
    epsilon = 1e-7
    precision = TP_tot / (TP_tot + FP_tot + epsilon)
    recall = TP_tot / (TP_tot + FN_tot + epsilon)
    return (precision, recall)


class TverskyLoss(torch.nn.Module):
    def __init__(self, alpha_t=0.5, beta_t=0.5,
                 # weight=True, size_average=True,
                 ):
        super(TverskyLoss, self).__init__()
        self.alpha_t = alpha_t
        self.beta_t = beta_t

    def forward(self, label, pred, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        # label = torch.sigmoid(label)

        # flatten label and prediction tensors

        # label = label.permute(3, 1, 2, 0)
        # pred = pred.permute(3, 1, 2, 0)

        flat_label = label.flatten()
        flat_pred = pred.flatten()
        TP = torch.sum(flat_label * flat_pred, -1)
        FN = torch.sum(flat_label * (1 - flat_pred), -1)
        FP = torch.sum((1 - flat_label) * flat_pred, -1)

        Tversky = (TP + smooth) / (TP + self.alpha_t * FP + self.beta_t * FN + smooth)
        # print(self.alpha_t, self.beta_t)

        return 1 - Tversky

class MultiTaskLoss(torch.nn.Module):
    def __init__(self, loss_fn, eta) -> None:
        super(MultiTaskLoss, self).__init__()
        self.loss_fn = loss_fn
        self.eta = torch.nn.Parameter(torch.Tensor(eta))

    def forward(self, outputs, targets) -> (torch.Tensor, torch.Tensor):
        # outputs = self.model(input)
        loss = [l(o,y) for l, o, y in zip(self.loss_fn, outputs, targets)]
        total_loss = torch.stack(loss) * torch.exp(-self.eta) + self.eta
        return loss, total_loss.sum() # omit 1/2
