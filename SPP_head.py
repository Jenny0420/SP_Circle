import torch
from math import ceil
import torch.nn as nn
import numpy as np

class ConvResidual(nn.Module):
    def __init__(self, inc, ouc,  stride):
        super(ConvResidual, self).__init__()
        self._pad = ceil((3 - stride) / 2)
        self.stride = stride
        self.convs = nn.Sequential(
            nn.Conv2d(inc, ouc, kernel_size=3, stride=stride, padding=self._pad, bias=False),
            nn.BatchNorm2d(ouc),
            nn.LeakyReLU(0.1),
            nn.Conv2d(ouc, ouc, kernel_size=3, stride=1, padding=self._pad, bias=False),
            nn.BatchNorm2d(ouc),
            nn.LeakyReLU(0.1)
        )
        if self.stride != 1:
            self.res = nn.Sequential(
                nn.Conv2d(inc, ouc, kernel_size=3, stride=stride, padding=self._pad, bias=False),
                nn.BatchNorm2d(ouc),
                nn.LeakyReLU(0.1)
            )
        else:
            self.res = nn.Sequential(
                nn.Conv2d(inc, ouc, kernel_size=1, bias=False),
                nn.BatchNorm2d(ouc),
                nn.LeakyReLU(0.1)
            )


    def forward(self, x):
        inplace = x
        main_out = self.convs(x)
        inplace = self.res(inplace)
        out = main_out + inplace
        return out

class SPP_head(nn.Module):
    def __init__(self, conf_thresh):
        super(SPP_head, self).__init__()
        self.conf_thresh = conf_thresh
        self.convs = nn.Sequential(
            ConvResidual(128, 256, 1)
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 3, kernel_size=1)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, heat_map):
        heat_map_npy = heat_map.detach().cpu().numpy()
        x = self.convs(heat_map)
        out = torch.sigmoid(self.final_conv(x))
        out_npy = out.detach().cpu().numpy()

        return out

    def _post_processing(self, prob):
        x = prob.detach().cpu().numpy()
        points = []
        for x_ in x:
            all, _, _ = find_peak(x_[:-2])
            C, I, J = np.where(all)  ## class, i and j
            S = x_[:-2][C, I, J]  ## scores list
            idx = S.argsort()[::-1]  ## sort by score, in descending order
            C, I, J, S = map(lambda x: x[idx], [C, I, J, S])

            points.append([])
            for _, (c, i, j, s) in zip(range(8), zip(C, I, J, S)):
                if s < self.conf_thresh: continue  ## below confidence threshold
                off = (x_[-2:, i, j] - 0.5) * 1.25 + 0.5
                off = np.clip(off, 0, 1)
                y, x = ([i, j] + off) * 8
                points[-1].append((y, x, s))
        return points

class Focal_Loss(nn.Module):
    def __init__(self, gamma=2, beta = 4, num_class=1, loss_weight=None):
        super(Focal_Loss, self).__init__()

        if loss_weight is None: loss_weight = {}
        self._num_class = num_class
        self.gamma = gamma
        self.beta = beta
        self._binary_cross_entropy_fn = nn.BCELoss(reduction="none")
        self.l1_smooth = nn.SmoothL1Loss(reduction='none')
        self._losses = {}
        self._loss = 0

        self._loss_weights = {}

        for key in {"pos", "neg", "crd"}:
            self._loss_weights[key] = loss_weight.get(key, 1)

    def forward(self, probs, labels):

        real_class, real_yx = labels[:, :-2], labels[:, -2:]
        pred_class, pred_yx = probs[:, :-2], probs[:, -2:]
        # b = real_class.detach().numpy()
        # a = pred_class.detach().numpy()

        ## mask
        class_mask = (real_class > 0.999).float()  ## meaning [b, class, i, j] has an object
        object_mask = (torch.sum(class_mask, dim=1, keepdim=True) > 0).float()  ## meaning [b, :, i, j] has an object

        neg_weights = torch.pow(1 - real_class, self.beta)
        ## confidence - focal loss

        loss_pos = - torch.log(pred_class + 1e-7) * torch.pow(1 - pred_class, self.gamma) * class_mask
        loss_neg = - torch.log(1 - pred_class + 1e-7) * torch.pow(pred_class, self.gamma) * neg_weights * (1-class_mask)
        # n = loss_neg.detach().numpy()
        # p = loss_pos.detach().numpy()

        ## yx - kl loss
        loss_crd = self.l1_smooth(pred_yx, real_yx) * object_mask

        ## loss
        b, c, *_ = real_class.shape
        npos = class_mask.sum()
        nneg = np.prod(class_mask.shape) - npos

        losses = {
            "pos": loss_pos.sum() / (npos + 1e-8),
            "neg": loss_neg.sum() / (nneg + 1e-8),
            "crd": loss_crd.sum() / (npos + 1e-8)
        }

        loss = sum([v * self._loss_weights[k] for k, v in losses.items()])

        self._loss, _losses = loss, losses
        self._losses = {k: v.item() for k, v in _losses.items()}

        return self._loss, self._losses
# class Loss_function(nn.Module):
#     def __init__(self, alpha=1, beta=1):
#         super(Loss_function, self).__init__()
#         self.alpha = alpha
#         self.beta = beta
#         self.bce = nn.BCELoss(reduction='mean')
#         self.L1 = nn.SmoothL1Loss(reduction='mean')
#
#     def forward(self, pred, labels):
#         real_class, real_yx = labels[:, :-2], labels[:, -2:]
#         pred_class, pred_yx = pred[:, :-2], pred[:, -2:]
#
#         class_loss = self.bce(pred_class, real_class)
#         bios_loss = self.L1(pred_yx, real_yx)
#
#         loss = self.alpha * class_loss + self.beta * bios_loss
#
#         return loss, class_loss, bios_loss

def zero_pad(x):
    b, h, w = x.shape
    x_ = np.zeros([b, h + 2, w + 2])
    x_[..., 1:h + 1, 1:w + 1] = x
    return x_

def find_peak(x):
    h, w = x.shape[-2:]

    xtl, xtc, xtr = x[..., 0:h - 2, 0:w - 2], x[..., 0:h - 2, 1:w - 1], x[..., 0:h - 2, 2:w]
    xcl, xcc, xcr = x[..., 1:h - 1, 0:w - 2], x[..., 1:h - 1, 1:w - 1], x[..., 1:h - 1, 2:w]
    xbl, xbc, xbr = x[..., 2:h - 0, 0:w - 2], x[..., 2:h - 0, 1:w - 1], x[..., 2:h - 0, 2:w]

    dtl, dtc, dtr = map(lambda x: xcc - x, [xtl, xtc, xtr])
    dcl, dcr = map(lambda x: xcc - x, [xcl, xcr])
    dbl, dbc, dbr = map(lambda x: xcc - x, [xbl, xbc, xbr])

    right_cross = (dtc > 0) & (dbc > 0) & (dcr > 0) & (dcl > 0)
    diag_cross = (dtl > 0) & (dbl > 0) & (dtr > 0) & (dbr > 0)
    all_cross = right_cross & diag_cross

    return zero_pad(all_cross), zero_pad(right_cross), zero_pad(diag_cross)
        