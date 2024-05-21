from cProfile import label
import mmcv
import torch
import torch.nn as nn

from ..builder import LOSSES
from .utils import weighted_loss
import torch.nn.functional as F
from mmdet.models.losses import accuracy
from mmdet.models.losses.cross_entropy_loss import _expand_onehot_labels
from .utils import weight_reduce_loss
from mmdet.models.builder import build_loss
from mmdet.models.losses import FocalLoss
from sklearn.metrics import average_precision_score
import numpy as np

def binary_cross_entropy(pred,
                         label,
                         weight=None,
                         reduction='mean',
                         avg_factor=None,
                         class_weight=None):
    """Calculate the binary CrossEntropy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, 1).
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.

    Returns:
        torch.Tensor: The calculated loss
    """
    if pred.dim() != label.dim():
        label, weight = _expand_onehot_labels(label, weight, pred.size(-1))

    # weighted element-wise losses
    if weight is not None:
        weight = weight.float()
    loss = F.binary_cross_entropy(pred, label.float(), reduction='none')   # modified here
    # do the reduction for the weighted loss
    loss = weight_reduce_loss(
        loss, weight, reduction=reduction, avg_factor=avg_factor)
    return loss


class CrossEntropyLoss(nn.Module):
    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0):
        """CrossEntropyLoss.

        Args:
            use_sigmoid (bool, optional): Whether the prediction uses sigmoid
                of softmax. Defaults to False.
            use_mask (bool, optional): Whether to use mask cross entropy loss.
                Defaults to False.
            reduction (str, optional): . Defaults to 'mean'.
                Options are "none", "mean" and "sum".
            class_weight (list[float], optional): Weight of each class.
                Defaults to None.
            loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        """
        super(CrossEntropyLoss, self).__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = class_weight

        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        # elif self.use_mask:
        #     self.cls_criterion = mask_cross_entropy

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            cls_score (torch.Tensor): The prediction.
            label (torch.Tensor): The learning label of the prediction.
            weight (torch.Tensor, optional): Sample-wise loss weight.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction (str, optional): The method used to reduce the loss.
                Options are "none", "mean" and "sum".
        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(
                self.class_weight, device=cls_score.device)
        else:
            class_weight = None
        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_cls

# added by guo----------------------------------------------------------------------------------- #
@LOSSES.register_module()
class InsLoss(nn.Module):

    def __init__(self,
                 binary_ins=False,
                 loss_weight=1.0, eps=1e-6):
        """
        Args:
            use_binary (bool, optional): Whether to the prediction is
                used for binary cross entopy
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'. Options are "none", "mean" and
                "sum".
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
        """
        super(InsLoss, self).__init__()
        self.loss_weight = loss_weight
        self.eps = eps

    def pse_acc(self, pse_label, label):
        pse_acc = (pse_label == label).sum()/len(label)
        return pse_acc




    def forward(self, bag_cls_prob, labels, pse_labels, valid, weight=None):
        """
            bag_cls_outs: (B, N, C),
            bag_ins_outs: (B, N, C*2/C)
            valid: (B, N, 1/C)
            labels: (B, )
        Returns:
        """

        B, N, C = bag_cls_prob.shape
        bag_cls_prob = bag_cls_prob.reshape(-1, C)  # num_gt*num_prop C
        labels = torch.repeat_interleave(labels, N) # num_gt*num_prop C
        # weight = torch.repeat_interleave(weight, N).unsqueeze(-1)

        pse_labels = torch.repeat_interleave(pse_labels, N) 
        acc = accuracy(bag_cls_prob, labels)
        
        pse_acc = self.pse_acc(pse_labels, labels)
        valid = valid.view(-1,1)
        label_weights = (valid.sum(dim=-1) > 0).float()     # [num_gt, 1]label_weights都是1，应该是过滤掉一些无效的proposal，dim=1表示每个cluster中的42个proposal，有一个是有效的就可以。
        pse_labels = _expand_onehot_labels(pse_labels, None, C)[0].float()  # labels 是one_hot编码，shape应该是[num_gt, num_class]
        num_sample = max(torch.sum(label_weights > 0).float().item(), 1.)   # num_sample是batch中的num_gt
        prob = bag_cls_prob.clamp(0, 1)

        # modified by fei ##############################################################3
        # loss = F.nll_loss(prob, labels.float())
        loss = F.binary_cross_entropy(prob, pse_labels.float(), None, reduction="none")
        loss = weight_reduce_loss(loss, weight, avg_factor=num_sample) * self.loss_weight  # weight_reduce_loss: 有两个作用，一个是loss*weight,另一个是求平均。avg_factor是样本数目。
        return loss, acc, num_sample
        
        

# ------------------------------------------------------------------------------------------------
@LOSSES.register_module()
class MSEntropyLoss(nn.Module):

    def __init__(self,
                 # use_binary=True,
                 # reduction='mean',
                 binary_ins=False,
                 loss_weight=1.0, eps=1e-6, loss_type='gfocal_loss'):
        """
        Args:
            use_binary (bool, optional): Whether to the prediction is
                used for binary cross entopy
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'. Options are "none", "mean" and
                "sum".
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
        """
        super(MSEntropyLoss, self).__init__()
        # self.use_binary = use_binary
        # self.reduction = reduction
        self.loss_weight = loss_weight
        # if self.use_sigmoid:
        # self.loss_cls = CrossEntropyLoss(use_sigmoid=True, loss_weight=loss_weight)
        self.eps = eps
        self.loss_type = loss_type
        self.binary_ins = binary_ins

    def gfocal_loss(self, p, q, w=1.0):
        l1 = (p - q) ** 2
        l2 = q * (p + self.eps).log() + (1 - q) * (1 - p + self.eps).log()
        return -(l1 * l2 * w).sum(dim=-1)


    def forward(self, bag_cls_prob,  labels, valid, actual_labels=None, weight=None):
        """
            bag_cls_outs: (B, N, C),
            bag_ins_outs: (B, N, C*2/C)
            valid: (B, N, 1/C)
            labels: (B, )
        Returns:
        """


        B, N, C = bag_cls_prob.shape
        prob_cls = bag_cls_prob.unsqueeze(dim=-1)  # (B, N, C, 1)
        
        

        prob = (prob_cls).mean(dim=1)       # (B C 1)  

        if actual_labels is not None:
            acc = accuracy(prob[..., 0], actual_labels)
        else:
            acc = accuracy(prob[..., 0], labels)

        label_weights = (valid.sum(dim=1) > 0).float()     # [num_gt, 1]label_weights都是1，应该是过滤掉一些无效的proposal，dim=1表示每个cluster中的42个proposal，有一个是有效的就可以。
        
        labels = _expand_onehot_labels(labels, None, C)[0].float()  # labels 是one_hot编码，shape应该是[num_gt, num_class]
        
        num_sample = max(torch.sum(label_weights.sum(dim=-1) > 0).float().item(), 1.)   # num_sample是batch中的num_gt
        
        if prob.shape[-1] == 1:
            prob = prob.squeeze(dim=-1)
        elif prob.shape[-1] == 2:  # with binary ins
            pos_prob, neg_prob = prob[..., 0], prob[..., 1]
            prob = torch.cat([pos_prob, neg_prob])
            neg_labels = labels.new_zeros(labels.shape)
            labels = torch.cat([labels, neg_labels])
            label_weights = torch.cat([label_weights, label_weights])

        if self.loss_type == 'gfocal_loss':
            loss = self.gfocal_loss(prob, labels, label_weights)
            if weight is not None:
                # modified by fei ##############################################################3
                weight=weight.squeeze(-1)
        elif self.loss_type == 'binary_cross_entropy':
            # if self.use_sigmoid:
            # method 1:
            # loss = self.loss_cls(
            #     prob,
            #     labels,
            #     label_weights,
            #     avg_factor=avg_factor,
            #     reduction_override=reduction_override)
            # method 2
            prob = prob.clamp(0, 1)
            # modified by fei ##############################################################3
            loss = F.binary_cross_entropy(prob, labels.float(), None, reduction="none")
        else:
            raise ValueError()
        loss = weight_reduce_loss(loss, weight, avg_factor=num_sample) * self.loss_weight
        return loss, acc, num_sample
# -------------------------------------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------------------------------------ #
@LOSSES.register_module()
class MILLoss(nn.Module):

    def __init__(self,
                 # use_binary=True,
                 # reduction='mean',
                 binary_ins=False,
                 loss_weight=1.0, eps=1e-6, loss_type='gfocal_loss'):
        """
        Args:
            use_binary (bool, optional): Whether to the prediction is
                used for binary cross entopy
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'. Options are "none", "mean" and
                "sum".
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
        """
        super(MILLoss, self).__init__()
        # self.use_binary = use_binary
        # self.reduction = reduction
        self.loss_weight = loss_weight
        # if self.use_sigmoid:
        # self.loss_cls = CrossEntropyLoss(use_sigmoid=True, loss_weight=loss_weight)
        self.eps = eps
        self.loss_type = loss_type
        self.binary_ins = binary_ins

    def gfocal_loss(self, p, q, w=1.0):
        l1 = (p - q) ** 2
        l2 = q * (p + self.eps).log() + (1 - q) * (1 - p + self.eps).log()
        return -(l1 * l2 * w).sum(dim=-1)


    def forward(self, bag_cls_prob, bag_ins_outs, labels, valid, actual_labels=None, weight=None):
        """
            bag_cls_outs: (B, N, C),
            bag_ins_outs: (B, N, C*2/C)
            valid: (B, N, 1/C)
            labels: (B, )
        Returns:
        """
        # if self.binary_ins:
        #     assert bag_ins_outs.shape[-1] / bag_cls_prob.shape[-1] == 2
        # else:
        #     assert bag_ins_outs.shape[-1] == bag_cls_prob.shape[-1]

        B, N, C = bag_cls_prob.shape
        prob_cls = bag_cls_prob.unsqueeze(dim=-1)  # (B, N, C, 1)
        
        # =======================================================================#
        # prob_ins = bag_ins_outs.reshape(B, N, C, -1)  # (B, N, C, 2/1)
        # prob_ins = prob_ins.softmax(dim=1) * valid.unsqueeze(dim=-1)
        # prob_ins = F.normalize(prob_ins, dim=1, p=1)
        # prob = (prob_cls * prob_ins).sum(dim=1)
        #========================================================================#
        
        # =======================================================================#
        if bag_ins_outs is not None:
            prob_ins = bag_ins_outs.reshape(B, N, C, -1)  # (B, N, C, 1)
            prob_ins = prob_ins.softmax(dim=1) * valid.unsqueeze(dim=-1)
            prob_ins_ = prob_ins[torch.arange(len(prob_ins)),:,labels,:]
            prob_cls_ = prob_cls[torch.arange(len(prob_cls)), :, labels,:]
            prob_ = (prob_ins_ * prob_cls_).sum(dim=1)
            prob = prob_cls.mean(dim=1)
            prob[torch.arange(len(prob_cls)), labels,:] = prob_
        else:
            prob = prob_cls.mean(dim=1)
        # =============================================================================#
        # prob_ins = F.normalize(prob_ins, dim=1, p=1)   # NOTE !!!! zhu shi

        # for i in range(len(prob_ins)):
            
        ###############################################
        # prob_ins_ = prob_ins[torch.arange(len(prob_ins)),:,labels]
        # prob_cls_ = prob_cls[torch.arange(len(prob_cls)), :, labels]
        # prob_ = (prob_ins_ * prob_cls_).sum(dim=0)
        # prob = prob_cls.mean(dim=1)
        # prob[torch.arange(len(prob_cls)), :, labels] = prob_
        
        ##############################################

        

        if actual_labels is not None:
            acc = accuracy(prob[..., 0], actual_labels)
        else:
            acc = accuracy(prob[..., 0], labels)

        label_weights = (valid.sum(dim=1) > 0).float()     # [num_gt, 1]label_weights都是1，应该是过滤掉一些无效的proposal，dim=1表示每个cluster中的42个proposal，有一个是有效的就可以。
        
        labels = _expand_onehot_labels(labels, None, C)[0].float()  # labels 是one_hot编码，shape应该是[num_gt, num_class]
        
        num_sample = max(torch.sum(label_weights.sum(dim=-1) > 0).float().item(), 1.)   # num_sample是batch中的num_gt
        
        if prob.shape[-1] == 1:
            prob = prob.squeeze(dim=-1)
        elif prob.shape[-1] == 2:  # with binary ins
            pos_prob, neg_prob = prob[..., 0], prob[..., 1]
            prob = torch.cat([pos_prob, neg_prob])
            neg_labels = labels.new_zeros(labels.shape)
            labels = torch.cat([labels, neg_labels])
            label_weights = torch.cat([label_weights, label_weights])

        if self.loss_type == 'gfocal_loss':
            loss = self.gfocal_loss(prob, labels, label_weights)
            if weight is not None:
                # modified by fei ##############################################################3
                weight=weight.squeeze(-1)
        elif self.loss_type == 'binary_cross_entropy':
            # if self.use_sigmoid:
            # method 1:
            # loss = self.loss_cls(
            #     prob,
            #     labels,
            #     label_weights,
            #     avg_factor=avg_factor,
            #     reduction_override=reduction_override)
            # method 2
            prob = prob.clamp(0, 1)
            # modified by fei ##############################################################3
            loss = F.binary_cross_entropy(prob, labels.float(), None, reduction="none")
        else:
            raise ValueError()
        loss = weight_reduce_loss(loss, weight, avg_factor=num_sample) * self.loss_weight
        return loss, acc, num_sample


# --------------------------------- added by guo ---------------------------------------------------------
@LOSSES.register_module()
class ImgMILLoss(nn.Module):
    def __init__(self,
                 # use_binary=True,
                 # reduction='mean',
                 binary_ins=False,
                 loss_weight=1.0, eps=1e-6, loss_type='gfocal_loss'):
        """
        Args:
            use_binary (bool, optional): Whether to the prediction is
                used for binary cross entopy
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'. Options are "none", "mean" and
                "sum".
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
        """
        super(ImgMILLoss, self).__init__()
        # self.use_binary = use_binary
        # self.reduction = reduction
        self.loss_weight = loss_weight
        # if self.use_sigmoid:
        # self.loss_cls = CrossEntropyLoss(use_sigmoid=True, loss_weight=loss_weight)
        self.eps = eps
        self.loss_type = loss_type
        self.binary_ins = binary_ins

    def gfocal_loss(self, p, q, w=1.0):
        l1 = (p - q) ** 2
        l2 = q * (p + self.eps).log() + (1 - q) * (1 - p + self.eps).log()
        return -(l1 * l2 * w).sum(dim=-1)

    def getLabelVector(self, labels, C):
        label = torch.zeros(C, device='cuda')
        for c in labels:
            index = c
            label[index] = 1.0 # / label_num
        return label
    
    def compute_mAP(self, outputs, labels):    # added by guo
        y_true = labels.cpu().detach().numpy()
        y_pred = outputs.cpu().detach().numpy()
        AP = []
        for i in range(y_true.shape[0]):
            if np.sum(y_true[i]) > 0:
                
                ap_i = average_precision_score(y_true[i], y_pred[i])
                AP.append(ap_i)
                # print(ap_i)
        mAP = sum(AP)/len(AP)
        return mAP

    def forward(self, bag_cls_prob, bag_ins_outs, labels, valid, weight=None):
        """
            bag_cls_outs: (N, C),
            bag_ins_outs: (N, C*2/C)
            valid: ( N, 1/C)
            labels: (N, )
            N指的是1个batch中所有图片所有proposal的数目之和
        Returns:
        """
        assert len(bag_cls_prob) == len(bag_ins_outs)

        if self.binary_ins:
            assert bag_ins_outs[0].shape[-1] / bag_cls_prob[0].shape[-1] == 2
        else:
            assert bag_ins_outs[0].shape[-1] == bag_cls_prob[0].shape[-1]
        # print(bag_cls_prob.shape)
        # print(bag_ins_outs.shape)
        # print(labels.shape)
        # print(valid.shape)
        # print(labels)
        # exit()
        prob_list = []
        labels_list = []
        B = len(bag_cls_prob)
        for i in range(B):
            N, C = bag_cls_prob[i].shape
            prob_cls = bag_cls_prob[i].unsqueeze(dim=-1)  # (N, C, 1)
            prob_ins = bag_ins_outs[i].reshape(N, C, -1)  # (N, C, 1)
            prob_ins = prob_ins.softmax(dim=0) * valid[i].unsqueeze(dim=-1)  
            prob_ins = F.normalize(prob_ins, dim=0, p=1)
            prob = (prob_cls * prob_ins).sum(dim=0)       # (C 1)  从实例维度上求和，得到每一个cluster的得分
            prob_list.append(prob)
            
            label_weights = valid[i]
            labels_per_img = self.getLabelVector(labels[i], C).float()
            labels_list.append(labels_per_img)
        
        num_sample = B

        prob = torch.stack(prob_list)
        labels = torch.stack(labels_list)
        acc = self.compute_mAP(prob[..., 0], labels)
        
        # prob_cls = bag_cls_prob.unsqueeze(dim=-1)  # (B, N, C, 1)
        
        # prob_ins = bag_ins_outs.reshape(B, N, C, -1)  # (B, N, C, 1)
        
        # prob_ins = prob_ins.softmax(dim=1) * valid.unsqueeze(dim=-1)
        # prob_ins = F.normalize(prob_ins, dim=1, p=1)
        # prob = (prob_cls * prob_ins).sum(dim=1)       # (B C 1)  从实例维度上求和，得到每一个cluster的得分
        # acc = accuracy(prob[..., 0], labels)      

        # label_weights = (valid.sum(dim=1) > 0).float()
        # labels = _expand_onehot_labels(labels, None, C)[0].float()
        # num_sample = max(torch.sum(label_weights.sum(dim=-1) > 0).float().item(), 1.)

        if prob.shape[-1] == 1:
            prob = prob.squeeze(dim=-1)
        elif prob.shape[-1] == 2:  # with binary ins
            pos_prob, neg_prob = prob[..., 0], prob[..., 1]
            prob = torch.cat([pos_prob, neg_prob])
            neg_labels = labels.new_zeros(labels.shape)
            labels = torch.cat([labels, neg_labels])
            label_weights = torch.cat([label_weights, label_weights])
        
        if self.loss_type == 'gfocal_loss':
            
            # loss = self.gfocal_loss(prob, labels, label_weights)
            loss = self.gfocal_loss(prob, labels)  # 把原来的label_weights去掉了，因为之前是实例级的损失，可能有的实例是invalid，但是现在是图像级的损失。

            # if weight is not None:
            #     print(weight)
            #     # modified by fei ##############################################################3
            #     weight=weight.squeeze(-1)
        elif self.loss_type == 'binary_cross_entropy':
            # if self.use_sigmoid:
            # method 1:
            # loss = self.loss_cls(
            #     prob,
            #     labels,
            #     label_weights,
            #     avg_factor=avg_factor,
            #     reduction_override=reduction_override)
            # method 2
            prob = prob.clamp(0, 1)
            # modified by fei ##############################################################3
            loss = F.binary_cross_entropy(prob, labels.float(), None, reduction="none")
        else:
            raise ValueError()
        loss = weight_reduce_loss(loss, weight, avg_factor=num_sample) * self.loss_weight   # loss_weight 是损失占的权重， weight是各个样本的损失占的权重。
        return loss, torch.tensor(acc, device='cuda').float(), num_sample  # TODO: acc还没实现，先用0代替



@LOSSES.register_module()
class AllPosLoss(MILLoss):
    def forward(self, bag_cls_prob, bag_ins_outs, labels, valid, weight=None):
        """
            bag_cls_outs: (B, N, C),
            # bag_ins_outs: (B, N, C*2/C)
            valid: (B, N, 1/C)
            labels: (B, )
        Returns:
        """
        B, N, C = bag_cls_prob.shape
        prob_cls = bag_cls_prob.unsqueeze(dim=-1)
        prob = prob_cls.reshape(B*N, C)
        labels = labels.unsqueeze(dim=-1).repeat(1, N).flatten()  # (B*N, )
        valid = valid.reshape(B*N, -1)
        acc = accuracy(prob, labels)

        label_weights = valid.float()
        labels = _expand_onehot_labels(labels, None, C)[0].float()
        num_sample = max(torch.sum(label_weights.sum(dim=-1) > 0).float().item(), 1.)

        if self.loss_type == 'gfocal_loss':
            loss = self.gfocal_loss(prob, labels, label_weights)
        elif self.loss_type == 'binary_cross_entropy':
            # if self.use_sigmoid:
            # method 1:
            # loss = self.loss_cls(
            #     prob,
            #     labels,
            #     label_weights,
            #     avg_factor=avg_factor,
            #     reduction_override=reduction_override)
            # method 2
            loss = F.binary_cross_entropy(prob, labels.float(), weight, reduction="none")
        else:
            raise ValueError()
        loss = weight_reduce_loss(loss, weight, avg_factor=num_sample) * self.loss_weight
        return loss + bag_ins_outs * 0, acc, num_sample


# @LOSSES.register_module()
# class MIL2Loss(MILLoss):
#
#     def __init__(self, **kwargs):
#         """
#         Args:
#             use_sigmoid (bool, optional): Whether to the prediction is
#                 used for sigmoid or softmax. Defaults to True.
#             reduction (str, optional): The method used to reduce the loss into
#                 a scalar. Defaults to 'mean'. Options are "none", "mean" and
#                 "sum".
#             loss_weight (float, optional): Weight of loss. Defaults to 1.0.
#         """
#         super(MIL2Loss, self).__init__(**kwargs)
#         self.pos_k = 5
#         self.neg_k = 5
#
#     def focal_loss(self, p, q, w):
#         return - ((p + self.eps).log() * (q - p) ** 2 * w +
#                   (1 - p + self.eps).log() * (1 - q - p) ** 2 * (1-w))
#
#     def forward(self, bag_cls_outs, bag_ins_outs, labels, weight=None, avg_factor=None, reduction_override=None):
#         """
#         Returns:
#         """
#         all_gt_idx = torch.arange(len(bag_ins_outs)).to(bag_ins_outs.device)
#
#         prob_cls = bag_cls_outs.softmax(dim=2)
#         prob_ins = bag_ins_outs.sigmoid()
#         assert prob_ins.shape[-1] in [80, 81]
#
#         prob = (prob_cls * prob_ins).sum(dim=1)  # (num_gt, C)
#         acc = accuracy(prob, labels)
#
#         prob = prob_cls[all_gt_idx, :, labels]
#         sample_q = prob_ins[all_gt_idx, :, labels]
#
#         sort_prob, sort_idx = prob.sort(dim=1)
#         loss = - sort_prob[:, -self.pos_k:].log().mean(dim=1) \
#                - (1 - sort_prob[:, :self.neg_k]).log().mean(dim=1) \
#                - ((prob + self.eps).log() * ((1-prob) ** 2) * sample_q + (1 - prob + self.eps).log() * (1 - sample_q)).mean(dim=1)
#         loss = self.focal_loss(sort_prob[:, -self.pos_k:], 1)
#         loss = loss / 3
#
#         # loss = - sort_prob[:, -self.pos_k:].log().mean(dim=1) \
#         #        - (1 - sort_prob[:, :self.neg_k]).log().mean(dim=1)
#         # loss = loss / 2
#
#         # loss = - sort_prob[:, -self.pos_k:].log().sum(dim=1) \
#         #        - (1 - sort_prob[:, :self.neg_k]).log().sum(dim=1) \
#         #        - ((prob + self.eps).log() * sample_q + (1 - prob + self.eps).log() * (1 - sample_q)).sum(dim=1)
#         # loss = loss / (num_samples + self.pos_k + self.neg_k)
#
#         loss = weight_reduce_loss(loss, weight, avg_factor=avg_factor)
#         return loss, {
#             "bag_acc": acc,
#             "q_dis": 2*(sample_q - 0.5).abs().mean(),
#             "r_pos": (sample_q > 0.5).float().sum() / (sample_q >= 0).float().sum()
#         }

    # def forward(self, bag_cls_outs, bag_ins_outs, labels,
    #             weight=None, avg_factor=None, reduction_override=None):
    #     """
    #     P_cls = softmax(O_cls, dim=-1)  # (num_gt, num_samples, C)
    #     P_ins = sigmoid(O_ins)          # (num_gt, num_samples, C, 2)
    #     P_cls = P_cls[all_gt_idx, :, labels]
    #     (P_cls * P_ins).sum(dim=-2)
    #     (1-P_cls) * (1-P_ins).sum(dim=-2)
    #
    #         bag_cls_outs: (B, N, C),
    #         bag_ins_outs: (B, N, C)
    #         labels: (B, )
    #     Returns:
    #     """
    #     if not self.use_binary:
    #         return self.forward2(bag_cls_outs, bag_ins_outs, labels, weight, avg_factor, reduction_override)
    #
    #     num_cls = bag_cls_outs.shape[-1] // 2
    #     all_gt_idx = torch.arange(len(bag_ins_outs)).to(bag_ins_outs.device)
    #
    #     prob_cls = bag_cls_outs[..., :num_cls].softmax(dim=2)
    #     prob_ins = bag_ins_outs.softmax(dim=1).reshape(*prob_cls.shape, 2)
    #     prob = (prob_cls * prob_ins[..., 0]).sum(dim=1)
    #
    #     acc = accuracy(prob, labels)
    #
    #     pos_prob = prob[all_gt_idx, labels]
    #     prob_cls = prob_cls[all_gt_idx, :, labels]  # (num_gt, num_sample)
    #     prob_ins = prob_ins[all_gt_idx, :, labels]  # (num_gt, num_sample, 2)
    #     neg_prob = ((1 - prob_cls) * prob_ins[..., 1]).sum(dim=1)
    #     prob = torch.stack([pos_prob, neg_prob], dim=-1)
    #
    #     label_weights = torch.full((prob.shape[0],), 1, dtype=prob.dtype).to(prob.device)
    #     avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
    #
    #     labels = torch.ones(len(labels), 2).to(prob.device)
    #     # labels, _ = _expand_onehot_labels(labels, None, prob.shape[-1])
    #     loss = F.binary_cross_entropy(prob, labels.float(), weight, reduction="none")
    #     loss = weight_reduce_loss(loss, weight, avg_factor=avg_factor)
    #     return loss, acc
    #
    # def forward2(self, bag_cls_outs, bag_ins_outs, labels,
    #             weight=None, avg_factor=None, reduction_override=None):
    #     """
    #     P_cls = softmax(O_cls, dim=-1)  # (num_gt, num_samples, C)
    #     P_ins = sigmoid(O_ins)          # (num_gt, num_samples, C, 2)
    #     P_cls = P_cls[all_gt_idx, :, labels]
    #     (P_cls * P_ins).sum(dim=-2)
    #     (1-P_cls) * (1-P_ins).sum(dim=-2)
    #
    #         bag_cls_outs: (B, N, C),
    #         bag_ins_outs: (B, N, C)
    #         labels: (B, )
    #     Returns:
    #     """
    #     num_cls = bag_cls_outs.shape[-1] // 2
    #     all_gt_idx = torch.arange(len(bag_ins_outs)).to(bag_ins_outs.device)
    #
    #     prob_cls = bag_cls_outs[..., :num_cls].softmax(dim=2)
    #     prob_ins = bag_ins_outs[..., :num_cls].softmax(dim=1)
    #     prob = (prob_cls * prob_ins).sum(dim=1)  # (num_gt, C)
    #
    #     acc = accuracy(prob, labels)
    #
    #     pos_prob = prob[all_gt_idx, labels]
    #     neg_prob, neg_label = ((labels.unsqueeze(dim=1) != torch.arange(num_cls)
    #                             .to(labels.device)).float() * prob).max(dim=1)
    #
    #     prob = torch.stack([pos_prob, neg_prob], dim=-1)
    #
    #     label_weights = torch.full((prob.shape[0],), 1, dtype=prob.dtype).to(prob.device)
    #     avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
    #
    #     labels = torch.ones(len(labels), 2).to(prob.device)
    #     # labels, _ = _expand_onehot_labels(labels, None, prob.shape[-1])
    #     loss = F.binary_cross_entropy(prob, labels.float(), weight, reduction="none")
    #     loss = weight_reduce_loss(loss, weight, avg_factor=avg_factor)
    #     return loss, acc
