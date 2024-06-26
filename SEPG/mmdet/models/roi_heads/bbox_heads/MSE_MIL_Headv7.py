import torch.nn as nn
from mmcv.cnn import ConvModule
import torch
from mmdet.models.builder import HEADS
from mmdet.models.utils import build_linear_layer
from .bbox_head import BBoxHead
from mmcv.runner import BaseModule, auto_fp16, force_fp32
from mmdet.models.losses import accuracy
from mmdet.models.builder import build_loss
from mmdet.models.losses.cross_entropy_loss import _expand_onehot_labels
import torch.nn.functional as F

num_clusters = 5   # NOTE !!

@HEADS.register_module()
class MSEMILHeadv7(BBoxHead):

    def __init__(self,
                 num_stages=1,
                 num_shared_fcs=2,
                 num_cls_fcs=0,
                 num_reg_fcs=0,
                 num_ref_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 loss_type=None,
                 with_loss_pseudo=True,
                 loss_mil1=dict(
                     type='MILLoss',
                     binary_ins=False,
                     loss_weight=0.25),  # weight
                 loss_mil2=None,
                 conv_cfg=None,
                 norm_cfg=None,
                 init_cfg=dict(
                     type='Normal',
                     layer=['Conv2d', 'Linear'],
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='fc_cls',
                         std=0.01,
                         bias_prob=0.01), ),
                 *args,
                 **kwargs):
        super(MSEMILHeadv7, self).__init__(
            *args, init_cfg=init_cfg, **kwargs)
        assert (num_shared_fcs + num_cls_fcs + num_reg_fcs > 0)

        self.with_reg = False # ori is False, Note guo ##################################


        if not self.with_cls:
            assert num_cls_fcs == 0
        if not self.with_reg:
            assert num_reg_fcs == 0
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_fcs = num_reg_fcs
        self.num_ref_fcs = num_ref_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.loss_mil1 = build_loss(loss_mil1)
        self.loss_mil2 = build_loss(loss_mil2)
        self.with_ins = self.with_cls
        self.num_stages = num_stages
        self.loss_type = loss_type
        self.with_loss_pseudo = with_loss_pseudo
        
        # 添加cls和ins共享的卷积层
        # add shared convs and fcs
        _, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(0, self.num_shared_fcs, self.in_channels, True)
        self.shared_out_channels = last_layer_dim

        # add cls specific branch  添加cls特有的卷积层
        _, self.cls_fcs, self.cls_last_dim = \
            self._add_conv_fc_branch(0, self.num_cls_fcs, self.shared_out_channels)

        # add sim fc for 1st branch
        # _, self.sim_fcs, self.sim_last_dim = \
        #     self._add_conv_fc_branch(0, self.num_shared_fcs, self.in_channels, True)
        self.sim_fcs = nn.ModuleList()
        self.sim_fcs.extend([nn.Linear(self.in_channels*self.roi_feat_area, 512),
                            nn.Linear(512, 512),
                            nn.Linear(512, 128)])
        # add sim fc for 2st branch
        # _, self.sim_fcs_2, self.sim_last_dim = \
        #     self._add_conv_fc_branch(0, self.num_shared_fcs, self.shared_out_channels)
        
        # add ins specific branch  添加ins特有的卷积层
        _, self.ins_fcs, self.ins_last_dim = \
            self._add_conv_fc_branch(0, self.num_cls_fcs, self.shared_out_channels)

        # add reg specific branch
        if self.with_reg:
            _, self.reg_fcs, self.reg_last_dim = \
                self._add_conv_fc_branch(0, self.num_reg_fcs, self.shared_out_channels)
        if self.num_stages == 2:
            _, self.ref_fcs, self.ref_last_dim = \
                self._add_conv_fc_branch(0, self.num_ref_fcs, self.in_channels, is_shared=True)
            if self.num_stages == 3:
                _, self.ref2_fcs, self.ref2_last_dim = \
                    self._add_conv_fc_branch(0, self.num_ref_fcs, self.in_channels, is_shared=True)
                
        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_cls_fcs == 0:
                self.cls_last_dim *= self.roi_feat_area
            if self.num_cls_fcs == 0:
                self.ins_last_dim *= self.roi_feat_area
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_area

        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed

        # 添加分类头，实例头
        self.fc_cls = nn.ModuleList()
        # self.fc_ins = nn.ModuleList()
        self.fc_reg = nn.ModuleList()
        for i in range(self.num_stages):
            if i < 1:
                num_cls = self.num_classes + 1
            else:
                num_cls = self.num_classes
            if self.with_cls:
                self.fc_cls.append(build_linear_layer(
                    self.cls_predictor_cfg,
                    in_features=self.cls_last_dim,
                    out_features=num_cls))
            # if self.with_ins:
            #     if i > 0:
            #         self.fc_ins.append(build_linear_layer(
            #             self.cls_predictor_cfg,
            #             in_features=self.ins_last_dim,
            #             out_features=num_cls))
            if self.with_reg:
                out_dim_reg = (4 if self.reg_class_agnostic else 4 *
                                                                 self.num_classes)
                self.fc_reg.append(build_linear_layer(
                    self.reg_predictor_cfg,
                    in_features=self.reg_last_dim,
                    out_features=out_dim_reg))
        # 语义原型
        self.semantic_proto = nn.Embedding(self.num_classes+1, 128)
        self.semantic_proto.weight.requires_grad = False
        with torch.no_grad():
            nn.init.zeros_(self.semantic_proto.weight)
        
        # self.meta_prototypes = [[] for cls_ in range(self.num_classes)]

        # self.cls_wise_feature = nn.Embedding(self.num_classes, 256)
        # cls_wise_feature = self.cls_wise_feature.weight.clone()

    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch.
        
        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim


    def forward(self, x, stage, gt_points, gt_labels, unique_id):
        # shared part
        # x: [N 256, 7, 7]
        # if stage == 0:
        #     print(self.semantic_proto.weight)
        
        if stage == 0:
            if self.num_shared_fcs > 0:
                if self.with_avg_pool:
                    x = self.avg_pool(x)
                x = x.flatten(1)                 # x: [N, 256*7*7]
                x_sim = x
                for fc in self.shared_fcs:
                    x = self.relu(fc(x))         # x: [N, 1024]
                
                for fc in self.sim_fcs:
                    x_sim = self.relu(fc(x_sim))
                
                if gt_points is not None:
                    num_gt = torch.cat(gt_points).shape[0]
                    x_sim = x_sim.reshape(num_gt, num_clusters, 7, x_sim.shape[-1])    # TODO!!!!
                    x_sim = x_sim.mean(2)
                    
                    # similaritys = F.cosine_similarity(x_sim[0].unsqueeze(0), x_sim[0].unsqueeze(1), dim=2).detach()
                    
                    similaritys = F.cosine_similarity(x_sim.reshape(-1, x_sim.shape[-1]).unsqueeze(0), self.semantic_proto.weight.unsqueeze(1), dim=2).transpose(0,1).reshape(-1, num_clusters, self.num_classes+1).detach()  # NOTE: 5 is num-cluster
                    
                    # similaritys = F.cosine_similarity(x_sim.reshape(-1, x_sim.shape[-1]).unsqueeze(0), torch.zeros((self.num_classes+1, 1, 1024), device=x_sim.device), dim=2).transpose(0,1).reshape(-1, 5, self.num_classes+1).detach()  # NOTE: 5 is num-cluster
                    
                    
                    # similaritys = []
                    # for i in range(len(x_sim)):
                    #     label = gt_labels[i]
                    #     meta = self.semantic_proto.weight[label].unsqueeze(0)
                    #     similarity = F.cosine_similarity(x_sim[i].unsqueeze(0), self.semantic_proto.weight.unsqueeze(1), dim=2).transpose(0,1)
                    #     # print(similarity, similarity.shape)
                    #     similaritys.append(similarity)
                    # self.similaritys = torch.stack(similaritys).detach()
                    # print(similaritys.shape)
                x_cls = x
                x_ins = x
                x_reg = x


        elif stage >= 1:
            if self.num_ref_fcs > 0:
                x_ref = x
                if self.with_avg_pool:
                    x = self.avg_pool(x)
                x_ref = x_ref.flatten(1)
                for fc in self.ref_fcs:
                    x_ref = self.relu(fc(x_ref))
                
                x_cls = x_ref
                x_ins = x_ref
                x_reg = x_ref
            
            else:
                if self.with_avg_pool:
                    x = self.avg_pool(x)
                x = x.flatten(1)
                x_sim = x
                for fc in self.shared_fcs:
                    x = self.relu(fc(x))
                # x_sim = x
                # for fc in self.sim_fcs:
                #     x_sim = self.relu(fc(x_sim))
                
                if gt_points is not None:
                    
                    for fc in self.sim_fcs:
                        x_sim = self.relu(fc(x_sim))
                    self.x_update_proto_feats = x_sim.detach()
                    num_gt = torch.cat(gt_points).shape[0]
                    # print(x_sim.shape)
                    
                similaritys = None
                # if unique_id is not None:  # 
                #     for id in unique_id:
                #         self.semantic_proto.weight[id[1]] = self.semantic_proto.weight[id[1]] * 0.99 + x[id[0]] * (1-0.99)  # TODO!!!
                # elif gt_points is None:
                #     self.semantic_proto.weight[-1] = self.semantic_proto.weight[-1] * 0.99 + x.mean(0) * (1-0.99)  
                
                x_cls = x
                x_ins = x
                x_reg = x

        for fc in self.cls_fcs:
            x_cls = self.relu(fc(x_cls))

        for fc in self.ins_fcs:
            x_ins = self.relu(fc(x_ins))
        if self.with_reg:
            for fc in self.reg_fcs:
                if x_reg.dim() > 2:
                    x_reg = x_reg.flatten(1)
            x_reg = self.relu(fc(x_reg))


        cls_score = self.fc_cls[stage](x_cls) if self.with_cls else None
        

        reg_box = self.fc_reg[stage](x_reg) if self.with_reg else None
        
        return cls_score, similaritys, reg_box
    
        
        
    @force_fp32(apply_to=('cls_score', 'ins_score', 'neg_cls_score', 'reg_box'))
    def loss_mil(self,
                 stage,
                 cls_score,
                 ins_score,
                 proposals_valid_list,
                 neg_cls_score,
                 neg_weights,
                 reg_box,
                 labels,
                 gt_boxes,
                 label_weights,
                 retrain_weights,
                 reduction_override=None):
        losses = dict()
        from mmdet.models.losses.utils import weight_reduce_loss
        if stage < 1:
            if cls_score is not None:
                avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
                if cls_score.numel() > 0:
                    label_valid = proposals_valid_list
                    cls_score = cls_score.view(cls_score.shape[0], -1).softmax(dim=-1).view(cls_score.shape[0], -1, self.num_classes+1)
                    
                    # cls_score = cls_score.softmax(dim=-1)
                    # print(cls_score.shape)
                    # print(self.similaritys.shape)
                    # print(self.similaritys.requires_grad)
                    
                    num_sample = cls_score.shape[0]
                    pos_loss, bag_acc, num_pos = self.loss_mil1(
                        cls_score,
                        ins_score,
                        labels,
                        label_valid,
                        weight=label_weights.unsqueeze(-1), )
                    if isinstance(pos_loss, dict):
                        losses.update(pos_loss)
                    else:
                        losses['loss_instance_mil'] = pos_loss
                    losses['bag_acc'] = bag_acc
                    
                if neg_cls_score is not None:
                    num_neg, num_class = neg_cls_score.shape
                    neg_cls_score = neg_cls_score.softmax(dim=-1)
                    neg_labels = torch.full((num_neg, num_class), 0, dtype=torch.float32).to(neg_cls_score.device)
                    neg_labels[:, -1] = 1
                    loss_weights = 1.0
                    neg_valid = neg_weights.reshape(num_neg, -1)
                    # assert num_sample != 0
                    neg_cls_score = neg_cls_score.clamp(0, 1)
                    neg_loss = F.binary_cross_entropy(neg_cls_score, neg_labels, neg_valid.float(), reduction="none")
                    # neg_valid.float())
                    neg_loss = loss_weights *  label_weights.float().mean()*weight_reduce_loss(neg_loss, None, avg_factor=neg_cls_score.shape[0])
                    losses.update({"neg_loss": neg_loss})
        elif stage >= 1:
            mode = 'mil-2'
            # if stage == self.num_stages - 1:  # Note###############################
            #     mode = 're_train'
            
            if mode == 'mil-2':
                if cls_score is not None:
                    if cls_score.numel() > 0:
                        # label_valid = cls_score.new_full(
                        #     (cls_score.shape[0], cls_score.shape[1], 1), 1, dtype=torch.long)
                        label_valid = proposals_valid_list
                        cls_score = cls_score.sigmoid()  # origin 7/25
                        cls_score_label = cls_score[torch.arange(len(cls_score)), :, labels]
                        max_values, max_indices = torch.max(cls_score_label, dim=1)
                        max_indices = torch.tensor([max_indices[i] + i * cls_score.shape[1] for i in range(len(max_indices))], device=max_indices.device)
                        max_indices = torch.hstack((max_indices.reshape(-1, 1), labels.reshape(-1, 1)))
                        selected_indices = max_indices[max_values > 0.1]  # thr = 0.7 
                        # print(self.semantic_proto.weight)
                        
                        
                        if len(selected_indices) > 0:
                            for id in selected_indices:
                                self.semantic_proto.weight[id[1]] = self.semantic_proto.weight[id[1]] * 0.99 + self.x_update_proto_feats[id[0]] * (1-0.99)  # TODO!!!
                        
                        
                        # print(self.semantic_proto.weight)
                        # print(self.semantic_proto.weight)    
                        # print(selected_indices)
                        # # print(max_indices)
                        # exit()

                        # cls_score = cls_score.view(cls_score.shape[0], -1).softmax(dim=-2).view(cls_score.shape[0], -1, self.num_classes)
                        # print(cls_score[0].max())
                        num_sample = cls_score.shape[0]
                        # cls_score=cls_score.sigmoid()
                        pos_loss, bag_acc, num_pos = self.loss_mil2(
                            cls_score,
                            ins_score,
                            labels,
                            label_valid,
                            weight=label_weights.unsqueeze(-1), )
                        if isinstance(pos_loss, dict):
                            losses.update(pos_loss)
                        else:
                            losses['loss_instance_mil'] = pos_loss
                        losses['bag_acc'] = bag_acc


                if self.with_loss_pseudo and stage == self.num_stages - 1:
                    cls_score_pseudo = cls_score[:, 0]
                    label_valid = cls_score.new_full(
                        (cls_score.shape[0], 1), 1, dtype=torch.long)
                    labels_ = _expand_onehot_labels(labels, None, cls_score.shape[-1])[0].float()
                    pseudo_loss = self.loss_mil2.gfocal_loss(cls_score_pseudo, labels_, label_valid.float())
                    loss_weights = 0.125
                    pseudo_loss = loss_weights * weight_reduce_loss(pseudo_loss, label_weights, avg_factor=num_sample)
                    losses.update({"pseudo_box_loss": pseudo_loss})
            
            if neg_cls_score is not None:
                num_neg, num_class = neg_cls_score.shape
                neg_cls_score = neg_cls_score.sigmoid()
                neg_labels = torch.full((num_neg, num_class), 0, dtype=torch.float32).to(neg_cls_score.device)
                loss_weights = 0.75
                neg_valid = neg_weights.reshape(num_neg, -1)
                assert num_sample != 0
                neg_loss = self.loss_mil2.gfocal_loss(neg_cls_score, neg_labels, neg_valid.float())
                neg_loss = loss_weights * label_weights.float().mean() * weight_reduce_loss(neg_loss, None,
                                                                                            avg_factor=num_sample)
                losses.update({"neg_loss": neg_loss})
            # #
        return losses
    



    #-----------------------------------------added by guo -----------------------------------
    @force_fp32(apply_to=('cls_score', 'ins_score', 'neg_cls_score', 'reg_box'))
    def loss_img_mil(self,
                 stage,
                 batch_gt,
                 cls_score,
                 ins_score,
                 proposals_valid_list,
                 neg_cls_score,
                 neg_weights,
                 reg_box,
                 labels,
                 gt_boxes,
                 label_weights,
                 retrain_weights,
                 reduction_override=None):
        losses = dict()
        from mmdet.models.losses.utils import weight_reduce_loss

        if stage < 1:
            if cls_score is not None:
                avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
                if cls_score.numel() > 0:
                    label_valid = proposals_valid_list
                    cls_score = cls_score.softmax(dim=-1)
                    num_sample = cls_score.shape[0]
                    
                    # split batch
                    cls_score_batch = torch.split(cls_score, batch_gt['img_split'])
                    ins_score_batch = torch.split(ins_score, batch_gt['img_split'])
                    label_valid_batch = torch.split(label_valid.view(-1, 1), batch_gt['img_split'])
                    labels_batch = torch.split(labels, batch_gt['cluster_split'])

                    pos_loss, bag_acc, num_pos = self.loss_mil1(
                        cls_score_batch,
                        ins_score_batch,
                        labels_batch,
                        label_valid_batch,
                        label_weights.unsqueeze(-1), )
                    if isinstance(pos_loss, dict):
                        losses.update(pos_loss)
                    else:
                        losses['loss_instance_mil'] = pos_loss
                    losses['bag_acc'] = bag_acc
                    base_loss = False
                    if base_loss:
                        num_class = cls_score.shape[-1]
                        base_cls_score = cls_score.reshape(num_sample, -1, 5, num_class)[:, :, 0, :]
                        base_ins_score = ins_score.reshape(num_sample, -1, 5, num_class)[:, :, 0, :]
                        base_label_valid = label_valid.reshape(num_sample, -1, 5, 1)[:, :, 0, :]
                        base_pos_loss, _, _ = self.loss_mil1(
                            base_cls_score,
                            base_ins_score,
                            labels,
                            base_label_valid,
                            label_weights.unsqueeze(-1), )
                        base_weight = 0.5
                        losses['loss_base_mil'] = base_pos_loss * base_weight
                if neg_cls_score is not None:
                    num_neg, num_class = neg_cls_score.shape
                    
                    neg_cls_score = neg_cls_score.softmax(dim=-1)
                    neg_labels = torch.full((num_neg, num_class), 0, dtype=torch.float32).to(neg_cls_score.device)
                    neg_labels[:, -1] = 1
                    loss_weights = 1.0
                    neg_valid = neg_weights.reshape(num_neg, -1)
                    # assert num_sample != 0
                    neg_cls_score = neg_cls_score.clamp(0, 1)
                    neg_loss = F.binary_cross_entropy(neg_cls_score, neg_labels, neg_valid.float(), reduction="none")
                    # neg_valid.float())
                    neg_loss = loss_weights *  label_weights.float().mean()*weight_reduce_loss(neg_loss, None, avg_factor=neg_cls_score.shape[0])
                    losses.update({"neg_loss": neg_loss})
        
        elif stage >= 1:
            mode = 'mil-2'
            # if stage == self.num_stages - 1:
            #     mode = 're_train'
            if mode == 'cluster cam':
                if cls_score is not None:
                    cls_score = cls_score.sigmoid()
                    num_sample = cls_score.shape[0]
                    # label_valid = cls_score.new_full(
                    #     (cls_score.shape[0], 1), 1, dtype=torch.long)
                    label_valid = proposals_valid_list
                    cls_score = cls_score.mean(dim=1)
                    labels_ = _expand_onehot_labels(labels, None, cls_score.shape[-1])[0].float()
                    cluster_loss = self.loss_mil2.gfocal_loss(cls_score, labels_, label_valid.float())
                    loss_weights = 0.25
                    cluster_loss = loss_weights * weight_reduce_loss(cluster_loss, label_weights, avg_factor=num_sample)
                    losses.update({"cluster_loss": cluster_loss})
                    acc = accuracy(cls_score, labels)
                    losses['bag_acc'] = acc
            if mode == 'mil-1':
                if cls_score is not None:
                    avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
                    if cls_score.numel() > 0:
                        label_valid = proposals_valid_list
                        cls_score = cls_score.softmax(dim=-1)
                        num_sample = cls_score.shape[0]
                        pos_loss, bag_acc, num_pos = self.loss_mil1(
                            cls_score,
                            ins_score,
                            labels,
                            label_valid,
                            label_weights.unsqueeze(-1), )
                        if isinstance(pos_loss, dict):
                            losses.update(pos_loss)
                        else:
                            losses['loss_instance_mil'] = pos_loss
                        losses['bag_acc'] = bag_acc

            elif mode == 'mil-2':
                if cls_score is not None:
                    if cls_score.numel() > 0:
                        # label_valid = cls_score.new_full(
                        #     (cls_score.shape[0], cls_score.shape[1], 1), 1, dtype=torch.long)
                        label_valid = proposals_valid_list
                        cls_score = cls_score.sigmoid()
                        num_sample = cls_score.shape[0]
                        # cls_score=cls_score.sigmoid()
                        
                        # split batch
                        cls_score_batch = torch.split(cls_score, batch_gt['img_split'])
                        ins_score_batch = torch.split(ins_score, batch_gt['img_split'])
                        label_valid_batch = torch.split(label_valid.view(-1, 1), batch_gt['img_split'])
                        labels_batch = torch.split(labels, batch_gt['cluster_split'])
                        
                        pos_loss, bag_acc, num_pos = self.loss_mil2(
                            cls_score_batch,
                            ins_score_batch,
                            labels_batch,
                            label_valid_batch,
                            label_weights.unsqueeze(-1), )
                        if isinstance(pos_loss, dict):
                            losses.update(pos_loss)
                        else:
                            losses['loss_instance_mil'] = pos_loss
                        losses['bag_acc'] = bag_acc


                if self.with_loss_pseudo and stage == self.num_stages - 1:
                    cls_score_pseudo = cls_score[:, 0]
                    label_valid = cls_score.new_full(
                        (cls_score.shape[0], 1), 1, dtype=torch.long)
                    labels_ = _expand_onehot_labels(labels, None, cls_score.shape[-1])[0].float()
                    pseudo_loss = self.loss_mil2.gfocal_loss(cls_score_pseudo, labels_, label_valid.float())
                    loss_weights = 0.125
                    pseudo_loss = loss_weights * weight_reduce_loss(pseudo_loss, label_weights, avg_factor=num_sample)
                    losses.update({"pseudo_box_loss": pseudo_loss})
            
            elif mode == 're_train':
                if cls_score is not None:
                    cls_score = cls_score.sigmoid()
                    cls_score_ = cls_score.reshape(-1, cls_score.shape[-1])
                    labels_ = (retrain_weights == 0) * self.num_classes + (retrain_weights > 0) * labels.unsqueeze(-1)
                    labels_ = labels_.reshape(-1)
                    ## *3 because here is 0.25,but neg weight is 0.75
                    label_weights_ = (retrain_weights == 0) * label_weights.mean() * 3 + (
                            retrain_weights > 0) * label_weights.unsqueeze(-1).expand(cls_score.shape[:2])

                    num_sample = retrain_weights.sum()
                    label_valid = cls_score.new_full((cls_score_.shape[0], 1), 1)
                    _labels_ = _expand_onehot_labels(labels_, None, cls_score.shape[-1])[0].float()
                    cluster_loss = self.loss_mil2.gfocal_loss(cls_score_, _labels_, label_weights_.reshape(-1, 1))
                    loss_weights = 0.25
                    cluster_loss = loss_weights * weight_reduce_loss(cluster_loss, None,
                                                                     avg_factor=num_sample)
                    losses.update({"loss_retrain_cls": cluster_loss})
                    acc = accuracy(cls_score_, labels_)
                    losses['bag_acc'] = acc
                if reg_box is not None:
                    gt_boxes = gt_boxes.unsqueeze(1).expand(reg_box.shape)
                    reg_box = reg_box.reshape(-1, 4)

                    gt_boxes = gt_boxes.reshape(-1, 4)
                    # loss = self.loss_bbox()
                    num_reg_pos = retrain_weights.sum()

                    reg_weight = label_weights_ * retrain_weights

                    losses['loss_bbox'] = self.loss_bbox(
                        reg_box,
                        gt_boxes,
                        reg_weight.reshape(-1, 1),
                        avg_factor=((num_reg_pos + 1e-5) ** 2 / num_reg_pos))

            if neg_cls_score is not None:
                num_neg, num_class = neg_cls_score.shape
                neg_cls_score = neg_cls_score.sigmoid()
                neg_labels = torch.full((num_neg, num_class), 0, dtype=torch.float32).to(neg_cls_score.device)
                loss_weights = 0.75
                neg_valid = neg_weights.reshape(num_neg, -1)
                assert num_sample != 0
                neg_loss = self.loss_mil2.gfocal_loss(neg_cls_score, neg_labels, neg_valid.float())
                neg_loss = loss_weights * label_weights.float().mean() * weight_reduce_loss(neg_loss, None,
                                                                                            avg_factor=num_sample)
                losses.update({"neg_loss": neg_loss})
            # #
        return losses

