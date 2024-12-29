import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer
from mmcv.runner import BaseModule, force_fp32
from torch.cuda.amp.autocast_mode import autocast
from torch.utils.checkpoint import checkpoint

from mmdet.models.backbones.resnet import BasicBlock
from ..builder import NECKS
from .view_transformer import LSSViewTransformerBEVDepth, Mlp, SELayer, ASPP
from mmcv.ops.points_in_boxes import points_in_boxes_part
from mmdet.models.losses.focal_loss import FocalLoss
import torch.nn.functional as F

class BCEFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=0.25, reduction='mean'):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, pred, target, weight=None, use_sigmoid=True):
        if use_sigmoid:
            pred_sigmoid = pred.sigmoid()
        else:
            pred_sigmoid = pred
        target = target.type_as(pred)
        pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
        focal_weight = (self.alpha * target + (1 - self.alpha) * (1 - target)) * pt.pow(self.gamma)
        if weight is not None:
            focal_weight = focal_weight * weight
        loss = F.binary_cross_entropy_with_logits(
            pred, target, reduction='none') * focal_weight
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss

class DepthNet(nn.Module):

    def __init__(self,
                 in_channels,
                 mid_channels,
                 context_channels,
                 depth_channels,
                 use_dcn=True,
                 use_aspp=True,
                 with_cp=False,
                 stereo=False,
                 bias=0.0,
                 aspp_mid_channels=-1):
        super(DepthNet, self).__init__()
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(
                in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.context_conv = nn.Conv2d(
            mid_channels, context_channels, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm1d(27)
        self.depth_mlp = Mlp(27, mid_channels, mid_channels)
        self.depth_se = SELayer(mid_channels)  # NOTE: add camera-aware
        self.context_mlp = Mlp(27, mid_channels, mid_channels)
        self.context_se = SELayer(mid_channels)  # NOTE: add camera-aware
        depth_conv_input_channels = mid_channels
        downsample = None

        depth_conv_list = [BasicBlock(depth_conv_input_channels, mid_channels,
                                      downsample=downsample),
                           BasicBlock(mid_channels, mid_channels),
                           BasicBlock(mid_channels, mid_channels)]
        if use_aspp:
            if aspp_mid_channels<0:
                aspp_mid_channels = mid_channels
            depth_conv_list.append(ASPP(mid_channels, aspp_mid_channels))
        if use_dcn:
            depth_conv_list.append(
                build_conv_layer(
                    cfg=dict(
                        type='DCN',
                        in_channels=mid_channels,
                        out_channels=mid_channels,
                        kernel_size=3,
                        padding=1,
                        groups=4,
                        im2col_step=128,
                    )))
        self.depth_conv = nn.Sequential(*depth_conv_list)
        self.depth_head = nn.Conv2d(mid_channels, depth_channels, kernel_size=1, stride=1, padding=0)
        self.with_cp = with_cp
        self.depth_channels = depth_channels

    def forward(self, x, mlp_input, stereo_metas=None):
        mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1]))
        x = self.reduce_conv(x)
        context_se = self.context_mlp(mlp_input)[..., None, None]
        context = self.context_se(x, context_se)
        context = self.context_conv(context)
        depth_se = self.depth_mlp(mlp_input)[..., None, None]
        depth = self.depth_se(x, depth_se)

        if self.with_cp:
            depth_feat = checkpoint(self.depth_conv, depth)
        else:
            depth_feat = self.depth_conv(depth)
        depth = self.depth_head(depth_feat)
        return depth, context, depth_feat


@NECKS.register_module()
class RCSample(LSSViewTransformerBEVDepth):

    def __init__(self, scale_num, ins_channels, downsamples, depth_loss_cfg, fg_loss_cfg, 
                 loss_depth_weight, loss_fg_weight, keep_threshold=None, with_cp=False, 
                 depthnet_cfg=dict(), **kwargs):
        super(RCSample, self).__init__(**kwargs)
        self.loss_depth_weight = loss_depth_weight
        self.loss_fg_weight = loss_fg_weight
        self.ins_channels = ins_channels
        self.downsamples = downsamples
        self.keep_threshold = keep_threshold
        self.depth_net = DepthNet(self.ins_channels[0], self.ins_channels[0],
                                    self.out_channels, self.D + 1, with_cp=with_cp, **depthnet_cfg)
        self.scale_num = scale_num
        self.context_convs = nn.ModuleList([nn.Conv2d(self.ins_channels[i], self.out_channels, kernel_size=1) 
                         for i in range(1, self.scale_num)])
        self.depth_upconvs = nn.ModuleList([nn.Conv2d(self.ins_channels[i-1], self.ins_channels[i], kernel_size=1) 
                       for i in range(1, self.scale_num)])
        self.depth_convs = nn.ModuleList([nn.Conv2d(self.ins_channels[i], self.D + 1, kernel_size=3, padding=1) 
                       for i in range(1, self.scale_num)])
        self.depth_loss = BCEFocalLoss(**depth_loss_cfg)
        self.fg_loss = BCEFocalLoss(**fg_loss_cfg)
        self.frustums = [self.create_frustum(self.grid_config['depth'], self.input_size, downsample) 
                        for downsample in self.downsamples]
        if self.accelerate:
            self.pre_bev_coor = None

    def create_grid_infos(self, x, y, z, **kwargs):
        self.grid_lower_bound = torch.Tensor([cfg[0] for cfg in [x, y, z]])
        self.grid_upper_bound = torch.Tensor([cfg[1] for cfg in [x, y, z]])
        self.grid_interval = torch.Tensor([cfg[2] for cfg in [x, y, z]])
        self.grid_size = torch.Tensor([(cfg[1] - cfg[0]) / cfg[2]
                                       for cfg in [x, y, z]])
        x_coor = torch.tensor(list(range(self.grid_size[0].int()))) * x[2] + x[0] + 0.5 * x[2]
        y_coor = torch.tensor(list(range(self.grid_size[0].int()))) * y[2] + y[0] + 0.5 * y[2]
        y_coor, x_coor = torch.meshgrid(x_coor, y_coor)
        bev_coor = torch.stack([x_coor, y_coor, torch.zeros_like(x_coor)], dim=-1)
        self.bev_coor = bev_coor

    @force_fp32()
    def get_sample_coor(self, coor, sensor2ego, ego2global, cam2imgs, post_rots, post_trans, bda, bda_paste=None):
        # calculate the projected BEV coordinates
        B, N, _, _ = sensor2ego.shape
        coor = coor.unsqueeze(-1)
        if bda_paste is not None:
            coor = torch.inverse(bda_paste).view(B, 1, 1, 3, 3).matmul(coor)
        coor = torch.inverse(bda).view(B, 1, 1, 3, 3).matmul(coor)
        coor = coor.unsqueeze(1).expand(-1, N, -1, -1, -1, -1)
        coor = coor - sensor2ego[:,:,:3, 3].view(B, N, 1, 1, 3, 1)
        coor = cam2imgs.matmul(torch.inverse(sensor2ego[:,:,:3,:3])).view(B, N, 1, 1, 3, 3).matmul(coor)
        coor[..., :2, :] /= coor[..., 2:3, :]
        coor = post_rots.view(B, N, 1, 1, 3, 3).matmul(coor)
        coor = coor + post_trans.view(B, N, 1, 1, 3, 1)
        coor = coor.squeeze(-1)
        coor[..., :2] /= self.downsamples[-1]
        coor[..., 2] = (coor[..., 2] - self.grid_config['depth'][0]) / self.grid_config['depth'][-1]
        coor = torch.stack([coor[..., 0], coor[..., 2]], dim=-1)
        return coor

    @autocast(False)
    def get_lidar_coors(self, sensor2ego, ego2global, cam2imgs, post_rots, post_trans,
                       bda, bda_paste=None):
        B, N, _, _ = sensor2ego.shape

        coors = []
        for i in range(len(self.frustums)):
            # post-transformation
            # B x N x D x H x W x 3
            points = self.frustums[i].to(sensor2ego) - post_trans.view(B, N, 1, 1, 1, 3)
            points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3)\
                .matmul(points.unsqueeze(-1))

            # cam_to_ego
            points = torch.cat(
                (points[..., :2, :] * points[..., 2:3, :], points[..., 2:3, :]), 5)
            combine = sensor2ego[:,:,:3,:3].matmul(torch.inverse(cam2imgs))
            points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
            points += sensor2ego[:,:,:3, 3].view(B, N, 1, 1, 1, 3)
            points = bda.view(B, 1, 1, 1, 1, 3,
                            3).matmul(points.unsqueeze(-1)).squeeze(-1)
            if bda_paste is not None:
                points = bda_paste[0].view(B, 1, 1, 1, 1, 3,
                                3).matmul(points.unsqueeze(-1)).squeeze(-1)
            coors.append(points)
        return coors

    def rotate_points_along_z(self, points, angle):
        cosa = torch.cos(angle)
        sina = torch.sin(angle)
        zeros = angle.new_zeros(points.shape[0])
        ones = angle.new_ones(points.shape[0])
        rot_matrix = torch.stack((
            cosa,  sina, zeros,
            -sina, cosa, zeros,
            zeros, zeros, ones
        ), dim=1).view(-1, 3, 3).float()
        points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
        points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
        return points_rot

    def get_centerness(self, coor, inbox_label, boxes, normalize=True):
        pos_mask = inbox_label >= 0
        pos_coor = coor[pos_mask]
        pos_inbox_label = inbox_label[pos_mask].long()
        pos_boxes = boxes[0][pos_inbox_label]
        pos_boxes[:,2] += pos_boxes[:, 5] / 2
        xyz_coords = pos_coor.clone().detach()
        offset_xyz = xyz_coords[:, 0:3] - pos_boxes[:, 0:3]
        offset_xyz_canical = self.rotate_points_along_z(offset_xyz.unsqueeze(dim=1), -pos_boxes[:, 6]).squeeze(dim=1)
        template = pos_boxes.new_tensor(([1, 1, 1], [-1, -1, -1])) / 2
        margin = pos_boxes[:, None, 3:6].repeat(1, 2, 1) * template[None, :, :]
        distance = margin - offset_xyz_canical[:, None, :].repeat(1, 2, 1)
        distance[:, 1, :] = -1 * distance[:, 1, :]
        distance_min = torch.where(distance[:, 0, :] < distance[:, 1, :], distance[:, 0, :], distance[:, 1, :])
        distance_max = torch.where(distance[:, 0, :] > distance[:, 1, :], distance[:, 0, :], distance[:, 1, :])
        centerness = distance_min / distance_max
        centerness = centerness[:, 0] * centerness[:, 1] * centerness[:, 2]
        centerness = torch.clamp(centerness, min=1e-6)
        centerness = torch.pow(centerness, 1/3)
        if normalize:
            centerness = centerness / centerness.sum() * centerness.numel()
        return centerness, pos_mask

    def get_downsampled_labels(self, gt_fg, gt_depths, downsample, gt_bboxes_3d, coor):
        B, N, H, W = gt_fg.shape
        gt_fg = gt_fg.view( B, N, H // downsample, downsample,
                            W // downsample, downsample)
        gt_fg = gt_fg.permute(0, 1, 2, 4, 3, 5).contiguous()
        gt_fg = gt_fg.view(B, N, H // downsample, W // downsample, downsample * downsample)
        fg_label = torch.max(gt_fg, dim=-1).values

        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(B * N, H // downsample,
                                   downsample, W // downsample,
                                   downsample, 1)
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(-1, downsample * downsample)
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // downsample,
                                   W // downsample)

        gt_depths = (
            gt_depths -
            (self.grid_config['depth'][0] -
             self.grid_config['depth'][2])) / self.grid_config['depth'][2]
        gt_depths = torch.where((gt_depths < self.D + 1) & (gt_depths >= 0.0),
                                gt_depths, torch.zeros_like(gt_depths)).long()
        gt_depths_tmp = torch.tensor(range(self.D + 1)).to(gt_depths).view(1, 1, 1, self.D + 1)\
                            .expand(gt_depths.shape[0], gt_depths.shape[1], gt_depths.shape[2], self.D+1)
        ignore_mask = (gt_depths_tmp > gt_depths.unsqueeze(-1)) & (gt_depths.unsqueeze(-1) > 0)
        gt_depths_tmp = F.one_hot(gt_depths.long(), num_classes=self.D + 1)
        gt_depths_tmp[ignore_mask] = -1
        gt_depths = gt_depths_tmp.view(-1, self.D + 1)[:, 1:].float()

        B, N, D, H, W, dim = coor.shape
        coor = coor.permute(0, 1, 3, 4, 2, 5).contiguous()
        inbox_label = coor.new_zeros(coor.shape[:5]) - 1
        inbox_label = inbox_label.view(B, -1)
        centerness = inbox_label.clone() + 2
        for batch_idx in range(B):
            coor_tmp = coor[batch_idx].view(-1, dim).unsqueeze(0)
            if len(coor_tmp) == 0: continue
            gt_bboxes_tmp = gt_bboxes_3d[batch_idx].tensor[:, :7].cuda()
            distance = gt_bboxes_tmp[:,0].pow(2) + gt_bboxes_tmp[:,1].pow(2)
            sorted_idx = torch.sort(distance, descending=True)[1]
            gt_bboxes_tmp = gt_bboxes_tmp[sorted_idx].unsqueeze(0)
            point_limit = 2000000
            if coor_tmp.shape[1] <= point_limit:
                inbox_tmp = points_in_boxes_part(coor_tmp, gt_bboxes_tmp).float()
            else:
                split_num = coor_tmp.shape[1] // point_limit
                inbox_tmp = []
                for split_idx in range(split_num):
                    inbox_tmp.append(points_in_boxes_part(
                        coor_tmp[:, split_idx*point_limit:(split_idx+1)*point_limit], gt_bboxes_tmp))
                inbox_tmp.append(points_in_boxes_part(
                    coor_tmp[:, split_num*point_limit:], gt_bboxes_tmp))
                inbox_tmp = torch.hstack(inbox_tmp).float()
            inbox_label[batch_idx] = inbox_tmp.view(-1)
            centerness_pos, pos_mask = self.get_centerness(coor_tmp.squeeze(0), inbox_label[batch_idx], gt_bboxes_tmp)
            centerness[batch_idx, pos_mask] = centerness_pos

        fg_mask = fg_label.view(-1, 1).repeat(1,D)
        inbox_label = inbox_label.view(-1, D)
        centerness = centerness.view(-1, D)
        nearest_idx = inbox_label.max(dim=1, keepdim=True).values
        # ignore the points not in the nearest box
        ignore_mask = (inbox_label >= 0) & (inbox_label < nearest_idx) | (inbox_label >= 0) & (fg_mask == 0)
        inbox_label = (inbox_label >= 0).float()
        inbox_label[ignore_mask] = -1
        # replace the bg depth
        inbox_label_mask = inbox_label.max(dim=1).values > 0
        gt_depths[inbox_label_mask] = inbox_label[inbox_label_mask]
        return fg_label.view(-1), gt_depths, centerness

    @force_fp32()
    def get_losses(self, gt_fg, gt_depths, fg_preds, inbox_preds, gt_bboxes_3d, coors):
        inbox_loss = 0
        fg_loss = 0
        for i in range(self.scale_num):
            fg_label, inbox_label, centerness = self.get_downsampled_labels(gt_fg, gt_depths, self.downsamples[i], gt_bboxes_3d, coors[i])
            inbox_pred = inbox_preds[i].permute(0, 2, 3, 1).contiguous().view(-1, self.D)
            fg_pred = fg_preds[i].contiguous().view(-1)
            valid_mask = inbox_label.max(dim=1).values > 0
            inbox_label = inbox_label[valid_mask]
            inbox_pred = inbox_pred[valid_mask]
            centerness = centerness[valid_mask]
            inbox_label = inbox_label.view(-1).long()
            inbox_pred = inbox_pred.view(-1)
            centerness = centerness.view(-1)
            inbox_not_ignore = inbox_label >= 0
            inbox_label = inbox_label[inbox_not_ignore]
            inbox_pred = inbox_pred[inbox_not_ignore]
            centerness = centerness[inbox_not_ignore]
            with autocast(enabled=False):
                fg_loss += self.fg_loss(fg_pred, fg_label) * self.loss_fg_weight[i]
                inbox_loss += self.depth_loss(inbox_pred, inbox_label, centerness) * self.loss_depth_weight[i]
        return fg_loss, inbox_loss

    @autocast(False)
    def forward(self, input, frame_id):
        (x, sensor2ego, ego2global, cam2imgs, post_rots, post_trans, bda,
         mlp_input, prev_data, bda_paste) = input[:10]

        depths = []
        fgs = []
        B, N, C, H, W = x[0].shape
        depth_input = x[0].view(B * N, C, H, W).float()
        depth, context, depth_feat = self.depth_net(depth_input, mlp_input)
        depths.append(depth[:, :self.D])
        fgs.append(depth[:, self.D:])

        for i in range(1, self.scale_num):
            B, N, C, H, W = x[i].shape
            img_feat = x[i].view(B * N, C, H, W).float()
            context = F.interpolate(context, scale_factor=2, mode='bilinear')
            context += self.context_convs[i - 1](img_feat)
            depth_feat = F.interpolate(depth_feat, scale_factor=2, mode='bilinear')
            depth_feat = self.depth_upconvs[i - 1](depth_feat) + img_feat
            depth = self.depth_convs[i - 1](depth_feat)
            depths.append(depth[:, :self.D])
            fgs.append(depth[:, self.D:])

        depth_weight = torch.where(depths[-1].softmax(1) >= 1 / self.D, depths[-1].sigmoid(), 
                                   torch.zeros_like(depths[-1]))
        fg_weight = fgs[-1].sigmoid()
        if self.keep_threshold is not None:
            fg_weight = fg_weight >= self.keep_threshold
        context_weight = depth_weight * fg_weight
        frustum_feat = torch.matmul(context_weight.permute(0, 3, 1, 2), 
                                    context.permute(0, 3, 2, 1)).permute(0, 3, 2, 1).contiguous()
        _, C, D, W = frustum_feat.shape
        h, w, _ = self.bev_coor.shape

        if prev_data is None:
            bev_coor = self.bev_coor.clone().to(frustum_feat).unsqueeze(0).expand(B, -1, -1, -1)
            bev_coor = self.get_sample_coor(bev_coor, sensor2ego, ego2global, cam2imgs, post_rots, post_trans, bda)
            norm_bev_coor = bev_coor.view(B * N, h, w, 2)
            norm_bev_coor[..., 0] = norm_bev_coor[..., 0] / W * 2 -1
            norm_bev_coor[..., 1] = norm_bev_coor[..., 1] / D * 2 -1
            bev_feat = F.grid_sample(frustum_feat.view(B * N, C, D, W), norm_bev_coor)
            bev_feat = bev_feat.view(B, N, self.out_channels, h, w).sum(1)
        else:
            bev_feat_list = []
            for i in range(len(prev_data)):
                bev_coor = self.bev_coor.clone().to(frustum_feat).unsqueeze(0).expand(B, -1, -1, -1)
                bev_coor = self.get_sample_coor(bev_coor, sensor2ego, ego2global, cam2imgs, post_rots, post_trans, 
                                                bda, bda_paste[i])
                norm_bev_coor = bev_coor.view(B * N, h, w, 2)
                norm_bev_coor[..., 0] = norm_bev_coor[..., 0] / W * 2 -1
                norm_bev_coor[..., 1] = norm_bev_coor[..., 1] / D * 2 -1
                bev_feat = F.grid_sample(frustum_feat.view(B * N, C, D, W), norm_bev_coor)\
                                        .view(B, N, self.out_channels, h, w).sum(1)

                if prev_data[i]:
                    frustum_feat_prev = prev_data[i]['frustum_feats'][frame_id].cuda()
                    transform_prev = prev_data[i]['transforms'][frame_id]
                    if frustum_feat_prev is not None:
                        bev_coor = self.bev_coor.clone().to(frustum_feat).unsqueeze(0).expand(B, -1, -1, -1)
                        bev_coor_prev = self.get_sample_coor(bev_coor, *transform_prev, bda_paste[i])
                        norm_bev_coor_prev = bev_coor_prev.view(B * N, h, w, 2)
                        norm_bev_coor_prev[..., 0] = norm_bev_coor_prev[..., 0] / W * 2 -1
                        norm_bev_coor_prev[..., 1] = norm_bev_coor_prev[..., 1] / D * 2 -1
                        bev_feat += F.grid_sample(frustum_feat_prev.view(B * N, C, D, W), norm_bev_coor_prev)\
                                                .view(B, N, self.out_channels, h, w).sum(1)
                bev_feat_list.append(bev_feat)
            bev_feat = torch.cat(bev_feat_list, dim=0)
        coors = None

        if bev_feat.requires_grad:
            coors = self.get_lidar_coors(*input[1:7], bda_paste=bda_paste)
        
        curr_data = dict(depths=depths, fgs=fgs, coors=coors, frustum_feat=frustum_feat, 
                              transform=(sensor2ego, ego2global, cam2imgs, post_rots, post_trans, bda))
        return bev_feat, curr_data
