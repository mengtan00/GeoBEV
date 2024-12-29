# Copyright (c) Phigent Robotics. All rights reserved.
import torch
import numpy as np
from copy import copy

import torch.nn.functional as F
from mmcv.runner import force_fp32
from mmdet.models import DETECTORS
from ...datasets.pipelines.loading import LoadAnnotationsBEVDepth
from mmdet.core import multi_apply
from .bevdet import BEVDepth4D

@DETECTORS.register_module()
class GeoBEV(BEVDepth4D):

    def __init__(self, scale_num=2, bev_paste=True, prev_num=1, 
                 bda_aug_conf=None, **kwargs):
        super(GeoBEV, self).__init__(**kwargs)
        self.scale_num = scale_num
        self.bev_paste = bev_paste
        self.prev_num = prev_num
        self.prev_data = [dict() for i in range(prev_num)]
        self.bda_aug_conf = bda_aug_conf
        if self.bev_paste:
            if bda_aug_conf is None:
                bda_aug_conf = dict(
                    rot_lim=(-22.5, 22.5),
                    scale_lim=(0.95, 1.05),
                    flip_dx_ratio=0.5,
                    flip_dy_ratio=0.5)
            self.loader = LoadAnnotationsBEVDepth(bda_aug_conf, None, is_train=True)

    def prepare_bev_feat(self, img, rot, tran, intrin, post_rot, post_tran,
                         bda, mlp_input, prev_data, bda_paste, frame_id, img_metas=None):
        x, _ = self.image_encoder(img)
        bev_feat, img_preds = self.img_view_transformer(
            [x, rot, tran, intrin, post_rot, post_tran, bda, mlp_input, prev_data, bda_paste], frame_id)
        if self.pre_process:
            bev_feat = self.pre_process_net(bev_feat)[0]
        return bev_feat, img_preds
    
    def prepare_inputs(self, inputs, stereo=False):
        # split the inputs into each frame
        B, N, C, H, W = inputs[0].shape
        N = N // self.num_frame
        imgs = inputs[0].view(B, N, self.num_frame, C, H, W)
        imgs = torch.split(imgs, 1, 2)
        imgs = [t.squeeze(2) for t in imgs]
        sensor2egos, ego2globals, intrins, post_rots, post_trans, bda = \
            inputs[1:7]

        sensor2egos = sensor2egos.view(B, self.num_frame, N, 4, 4)
        ego2globals = ego2globals.view(B, self.num_frame, N, 4, 4)

        # calculate the transformation from sweep sensor to key ego
        keyego2global = ego2globals[:, 0, 0, ...].unsqueeze(1).unsqueeze(1)
        global2keyego = torch.inverse(keyego2global.double())
        sensor2keyegos = \
            global2keyego @ ego2globals.double() @ sensor2egos.double()
        sensor2keyegos = sensor2keyegos.float()

        curr2adjsensor = None
        if stereo:
            sensor2egos_cv, ego2globals_cv = sensor2egos, ego2globals
            sensor2egos_curr = \
                sensor2egos_cv[:, :self.temporal_frame, ...].double()
            ego2globals_curr = \
                ego2globals_cv[:, :self.temporal_frame, ...].double()
            sensor2egos_adj = \
                sensor2egos_cv[:, 1:self.temporal_frame + 1, ...].double()
            ego2globals_adj = \
                ego2globals_cv[:, 1:self.temporal_frame + 1, ...].double()
            curr2adjsensor = \
                torch.inverse(ego2globals_adj @ sensor2egos_adj) \
                @ ego2globals_curr @ sensor2egos_curr
            curr2adjsensor = curr2adjsensor.float()
            curr2adjsensor = torch.split(curr2adjsensor, 1, 1)
            curr2adjsensor = [p.squeeze(1) for p in curr2adjsensor]
            curr2adjsensor.extend([None for _ in range(self.extra_ref_frames)])
            assert len(curr2adjsensor) == self.num_frame

        extra = [
            sensor2keyegos,
            ego2globals,
            intrins.view(B, self.num_frame, N, 3, 3),
            post_rots.view(B, self.num_frame, N, 3, 3),
            post_trans.view(B, self.num_frame, N, 3)
        ]
        extra = [torch.split(t, 1, 1) for t in extra]
        extra = [[p.squeeze(1) for p in t] for t in extra]
        sensor2keyegos, ego2globals, intrins, post_rots, post_trans = extra

        if len(inputs) == 9:
            prev_data = inputs[7]
            bda_paste = inputs[8]
        else:
            prev_data = None
            bda_paste = None
        return imgs, sensor2keyegos, ego2globals, intrins, post_rots, post_trans, \
               bda, curr2adjsensor, prev_data, bda_paste

    def image_encoder(self, img):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        x = list(self.img_backbone(imgs))
        if self.with_img_neck:
            x[self.scale_num - 1] = self.img_neck(x[self.scale_num - 1:])
            if type(x) in [list, tuple]:
                x = x[:self.scale_num]
        for i in range(self.scale_num):
            _, output_dim, ouput_H, output_W = x[i].shape
            x[i] = x[i].view(B, N, output_dim, ouput_H, output_W)
        return x[:self.scale_num][::-1], None

    def extract_img_feat(self, img, img_metas, **kwargs):
        imgs, rots, trans, intrins, post_rots, post_trans, bda, _, prev_data, bda_paste = \
            self.prepare_inputs(img)
        """Extract features of images."""
        bev_feat_list = []
        img_preds_list = []
        key_frame = True  # back propagation for key frame only
        for frame_id, (img, rot, tran, intrin, post_rot, post_tran) in enumerate(zip(
                imgs, rots, trans, intrins, post_rots, post_trans)):
            if key_frame or self.with_prev:
                mlp_input = self.img_view_transformer.get_mlp_input(
                    rots[0], trans[0], intrin, post_rot, post_tran, bda)
                inputs_curr = (img, rot, tran, intrin, post_rot,
                               post_tran, bda, mlp_input, prev_data, bda_paste)
                if key_frame:
                    bev_feat, img_preds = self.prepare_bev_feat(*inputs_curr, frame_id, img_metas)
                else:
                    with torch.no_grad():
                        bev_feat, img_preds = self.prepare_bev_feat(*inputs_curr, frame_id)
            else:
                bev_feat = torch.zeros_like(bev_feat_list[0])
                img_preds = None
            bev_feat_list.append(bev_feat)
            img_preds_list.append(img_preds)
            key_frame = False

        bev_feat = torch.cat(bev_feat_list, dim=1)
        x = self.bev_encoder(bev_feat)
        # x = checkpoint(self.bev_encoder,bev_feat)
        return [x], img_preds_list
    
    def extract_feat(self, points, img, img_metas, **kwargs):
        """Extract features from images and points."""
        img_feats, img_preds_list = self.extract_img_feat(img, img_metas, **kwargs)
        pts_feats = None
        return (img_feats, pts_feats, img_preds_list)
    
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        
        gt_bboxes_inbox = gt_bboxes_3d
        if self.bev_paste:
            gt_bboxes_cur = copy(gt_bboxes_3d)
            gt_labels_cur = copy(gt_labels_3d)
            B = len(gt_labels_3d)
            bda_paste = []
            gt_bboxes_paste = []
            gt_labels_paste = []
            gt_bboxes_inbox = []
            # adding the samples in previous step to the current samples for BEV-Paste
            # the batch is expanded by prev_num times after BEV-Paste
            for i in range(self.prev_num):
                bda_mat_paste = []
                if self.prev_data[i]:
                    for j in range(B):
                        gt_bboxes_tmp = torch.cat([gt_bboxes_3d[j].tensor.clone(), 
                            self.prev_data[i]['gt_bboxes'][j].tensor.clone()], dim=0)
                        gt_labels_tmp = torch.cat([gt_labels_3d[j].clone(), 
                            self.prev_data[i]['gt_labels'][j].clone()], dim=0)
                        
                        rotate_bda, scale_bda, flip_dx, flip_dy = self.loader.sample_bda_augmentation()
                        gt_bboxes_tmp, bda_rot = self.loader.bev_transform(gt_bboxes_tmp.cpu(), 
                                                                rotate_bda, scale_bda, flip_dx, flip_dy)
                        gt_bboxes_inbox_tmp = gt_bboxes_3d[0].new_box(gt_bboxes_tmp[:len(gt_labels_3d[j])])
                        gt_bboxes_tmp = gt_bboxes_3d[0].new_box(gt_bboxes_tmp.cuda())

                        bda_mat_paste.append(bda_rot.cuda())
                        gt_bboxes_paste.append(gt_bboxes_tmp)
                        gt_labels_paste.append(gt_labels_tmp)
                        if i==0:
                            gt_bboxes_inbox.append(gt_bboxes_inbox_tmp)
                else:
                    for j in range(B):
                        gt_bboxes_tmp = gt_bboxes_3d[j].tensor.clone()
                        gt_labels_tmp = gt_labels_3d[j].clone()
                        
                        rotate_bda, scale_bda, flip_dx, flip_dy = self.loader.sample_bda_augmentation()
                        gt_bboxes_tmp, bda_rot = self.loader.bev_transform(gt_bboxes_tmp.cpu(), 
                                                                rotate_bda, scale_bda, flip_dx, flip_dy)
                        gt_bboxes_tmp = gt_bboxes_3d[0].new_box(gt_bboxes_tmp.cuda())

                        bda_mat_paste.append(bda_rot.cuda())
                        gt_bboxes_paste.append(gt_bboxes_tmp)
                        gt_labels_paste.append(gt_labels_tmp)
                        if i==0:
                            gt_bboxes_inbox.append(gt_bboxes_tmp)
                bda_paste.append(torch.stack(bda_mat_paste))

            img_inputs.append(self.prev_data)
            img_inputs.append(bda_paste)
            gt_bboxes_3d = gt_bboxes_paste
            gt_labels_3d = gt_labels_paste


        img_feats, pts_feats, img_preds_list = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)

        pred_depth = img_preds_list[0]['depths']
        pred_fg = img_preds_list[0]['fgs']
        coor = img_preds_list[0]['coors']
        gt_fg = kwargs['gt_fg']
        gt_depth = kwargs['gt_depth']
        loss_fg, loss_inbox = \
            self.img_view_transformer.get_losses(gt_fg, gt_depth, pred_fg, pred_depth, 
                                                 gt_bboxes_inbox, coor)
        losses = dict(loss_fg=loss_fg, loss_inbox=loss_inbox)

        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore)
        losses.update(losses_pts)

        if self.bev_paste:
            frustum_feats = []
            transforms = []
            for i in range(len(img_preds_list)):
                if img_preds_list[i] is not None:
                    frustum_feats.append(img_preds_list[i]['frustum_feat'].detach())
                    transforms.append(img_preds_list[i]['transform'])
                else:
                    frustum_feats.append(None)
                    transforms.append(None)
            cur_data = {'gt_bboxes': gt_bboxes_cur, 'gt_labels': gt_labels_cur, 
                        'frustum_feats':frustum_feats, 'transforms': transforms}
            self.prev_data.pop()
            self.prev_data.insert(0, cur_data)
        del img_preds_list
        return losses
    
    def forward_test(self,
                     points=None,
                     img_metas=None,
                     img_inputs=None,
                     **kwargs):

        for var, name in [(img_inputs, 'img_inputs'),
                          (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        num_augs = len(img_inputs)
        if num_augs != len(img_metas):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(img_inputs), len(img_metas)))

        if num_augs==1:
            img_inputs = [img_inputs] if img_inputs is None else img_inputs
            points = [points] if points is None else points
            return self.simple_test(points[0], img_metas[0], img_inputs[0],
                                    **kwargs)
        else:
            return self.aug_test(None, img_metas, img_inputs, **kwargs)
    
    def extract_feats(self, points, img_metas, imgs=None):
        """Extract point and image features of multiple samples."""
        if imgs is None:
            imgs = [None] * len(img_metas)
        if points is None:
            points = [None] * len(img_metas)
        img_feats, pts_feats, _= multi_apply(self.extract_feat, points, imgs,
                                           img_metas)
        return img_feats, pts_feats

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        img_feats, pts_feats = self.extract_feats(points, img_metas, imgs)
        bbox_list = dict()
        if img_feats and self.with_pts_bbox:
            pts_bbox = self.aug_test_pts(img_feats, img_metas, rescale)
            bbox_list.update(pts_bbox=pts_bbox)
        return [bbox_list]
