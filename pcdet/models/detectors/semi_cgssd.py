from .detector3d_template import Detector3DTemplate
from ..model_utils import model_nms_utils
import torch

class SemiCGSSD(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        self.model_type = None

    def set_model_type(self, model_type):
        assert model_type in ['origin', 'teacher', 'student']
        self.model_type = model_type
        self.dense_head.model_type = model_type

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        # origin: (training, return loss) (testing, return final boxes)
        if self.model_type == 'origin':
            if self.training:
                loss, tb_dict, disp_dict = self.get_training_loss()
                ret_dict = {
                    'loss': loss
                }
                return ret_dict, tb_dict, disp_dict
            else:
                pred_dicts, recall_dicts = self.post_processing(batch_dict)
                return pred_dicts, recall_dicts

        # teacher: (testing, return raw boxes)
        elif self.model_type == 'teacher':
            # assert not self.training
            return batch_dict

        # student: (training, return (loss & raw boxes w/ gt_boxes) or raw boxes (w/o gt_boxes) for consistency)
        #          (testing, return final_boxes)
        elif self.model_type == 'student':
            if self.training:
                if 'gt_boxes' in batch_dict: # for (pseudo-)labeled data
                    loss, tb_dict, disp_dict = self.get_training_loss()
                    ret_dict = {
                        'loss': loss
                    }
                    return batch_dict, ret_dict, tb_dict, disp_dict
                else:
                    return batch_dict
            else:
                pred_dicts, recall_dicts = self.post_processing(batch_dict)
                return pred_dicts, recall_dicts

        else:
            raise Exception('Unsupprted model type')

    def get_training_loss(self):
        disp_dict = {}

        # CHK MARK, try update aux loss too. Before, only dense head loss
        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_cgam, tb_dict_cgam = self.dense_head.aux_module.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict,
            **tb_dict_cgam,
        }

        loss = loss_rpn + loss_cgam
        return loss, tb_dict, disp_dict

    ## post process with corner rectify
    def post_processing(self, batch_dict):
        """ We add corner rectification to original post processing at 
        `DetectorTemplate.post_processing`. Codes are simplified here
        Args:
            batch_dict:
                batch_size: B
                batch_cls_preds: (B, num_boxes, num_classes)
                batch_box_preds: (B, num_boxes, 7+C)
        Returns:

        """
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        recall_dict = {}
        pred_dicts = []
        for index in range(batch_size):
            batch_mask = index
            box_preds = batch_dict['batch_box_preds'][batch_mask]
            cls_preds = batch_dict['batch_cls_preds'][batch_mask]
            cls_preds = torch.sigmoid(cls_preds)    # scores are not normalized

            cls_preds, label_preds = torch.max(cls_preds, dim=-1)
            label_preds = label_preds + 1           # consider bg class
            # calculate mean scores of 4 corners
            aux_module = self.dense_head.aux_module
            batch_corner_preds = aux_module.forward_ret_dict['pred_dicts'][0]
            batch_corner_hm = batch_corner_preds['hm']
            corner_hm = batch_corner_hm[batch_mask]
            corner_scores = aux_module.get_corner_scores(
                            corner_hm=corner_hm, boxes=box_preds)

            # do score thresh advance
            score_thresh = post_process_cfg.SCORE_THRESH
            score_mask = cls_preds > score_thresh
            corner_scores = corner_scores[score_mask]
            label_preds = label_preds[score_mask]
            cls_preds = cls_preds[score_mask]   # (M,)
            box_preds = box_preds[score_mask]   # (M, 7+C)

            # rectify
            rectifier = post_process_cfg.get('RECTIFIER', 0.3)
            cls_preds = torch.pow(cls_preds, 1 - rectifier) \
                      * torch.pow(corner_scores, rectifier)
            
            # nms
            selected, selected_scores = model_nms_utils.class_agnostic_nms(
                box_scores=cls_preds, box_preds=box_preds,
                nms_config=post_process_cfg.NMS_CONFIG,
            )

            final_scores = selected_scores
            final_labels = label_preds[selected]
            final_boxes = box_preds[selected]

            recall_dict = self.generate_recall_record(
                box_preds=final_boxes,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

            record_dict = {
                'pred_boxes': final_boxes,
                'pred_scores': final_scores,
                'pred_labels': final_labels
            }
            pred_dicts.append(record_dict)

        return pred_dicts, recall_dict