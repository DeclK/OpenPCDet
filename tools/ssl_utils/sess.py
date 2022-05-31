import torch
import torch.nn.functional as F
import numpy as np
from .semi_utils import filter_by_score_interval, reverse_transform, load_data_to_gpu, filter_boxes
from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu

def get_consistency_loss(teacher_boxes, student_boxes):
    center_losses, size_losses, cls_losses = [], [], []
    batch_normalizer = 0
    for teacher_box, student_box in zip(teacher_boxes, student_boxes):  # for each batch
        teacher_box_preds = teacher_box['pred_boxes'].detach_()
        student_box_preds = student_box['pred_boxes']
        Nt, Ns = teacher_box_preds.shape[0], student_box_preds.shape[0]
        if Nt == 0 or Ns == 0:
            batch_normalizer += 1
            continue
        teacher_cls_preds = teacher_box['pred_scores'].detach_().view(Nt, -1)    # CHK MARK, modify pred_cls_preds
        student_cls_preds = student_box['pred_scores'].view(Ns, -1)     # view to be compatible with 1 class

        teacher_centers, teacher_sizes, teacher_rot = teacher_box_preds[:, :3], teacher_box_preds[:, 3:6], teacher_box_preds[:, [6]]
        student_centers, student_sizes, student_rot = student_box_preds[:, :3], student_box_preds[:, 3:6], student_box_preds[:, [6]]

        with torch.no_grad():
            teacher_class = teacher_box['pred_labels'].view(Nt, -1) # [Nt, 1]
            student_class = student_box['pred_labels'].view(Ns, -1) # [Ns, 1]
            not_same_class = (teacher_class != student_class.T).float() # [Nt, Ns]
            MAX_DISTANCE = 1000000
            dist = teacher_centers[:, None, :] - student_centers[None, :, :] # [Nt, Ns, 3]
            dist = (dist ** 2).sum(-1) # [Nt, Ns]
            dist += not_same_class * MAX_DISTANCE # penalty on different classes
            student_dist_of_teacher, student_index_of_teacher = dist.min(1) # [Nt]
            teacher_dist_of_student, teacher_index_of_student = dist.min(0) # [Ns]
            # different from standard sess, we only consider distance<1m as matching
            MATCHED_DISTANCE = 1
            matched_teacher_mask = (teacher_dist_of_student < MATCHED_DISTANCE).float().unsqueeze(-1) # [Ns, 1]
            matched_student_mask = (student_dist_of_teacher < MATCHED_DISTANCE).float().unsqueeze(-1) # [Nt, 1]

        matched_teacher_centers = teacher_centers[teacher_index_of_student] # [Ns, :]
        matched_student_centers = student_centers[student_index_of_teacher] # [Nt, :]

        matched_student_sizes = student_sizes[student_index_of_teacher] # [Nt, :]
        matched_student_cls_preds = student_cls_preds[student_index_of_teacher] # [Nt, :]

        center_loss = (((student_centers - matched_teacher_centers) * matched_teacher_mask).abs().sum()
                       + ((teacher_centers - matched_student_centers) * matched_student_mask).abs().sum()) \
                      / (Nt + Ns)
        size_loss = F.mse_loss(matched_student_sizes, teacher_sizes, reduction='none')
        size_loss = (size_loss * matched_student_mask).sum() / Nt

        # kl_div is not feasible, since we use sigmoid instead of softmax for class prediction
        # cls_loss = F.kl_div(matched_student_cls_preds.log(), teacher_cls_preds, reduction='none')
        cls_loss = F.mse_loss(matched_student_cls_preds, teacher_cls_preds, reduction='none') # use mse loss instead
        cls_loss = (cls_loss * matched_student_mask).sum() / Nt

        center_losses.append(center_loss)
        size_losses.append(size_loss)
        cls_losses.append(cls_loss)
        batch_normalizer += 1

    return sum(center_losses)/batch_normalizer, sum(size_losses)/batch_normalizer, sum(cls_losses)/batch_normalizer

def sigmoid_rampup(current, rampup_start, rampup_end):
    assert rampup_start <= rampup_end
    if current < rampup_start:
        return 0
    elif (current >= rampup_start) and (current < rampup_end):
        rampup_length = max(rampup_end, 0) - max(rampup_start, 0)
        if rampup_length == 0: # no rampup
            return 1
        else:
            phase = 1.0 - (current - max(rampup_start, 0)) / rampup_length
            return float(np.exp(-5.0 * phase * phase))
    elif current >= rampup_end:
        return 1
    else:
        raise Exception('Impossible condition for sigmoid rampup')

def sess(teacher_model, student_model,
         ld_teacher_batch_dict, ld_student_batch_dict,
         ud_teacher_batch_dict, ud_student_batch_dict,
         cfgs, epoch_id, dist
        ):
    use_unlabel = False if ud_teacher_batch_dict is None else True  # CHK MARK
    load_data_to_gpu(ld_teacher_batch_dict)
    load_data_to_gpu(ld_student_batch_dict)
    if use_unlabel:
        load_data_to_gpu(ud_teacher_batch_dict)
        load_data_to_gpu(ud_student_batch_dict)

    # get loss for labeled data
    if not dist:
        ld_teacher_batch_dict = teacher_model(ld_teacher_batch_dict)
        ud_teacher_batch_dict = teacher_model(ud_teacher_batch_dict) if use_unlabel else None
        ld_student_batch_dict, ret_dict, tb_dict, disp_dict = student_model(ld_student_batch_dict)
        ud_student_batch_dict = student_model(ud_student_batch_dict) if use_unlabel else None
    else:
        ld_teacher_batch_dict, ud_teacher_batch_dict = teacher_model(ld_teacher_batch_dict, ud_teacher_batch_dict)
        (ld_student_batch_dict, ret_dict, tb_dict, disp_dict), (ud_student_batch_dict) = student_model(ld_student_batch_dict, ud_student_batch_dict)

    sup_loss = ret_dict['loss'].mean()

    if cfgs.get('FILTER_BOXES', True): # CHK MARK, compatible with centerpoint
        ld_teacher_boxes = filter_boxes(ld_teacher_batch_dict, cfgs)
        ld_student_boxes = filter_boxes(ld_student_batch_dict, cfgs)
    else:
        ld_teacher_boxes = ld_teacher_batch_dict['final_box_dicts']
        ld_student_boxes = ld_student_batch_dict['final_box_dicts']
    if cfgs.get('FILTER_BY_SCORE_INTERVAL', False):
        score_interval = cfgs.SCORE_INTERVAL
        ld_teacher_boxes = filter_by_score_interval(ld_teacher_boxes, score_interval)
        ld_student_boxes = filter_by_score_interval(ld_student_boxes, score_interval)
    ld_teacher_boxes = reverse_transform(ld_teacher_boxes, ld_teacher_batch_dict, ld_student_batch_dict)
    ld_center_loss, ld_size_loss, ld_cls_loss = get_consistency_loss(ld_teacher_boxes, ld_student_boxes)

    if use_unlabel:
        if cfgs.get('FILTER_BOXES', True):
            ud_teacher_boxes = filter_boxes(ud_teacher_batch_dict, cfgs)
            ud_student_boxes = filter_boxes(ud_student_batch_dict, cfgs)
        else:
            ud_teacher_boxes = ud_teacher_batch_dict['final_box_dicts']
            ud_student_boxes = ud_student_batch_dict['final_box_dicts']
        if cfgs.get('FILTER_BY_SCORE_INTERVAL', False):
            score_interval = cfgs.SCORE_INTERVAL
            ud_teacher_boxes = filter_by_score_interval(ud_teacher_boxes, score_interval)
            ud_student_boxes = filter_by_score_interval(ud_student_boxes, score_interval)
        ud_teacher_boxes = reverse_transform(ud_teacher_boxes, ud_teacher_batch_dict, ud_student_batch_dict)
        ud_center_loss, ud_size_loss, ud_cls_loss = get_consistency_loss(ud_teacher_boxes, ud_student_boxes)
    else:
        ud_center_loss, ud_size_loss, ud_cls_loss = 0, 0, 0

    consistency_loss = (ld_center_loss + ud_center_loss) * cfgs.CENTER_WEIGHT \
                       + (ld_size_loss + ud_size_loss) * cfgs.SIZE_WEIGHT \
                       + (ld_cls_loss + ud_cls_loss) * cfgs.CLASS_WEIGHT
    consistency_weight = cfgs.CONSISTENCY_WEIGHT * sigmoid_rampup(epoch_id, cfgs.TEACHER.EMA_EPOCH[0], cfgs.TEACHER.EMA_EPOCH[1])

    loss = sup_loss + consistency_weight * consistency_loss
    tb_dict['consistency_loss'] = consistency_loss.item()
    return loss, tb_dict, disp_dict