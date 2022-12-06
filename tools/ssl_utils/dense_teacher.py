import torch
import torch.nn.functional as F
import numpy as np
from .semi_utils import filter_by_score_interval, reverse_transform, load_data_to_gpu, filter_boxes
from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu

# DenseTeacher
def get_distill_loss(student_logits, teacher_logits, ratio=0.01):
    """ Calculate QualityFocalLoss between teacher & student in DenseTeacher
    Params:
        - student_logits & teacher_logits: (B, M, C)
        - ratio: use top k pixels as foreground signal
    """
    # check dimensionality
    B, M, C = student_logits.size()
    student_logits = student_logits.view(B * M, C)
    teacher_logits = teacher_logits.view(B * M, C)
    with torch.no_grad():
        # Region Selection according to teacher logits
        total_counts = B * M
        count_num = int(total_counts * ratio)       # k
        teacher_probs = teacher_logits.sigmoid()
        max_vals = torch.max(teacher_probs, 1)[0]   # max value along classes
        sorted_vals, sorted_inds = torch.topk(max_vals, total_counts)
        mask = torch.zeros_like(max_vals)
        mask[sorted_inds[:count_num]] = 1           # set mask
        fg_num = sorted_vals[:count_num].sum()
        b_mask= mask > 0
        
    loss_logits = QFLv2(
        student_logits.sigmoid(),
        teacher_logits.sigmoid(),
        weight=mask,
        reduction="sum",
    ) / fg_num

    # loss_deltas = iou_loss(
    #     student_deltas[b_mask],
    #     teacher_deltas[b_mask],
    #     box_mode="ltrb",  
    #     loss_type='giou',
    #     reduction="mean",
    # )

    # loss_quality = F.binary_cross_entropy(
    #     student_quality[b_mask].sigmoid(),
    #     teacher_quality[b_mask].sigmoid(),
    #     reduction='mean'
    # )

    return loss_logits

def QFLv2(pred_sigmoid,
          teacher_sigmoid,
          weight=None,
          beta=2.0,
          reduction='mean'):
    # all goes to 0
    pt = pred_sigmoid
    zerolabel = pt.new_zeros(pt.shape)
    loss = F.binary_cross_entropy(
        pred_sigmoid, zerolabel, reduction='none') * pt.pow(beta)
    pos = weight > 0

    # positive goes to bbox quality
    pt = teacher_sigmoid[pos] - pred_sigmoid[pos]
    loss[pos] = F.binary_cross_entropy(
        pred_sigmoid[pos], teacher_sigmoid[pos], reduction='none') * pt.pow(beta)

    valid = weight >= 0
    if reduction == "mean":
        loss = loss[valid].mean()
    elif reduction == "sum":
        loss = loss[valid].sum()
    return loss

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

def dense_teacher(teacher_model, student_model,
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
    if not dist:    # TODO CHK MARK, optimize codes here, merge dist and single gpu together
        ld_teacher_batch_dict = teacher_model(ld_teacher_batch_dict)
        ud_teacher_batch_dict = teacher_model(ud_teacher_batch_dict) if use_unlabel else None
        ld_student_batch_dict, ret_dict, tb_dict, disp_dict = student_model(ld_student_batch_dict)
        ud_student_batch_dict = student_model(ud_student_batch_dict) if use_unlabel else None
    else:
        ld_teacher_batch_dict, ud_teacher_batch_dict = teacher_model(ld_teacher_batch_dict, ud_teacher_batch_dict)
        (ld_student_batch_dict, ret_dict, tb_dict, disp_dict), (ud_student_batch_dict) \
        = student_model(ld_student_batch_dict, ud_student_batch_dict)

    sup_loss = ret_dict['loss'].mean()

    ld_student_cls = ld_student_batch_dict['batch_cls_preds']   # (B, H * W * num_anchors_per_pixel, num_classes)
    ld_student_box = ld_student_batch_dict['batch_box_preds']   # (B, H * W * num_anchors_per_pixel, 7)
    ld_teacher_cls = ld_teacher_batch_dict['batch_cls_preds']
    ld_teacher_box = ld_teacher_batch_dict['batch_box_preds']

    consistency_loss = get_distill_loss(ld_student_cls, ld_teacher_cls)
    consistency_weight = cfgs.CONSISTENCY_WEIGHT * sigmoid_rampup(epoch_id, cfgs.TEACHER.EMA_EPOCH[0], cfgs.TEACHER.EMA_EPOCH[1])

    loss = sup_loss + consistency_weight * consistency_loss
    tb_dict['consistency_loss'] = consistency_loss if isinstance(consistency_loss, float)\
                                                   else consistency_loss.item()
    return loss, tb_dict, disp_dict