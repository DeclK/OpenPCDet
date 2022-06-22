"""
Evaluation Server
Written by Jiageng Mao
"""

import numpy as np
import numba
import copy

from .iou_utils import rotate_iou_gpu_eval
from .eval_utils import compute_split_parts, overall_filter, distance_filter, overall_distance_filter

iou_threshold_dict = {
    'Car': 0.7,
    'Bus': 0.7,
    'Pedestrian': 0.3,
    'Cyclist': 0.5
}
# iou_threshold_dict = {
#     'Car': 0.1,
#     'Bus': 0.1,
#     'Pedestrian': 0.1,
#     'Cyclist': 0.1
# }

superclass_iou_threshold_dict = {
    'Vehicle': 0.7,
    'Pedestrian': 0.3,
    'Cyclist': 0.5
}


def get_evaluation_results(gt_annos,
                           pred_annos,
                           classes,
                           use_superclass=False,
                           iou_thresholds=None,
                           num_pr_points=50,
                           difficulty_mode='Overall&Distance',
                           ap_with_heading=False,
                           num_parts=100,
                           print_ok=False):

    if iou_thresholds is None:
        if use_superclass:
            iou_thresholds = superclass_iou_threshold_dict
        else:
            iou_thresholds = iou_threshold_dict

    assert len(gt_annos) == len(
        pred_annos), "the number of GT must match predictions"
    assert difficulty_mode in ['Overall&Distance', 'Overall',
                               'Distance'], "difficulty mode is not supported"
    if use_superclass:
        classes = [
            cls_name for cls_name in classes
            if cls_name not in ['Car', 'Bus', 'Truck']
        ]
        classes.insert(0, 'Vehicle')

    num_samples = len(gt_annos)
    split_parts = compute_split_parts(num_samples, num_parts)
    ious = compute_iou3d(gt_annos,
                         pred_annos,
                         split_parts,
                         with_heading=ap_with_heading)
    num_classes = len(classes)
    if difficulty_mode == 'Distance':
        num_difficulties = 3
        difficulty_types = ['0-30m', '30-50m', '50m-inf']
    elif difficulty_mode == 'Overall':
        num_difficulties = 1
        difficulty_types = ['overall']
    elif difficulty_mode == 'Overall&Distance':
        num_difficulties = 4
        difficulty_types = ['overall', '0-30m', '30-50m', '50m-inf']
    else:
        raise NotImplementedError

    precision = np.zeros([num_classes, num_difficulties, num_pr_points + 1])
    recall = np.zeros([num_classes, num_difficulties, num_pr_points + 1])

    for cls_idx, cur_class in enumerate(classes):
        print('cls_idx:', cls_idx, 'cls:', cur_class)
        iou_threshold = iou_thresholds[cur_class]
        for diff_idx in range(num_difficulties):
            ### filter data & determine score thresholds on p-r curve ###
            accum_all_scores, gt_flags, pred_flags = [], [], []
            num_valid_gt = 0
            for sample_idx in range(num_samples):
                gt_anno = gt_annos[sample_idx]
                pred_anno = pred_annos[sample_idx]
                pred_score = pred_anno['scores']
                iou = ious[sample_idx]
                gt_flag, pred_flag = filter_data(gt_anno,
                                                 pred_anno,
                                                 difficulty_mode,
                                                 difficulty_level=diff_idx,
                                                 class_name=cur_class,
                                                 use_superclass=use_superclass)
                gt_flags.append(gt_flag)
                pred_flags.append(pred_flag)
                num_valid_gt += sum(gt_flag == 0)
                accum_scores = accumulate_scores(iou,
                                                 pred_score,
                                                 gt_flag,
                                                 pred_flag,
                                                 iou_threshold=iou_threshold)
                accum_all_scores.append(accum_scores)
            all_scores = np.concatenate(accum_all_scores, axis=0)
            thresholds = get_thresholds(all_scores,
                                        num_valid_gt,
                                        num_pr_points=num_pr_points)

            ### compute tp/fp/fn ###
            confusion_matrix = np.zeros([len(thresholds),
                                         3])  # only record tp/fp/fn
            for sample_idx in range(num_samples):
                pred_score = pred_annos[sample_idx]['scores']
                iou = ious[sample_idx]
                gt_flag, pred_flag = gt_flags[sample_idx], pred_flags[
                    sample_idx]
                for th_idx, score_th in enumerate(thresholds):
                    tp, fp, fn = compute_statistics(
                        iou,
                        pred_score,
                        gt_flag,
                        pred_flag,
                        score_threshold=score_th,
                        iou_threshold=iou_threshold)
                    confusion_matrix[th_idx, 0] += tp
                    confusion_matrix[th_idx, 1] += fp
                    confusion_matrix[th_idx, 2] += fn
            ### draw p-r curve ###
            for th_idx in range(len(thresholds)):
                recall[cls_idx, diff_idx, th_idx] = confusion_matrix[th_idx, 0] / \
                                                    (confusion_matrix[th_idx, 0] + confusion_matrix[th_idx, 2])
                precision[cls_idx, diff_idx, th_idx] = confusion_matrix[th_idx, 0] / \
                                                       (confusion_matrix[th_idx, 0] + confusion_matrix[th_idx, 1])

            for th_idx in range(len(thresholds)):
                precision[cls_idx, diff_idx,
                          th_idx] = np.max(precision[cls_idx, diff_idx,
                                                     th_idx:],
                                           axis=-1)
                recall[cls_idx, diff_idx,
                       th_idx] = np.max(recall[cls_idx, diff_idx, th_idx:],
                                        axis=-1)
    AP = 0
    for i in range(1, precision.shape[-1]):
        AP += precision[..., i]
    AP = AP / num_pr_points * 100
    ret_dict = {}

    ret_str = "\n|AP@%-9s|" % (str(num_pr_points))
    for diff_type in difficulty_types:
        ret_str += '%-12s|' % diff_type
    ret_str += '\n'
    for cls_idx, cur_class in enumerate(classes):
        ret_str += "|%-12s|" % cur_class
        for diff_idx in range(num_difficulties):
            diff_type = difficulty_types[diff_idx]
            key = 'AP_' + cur_class + '/' + diff_type
            ap_score = AP[cls_idx, diff_idx]
            ret_dict[key] = ap_score
            ret_str += "%-12.2f|" % ap_score
        ret_str += "\n"
    mAP = np.mean(AP, axis=0)
    ret_str += "|%-12s|" % 'mAP'
    for diff_idx in range(num_difficulties):
        diff_type = difficulty_types[diff_idx]
        key = 'AP_mean' + '/' + diff_type
        ap_score = mAP[diff_idx]
        ret_dict[key] = ap_score
        ret_str += "%-12.2f|" % ap_score
    ret_str += "\n"

    if print_ok:
        print(ret_str)
    return ret_str, ret_dict


def get_evaluation_pr_results(gt_annos,
                              pred_annos,
                              classes,
                              use_superclass=False,
                              iou_thresholds=None,
                              num_pr_points=50,
                              difficulty_mode='Overall&Distance',
                              ap_with_heading=False,
                              num_parts=100,
                              print_ok=False):

    if iou_thresholds is None:
        if use_superclass:
            iou_thresholds = superclass_iou_threshold_dict
        else:
            iou_thresholds = iou_threshold_dict

    assert len(gt_annos) == len(
        pred_annos), "the number of GT must match predictions"
    assert difficulty_mode in ['Overall&Distance', 'Overall',
                               'Distance'], "difficulty mode is not supported"
    if use_superclass:
        classes = [
            cls_name for cls_name in classes
            if cls_name not in ['Car', 'Bus', 'Truck']
        ]
        classes.insert(0, 'Vehicle')

    num_samples = len(gt_annos)
    split_parts = compute_split_parts(num_samples, num_parts)
    ious = compute_iou3d(gt_annos,
                         pred_annos,
                         split_parts,
                         with_heading=ap_with_heading)
    num_classes = len(classes)
    if difficulty_mode == 'Distance':
        num_difficulties = 3
        difficulty_types = ['0-30m', '30-50m', '50m-inf']
    elif difficulty_mode == 'Overall':
        num_difficulties = 1
        difficulty_types = ['overall']
    elif difficulty_mode == 'Overall&Distance':
        num_difficulties = 4
        difficulty_types = ['overall', '0-30m', '30-50m', '50m-inf']
    else:
        raise NotImplementedError

    precision = np.zeros([num_classes, num_difficulties, num_pr_points + 1])
    recall = np.zeros([num_classes, num_difficulties, num_pr_points + 1])

    score_thres = np.zeros([num_classes, num_difficulties, num_pr_points + 1])
    for cls_idx, cur_class in enumerate(classes):
        print('cls_idx:', cls_idx, 'cls:', cur_class)
        iou_threshold = iou_thresholds[cur_class]
        for diff_idx in range(num_difficulties):
            ### filter data & determine score thresholds on p-r curve ###
            accum_all_scores, gt_flags, pred_flags = [], [], []
            num_valid_gt = 0
            for sample_idx in range(num_samples):
                gt_anno = gt_annos[sample_idx]
                pred_anno = pred_annos[sample_idx]
                pred_score = pred_anno['scores']
                iou = ious[sample_idx]
                gt_flag, pred_flag = filter_data(gt_anno,
                                                 pred_anno,
                                                 difficulty_mode,
                                                 difficulty_level=diff_idx,
                                                 class_name=cur_class,
                                                 use_superclass=use_superclass)
                gt_flags.append(gt_flag)
                pred_flags.append(pred_flag)
                num_valid_gt += sum(gt_flag == 0)
                accum_scores = accumulate_scores(iou,
                                                 pred_score,
                                                 gt_flag,
                                                 pred_flag,
                                                 iou_threshold=iou_threshold)
                accum_all_scores.append(accum_scores)
            all_scores = np.concatenate(accum_all_scores, axis=0)
            thresholds = get_thresholds(all_scores,
                                        num_valid_gt,
                                        num_pr_points=num_pr_points)
            score_thres[cls_idx, diff_idx, :len(thresholds)] = thresholds

            ### compute tp/fp/fn ###
            confusion_matrix = np.zeros([len(thresholds),
                                         3])  # only record tp/fp/fn
            for sample_idx in range(num_samples):
                pred_score = pred_annos[sample_idx]['scores']
                iou = ious[sample_idx]
                gt_flag, pred_flag = gt_flags[sample_idx], pred_flags[
                    sample_idx]
                for th_idx, score_th in enumerate(thresholds):
                    tp, fp, fn = compute_statistics(
                        iou,
                        pred_score,
                        gt_flag,
                        pred_flag,
                        score_threshold=score_th,
                        iou_threshold=iou_threshold)
                    confusion_matrix[th_idx, 0] += tp
                    confusion_matrix[th_idx, 1] += fp
                    confusion_matrix[th_idx, 2] += fn
            ### draw p-r curve ###
            for th_idx in range(len(thresholds)):
                recall[cls_idx, diff_idx, th_idx] = confusion_matrix[th_idx, 0] / \
                                                    (confusion_matrix[th_idx, 0] + confusion_matrix[th_idx, 2])
                precision[cls_idx, diff_idx, th_idx] = confusion_matrix[th_idx, 0] / \
                                                       (confusion_matrix[th_idx, 0] + confusion_matrix[th_idx, 1])

    return score_thres, precision, recall


def get_track_evaluation_results(gt_annos,
                                 pred_annos,
                                 use_superclass=False,
                                 iou_thresholds=None,
                                 ap_with_heading=False,
                                 num_parts=100,
                                 print_ok=False):
    # print('names:', classes)

    if iou_thresholds is None:
        if use_superclass:
            iou_thresholds = superclass_iou_threshold_dict
        else:
            iou_thresholds = iou_threshold_dict
    print("gt_annos:", len(gt_annos), len(pred_annos))
    assert len(gt_annos) == len(
        pred_annos), "the number of GT must match predictions"

    num_samples = len(gt_annos)
    split_parts = compute_split_parts(num_samples, num_parts)
    ious = compute_iou3d(gt_annos,
                         pred_annos,
                         split_parts,
                         with_heading=ap_with_heading)

    id_swift = 0
    id_sum = 0
    tracking_id_list = set()
    trajectories = dict()
    for sample_idx in range(num_samples):
        gt_anno = gt_annos[sample_idx]
        pred_anno = pred_annos[sample_idx]
        #    pred_score = pred_anno['score']
        iou = ious[sample_idx]
        matched_index = np.argmax(iou, axis=1)

        if pred_anno['first_frame']:
            print("trajectories: ", trajectories)
            for traking_id, value_list in trajectories.items():
                obj_info = dict()
                #   print("obj_dict:", value_list)
                for value in value_list:
                    obj_info.setdefault(value[2], 0)
                    obj_info[value[2]] += 1
                id_sum += len(value_list)
                id_swift += len(value_list) - max(
                    [v for k, v in obj_info.items()])
            trajectories = dict()

        for i, index in enumerate(matched_index):
            if index == 0 and iou[i][index] == 0:
                #          gt_anno.setdefault('tracking_id', []).append(None)
                continue
            else:
                tracking_id = pred_anno['tracking_id'][index]
                tracking_id_list.add(tracking_id)
                trajectories.setdefault(tracking_id, []).append([
                    pred_anno['names'][index], gt_anno['names'][i],
                    gt_anno['obj_id'][i], sample_idx
                ])
    #         gt_anno.setdefault('tracking_id', []).append(tracking_id)

    print("id sum: ", id_sum)
    print("id swift:", id_swift, id_swift / id_sum)

    return


# def match_post(iou, matched_index, gt_anno, pred_anno):
#     print("iou:", iou)
#     print("matched_index:", matched_index)
#     print("gt_anno:", gt_anno)
#     print("pred_anno: ", pred_anno)
#     for i, index in enumerate(matched_index):
#         if iou[i][index] == 0:
#             matched_index[i] = 0
#         else:
#             while gt_anno['name'][i] != pred_anno['name'][index] and iou[i][index] != 0:
#                 iou_list = iou[i]
#                 iou[index] = 0
#                 index = np.argmax(iou)


@numba.jit(nopython=True)
def get_thresholds(scores, num_gt, num_pr_points):
    eps = 1e-6
    scores.sort()
    scores = scores[::-1]
    recall_level = 0
    thresholds = []
    for i, score in enumerate(scores):
        l_recall = (i + 1) / num_gt
        if i < (len(scores) - 1):
            r_recall = (i + 2) / num_gt
        else:
            r_recall = l_recall
        if (r_recall + l_recall < 2 * recall_level) and i < (len(scores) - 1):
            continue
        thresholds.append(score)
        recall_level += 1 / num_pr_points
        # avoid numerical errors
        # while r_recall + l_recall >= 2 * recall_level:
        while r_recall + l_recall + eps > 2 * recall_level:
            thresholds.append(score)
            recall_level += 1 / num_pr_points
    return thresholds


@numba.jit(nopython=True)
def accumulate_scores(iou, pred_scores, gt_flag, pred_flag, iou_threshold):
    num_gt = iou.shape[0]
    num_pred = iou.shape[1]
    assigned = np.full(num_pred, False)
    accum_scores = np.zeros(num_gt)
    accum_idx = 0
    for i in range(num_gt):
        if gt_flag[i] == -1:  # not the same class
            continue
        det_idx = -1
        detected_score = -1
        for j in range(num_pred):
            if pred_flag[j] == -1:  # not the same class
                continue
            if assigned[j]:
                continue
            iou_ij = iou[i, j]
            pred_score = pred_scores[j]
            if (iou_ij > iou_threshold) and (pred_score > detected_score):
                det_idx = j
                detected_score = pred_score

        if (detected_score == -1) and (gt_flag[i] == 0):  # false negative
            pass
        elif (detected_score != -1) and (gt_flag[i] == 1
                                         or pred_flag[det_idx] == 1):  # ignore
            assigned[det_idx] = True
        elif detected_score != -1:  # true positive
            accum_scores[accum_idx] = pred_scores[det_idx]
            accum_idx += 1
            assigned[det_idx] = True

    return accum_scores[:accum_idx]


@numba.jit(nopython=True)
def compute_statistics(iou, pred_scores, gt_flag, pred_flag, score_threshold,
                       iou_threshold):
    num_gt = iou.shape[0]
    num_pred = iou.shape[1]
    assigned = np.full(num_pred, False)
    under_threshold = pred_scores < score_threshold

    tp, fp, fn = 0, 0, 0
    for i in range(num_gt):
        if gt_flag[i] == -1:  # different classes
            continue
        det_idx = -1
        detected = False
        best_matched_iou = 0
        gt_assigned_to_ignore = False

        for j in range(num_pred):
            if pred_flag[j] == -1:  # different classes
                continue
            if assigned[j]:  # already assigned to other GT
                continue
            if under_threshold[j]:  # compute only boxes above threshold
                continue
            iou_ij = iou[i, j]
            if (iou_ij > iou_threshold) and (
                    iou_ij > best_matched_iou
                    or gt_assigned_to_ignore) and pred_flag[j] == 0:
                best_matched_iou = iou_ij
                det_idx = j
                detected = True
                gt_assigned_to_ignore = False
            elif (iou_ij >
                  iou_threshold) and (not detected) and pred_flag[j] == 1:
                det_idx = j
                detected = True
                gt_assigned_to_ignore = True

        if (not detected) and gt_flag[i] == 0:  # false negative
            fn += 1
        elif detected and (gt_flag[i] == 1
                           or pred_flag[det_idx] == 1):  # ignore
            assigned[det_idx] = True
        elif detected:  # true positive
            tp += 1
            assigned[det_idx] = True

    for j in range(num_pred):
        if not (assigned[j] or pred_flag[j] == -1 or pred_flag[j] == 1
                or under_threshold[j]):
            fp += 1

    return tp, fp, fn


def filter_data(gt_anno, pred_anno, difficulty_mode, difficulty_level,
                class_name, use_superclass):
    """
    Filter data by class name and difficulty

    Args:
        gt_anno:
        pred_anno:
        difficulty_mode:
        difficulty_level:
        class_name:

    Returns:
        gt_flags/pred_flags:
            1 : same class but ignored with different difficulty levels
            0 : accepted
           -1 : rejected with different classes
    """
    pred_anno['names'] = np.array(pred_anno['names'])
    num_gt = len(gt_anno['names'])
    gt_flag = np.zeros(num_gt, dtype=np.int64)
    if use_superclass:
        if class_name == 'Vehicle':
            reject = np.logical_or(gt_anno['names'] == 'Pedestrian',
                                   gt_anno['names'] == 'Cyclist')
        else:
            reject = gt_anno['names'] != class_name
    else:
        # if difficulty_level == 0:
        #     print("gt_anno_name:", gt_anno["name"], class_name)
        reject = gt_anno['names'] != class_name
    gt_flag[reject] = -1
    num_pred = len(pred_anno['names'])
    pred_flag = np.zeros(num_pred, dtype=np.int64)
    if use_superclass:
        if class_name == 'Vehicle':
            reject = np.logical_or(pred_anno['names'] == 'Pedestrian',
                                   pred_anno['names'] == 'Cyclist')
        else:
            reject = pred_anno['names'] != class_name
    else:
        # if difficulty_level == 0:
        #     print("pred_anno_name:", pred_anno["name"], class_name)
        reject = pred_anno['names'] != class_name
    pred_flag[reject] = -1

    if difficulty_mode == 'Overall':
        ignore = overall_filter(gt_anno['boxes_3d'])
        gt_flag[ignore] = 1
        ignore = overall_filter(pred_anno['boxes_3d'])
        pred_flag[ignore] = 1
    elif difficulty_mode == 'Distance':
        ignore = distance_filter(gt_anno['boxes_3d'], difficulty_level)
        gt_flag[ignore] = 1
        ignore = distance_filter(pred_anno['boxes_3d'], difficulty_level)
        pred_flag[ignore] = 1
    elif difficulty_mode == 'Overall&Distance':
        ignore = overall_distance_filter(gt_anno['boxes_3d'], difficulty_level)
        gt_flag[ignore] = 1
        ignore = overall_distance_filter(pred_anno['boxes_3d'],
                                         difficulty_level)
        pred_flag[ignore] = 1
    else:
        raise NotImplementedError

    return gt_flag, pred_flag


def iou3d_kernel(gt_boxes, pred_boxes):
    """
    Core iou3d computation (with cuda)

    Args:
        gt_boxes: [N, 7] (x, y, z, w, l, h, rot) in Lidar coordinates
        pred_boxes: [M, 7]

    Returns:
        iou3d: [N, M]
    """
    intersection_2d = rotate_iou_gpu_eval(gt_boxes[:, [0, 1, 3, 4, 6]],
                                          pred_boxes[:, [0, 1, 3, 4, 6]],
                                          criterion=2)
    gt_max_h = gt_boxes[:, [2]] + gt_boxes[:, [5]] * 0.5
    gt_min_h = gt_boxes[:, [2]] - gt_boxes[:, [5]] * 0.5
    pred_max_h = pred_boxes[:, [2]] + pred_boxes[:, [5]] * 0.5
    pred_min_h = pred_boxes[:, [2]] - pred_boxes[:, [5]] * 0.5
    max_of_min = np.maximum(gt_min_h, pred_min_h.T)
    min_of_max = np.minimum(gt_max_h, pred_max_h.T)
    inter_h = gt_max_h - gt_min_h  # min_of_max - max_of_min
    inter_h[inter_h <= 0] = 0
    #inter_h[intersection_2d <= 0] = 0
    intersection_3d = intersection_2d
    gt_vol = gt_boxes[:, [3]] * gt_boxes[:, [4]]
    pred_vol = pred_boxes[:, [3]] * pred_boxes[:, [4]]
    union_3d = gt_vol + pred_vol.T - intersection_3d
    #eps = 1e-6
    #union_3d[union_3d<eps] = eps
    iou3d = intersection_3d / union_3d
    return iou3d


def iou3d_kernel_with_heading(gt_boxes, pred_boxes):
    """
    Core iou3d computation (with cuda)

    Args:
        gt_boxes: [N, 7] (x, y, z, w, l, h, rot) in Lidar coordinates
        pred_boxes: [M, 7]

    Returns:
        iou3d: [N, M]
    """
    intersection_2d = rotate_iou_gpu_eval(gt_boxes[:, [0, 1, 3, 4, 6]],
                                          pred_boxes[:, [0, 1, 3, 4, 6]],
                                          criterion=2)
    gt_max_h = gt_boxes[:, [2]] + gt_boxes[:, [5]] * 0.5
    gt_min_h = gt_boxes[:, [2]] - gt_boxes[:, [5]] * 0.5
    pred_max_h = pred_boxes[:, [2]] + pred_boxes[:, [5]] * 0.5
    pred_min_h = pred_boxes[:, [2]] - pred_boxes[:, [5]] * 0.5
    max_of_min = np.maximum(gt_min_h, pred_min_h.T)
    min_of_max = np.minimum(gt_max_h, pred_max_h.T)
    inter_h = min_of_max - max_of_min
    inter_h[inter_h <= 0] = 0
    #inter_h[intersection_2d <= 0] = 0
    intersection_3d = intersection_2d
    gt_vol = gt_boxes[:, [3]] * gt_boxes[:, [4]]
    pred_vol = pred_boxes[:, [3]] * pred_boxes[:, [4]]
    union_3d = gt_vol + pred_vol.T - intersection_3d
    #eps = 1e-6
    #union_3d[union_3d<eps] = eps
    iou3d = intersection_3d / union_3d

    # rotation orientation filtering
    diff_rot = gt_boxes[:, [6]] - pred_boxes[:, [6]].T
    diff_rot = np.abs(diff_rot)
    reverse_diff_rot = 2 * np.pi - diff_rot
    diff_rot[diff_rot >= np.pi] = reverse_diff_rot[
        diff_rot >= np.pi]  # constrain to [0-pi]
    iou3d[diff_rot > np.pi / 2] = 0  # unmatched if diff_rot > 90
    return iou3d


def compute_iou3d(gt_annos, pred_annos, split_parts, with_heading):
    """
    Compute iou3d of all samples by parts

    Args:
        with_heading: filter with heading
        gt_annos: list of dicts for each sample
        pred_annos:
        split_parts: for part-based iou computation

    Returns:
        ious: list of iou arrays for each sample
    """
    gt_num_per_sample = np.stack([len(anno["names"]) for anno in gt_annos], 0)
    pred_num_per_sample = np.stack([len(anno["names"]) for anno in pred_annos],
                                   0)
    ious = []
    sample_idx = 0
    for num_part_samples in split_parts:
        gt_annos_part = gt_annos[sample_idx:sample_idx + num_part_samples]
        pred_annos_part = pred_annos[sample_idx:sample_idx + num_part_samples]

        gt_boxes = np.concatenate([anno["boxes_3d"] for anno in gt_annos_part],
                                  0)
        pred_boxes = np.concatenate(
            [anno["boxes_3d"] for anno in pred_annos_part], 0)

        if with_heading:
            iou3d_part = iou3d_kernel_with_heading(gt_boxes, pred_boxes)
        else:
            iou3d_part = iou3d_kernel(gt_boxes, pred_boxes)

        gt_num_idx, pred_num_idx = 0, 0
        for idx in range(num_part_samples):
            gt_box_num = gt_num_per_sample[sample_idx + idx]
            pred_box_num = pred_num_per_sample[sample_idx + idx]
            ious.append(iou3d_part[gt_num_idx:gt_num_idx + gt_box_num,
                                   pred_num_idx:pred_num_idx + pred_box_num])
            gt_num_idx += gt_box_num
            pred_num_idx += pred_box_num
        sample_idx += num_part_samples
    return ious
