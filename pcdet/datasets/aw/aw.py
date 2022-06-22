import sys
import pickle
import json
import random
import operator
import numpy as np

from functools import reduce
from pathlib import Path
from copy import deepcopy

try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.eval.detection.config import config_factory
except:
    print("nuScenes devkit not found!")

from det3d.datasets.custom import PointCloudDataset
from det3d.datasets.aw.aw_common import (general_to_detection, cls_attr_dist,
                                         kitti_to_aw_box, aw_gt_to_eval,
                                         _lidar_nusc_box_to_global, eval_main)
from det3d.datasets.registry import DATASETS


@DATASETS.register_module
class AwDataset(PointCloudDataset):
    #  NumPointFeatures = 4  # x, y, z, intensity, time_diff

    def __init__(
        self,
        info_path,
        root_path,
        nsweeps=0,  # here set to zero to catch unset nsweep
        cfg=None,
        pipeline=None,
        class_names=None,
        test_mode=False,
        detect_mode=False,
        version="v1.0-trainval",
        **kwargs,
    ):
        super(AwDataset, self).__init__(root_path,
                                        info_path,
                                        pipeline,
                                        test_mode=test_mode,
                                        detect_mode=detect_mode,
                                        class_names=class_names)
        self._set_group_flag()

        self.nsweeps = nsweeps
        if self.nsweeps == 1:
            self.NumPointFeatures = 4
        else:
            self.NumPointFeatures = 5
        assert self.nsweeps > 0, "more sweep please"
        print("sweeps: ", nsweeps)

        if not hasattr(self, "_aw_infos"):
            self.load_infos(self._info_path)
        self._num_point_features = self.NumPointFeatures
        self._name_mapping = general_to_detection

        self.virtual = kwargs.get('virtual', False)
        if self.virtual:
            self._num_point_features = 16
       # self.data_format = kwargs.get("data_format", "kitti")
        self.data_format = "kitti"
        print("data_format: ", self.data_format)

        self.version = version
        self.eval_version = "aw"
        self.pc_range = kwargs.get("pc_range", [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])

    # self.idx = 0

    def reset(self):
        self.logger.info(f"re-sample {self.frac} frames from full set")
        random.shuffle(self._aw_infos)
        self._aw_infos = self._aw_infos[:self.frac]

    def load_infos(self, info_path):
        # if self.detect_mode:
        #     self._aw_infos = None
        with open(info_path, "rb") as f:
            _aw_info_all = pickle.load(f)

        if not self.test_mode and not self.detect_mode:
            train_ratio = 1
            self.frac = int(len(_aw_info_all) * train_ratio)

            _cls_infos = {name: [] for name in self._class_names}
            for info in _aw_info_all:
                for name in set(info["gt_names"]):
                    if name in self._class_names:
                        _cls_infos[name].append(info)

            duplicated_samples = sum([len(v) for _, v in _cls_infos.items()])
            _cls_dist = {
                k: len(v) / max(duplicated_samples, 1)
                for k, v in _cls_infos.items()
            }

            self._aw_infos = []

            frac = 1.0 / len(self._class_names)
            ratios = [frac / v for v in _cls_dist.values()]

            for cls_infos, ratio in zip(list(_cls_infos.values()), ratios):
                self._aw_infos += np.random.choice(cls_infos,
                                                   int(len(cls_infos) *
                                                       ratio)).tolist()

            _cls_infos = {name: [] for name in self._class_names}
            for info in self._aw_infos:
                for name in set(info["gt_names"]):
                    if name in self._class_names:
                        _cls_infos[name].append(info)

            _cls_dist = {
                k: len(v) / len(self._aw_infos)
                for k, v in _cls_infos.items()
            }
        else:
            if isinstance(_aw_info_all, dict):
                self._aw_infos = []
                for v in _aw_info_all.values():
                    self._aw_infos.extend(v)
            else:
                self._aw_infos = _aw_info_all

    def __len__(self):

        if not hasattr(self, "_aw_infos"):
            self.load_infos(self._info_path)

        return len(self._aw_infos)

    #    return 10

    @property
    def ground_truth_annotations(self):
        if "gt_boxes" not in self._aw_infos[0]:
            return None
        cls_range_map = config_factory(
            self.eval_version).serialize()['class_range']
        gt_annos = []
        for info in self._aw_infos:
            gt_names = np.array(info["gt_names"])
            gt_boxes = info["gt_boxes"]
            mask = np.array([n != "ignore" for n in gt_names], dtype=np.bool_)
            gt_names = gt_names[mask]
            gt_boxes = gt_boxes[mask]
            # det_range = np.array([cls_range_map[n] for n in gt_names_mapped])
            det_range = np.array([cls_range_map[n] for n in gt_names])
            det_range = det_range[..., np.newaxis] @ np.array([[-1, -1, 1, 1]])
            mask = (gt_boxes[:, :2] >= det_range[:, :2]).all(1)
            mask &= (gt_boxes[:, :2] <= det_range[:, 2:]).all(1)
            N = int(np.sum(mask))
            gt_annos.append({
                "bbox": np.tile(np.array([[0, 0, 50, 50]]), [N, 1]),
                "alpha": np.full(N, -10),
                "occluded": np.zeros(N),
                "truncated": np.zeros(N),
                "name": gt_names[mask],
                "location": gt_boxes[mask][:, :3],
                "dimensions": gt_boxes[mask][:, 3:6],
                "rotation_y": gt_boxes[mask][:, 6],
                "token": info["token"],
            })
        return gt_annos

    def get_sensor_data(self, idx):

        # if self.detect_mode:
        #     info = {
        #         "data_format": self.data_format,
        #         "token": None}
        # else:
        info = self._aw_infos[idx]
        info["data_format"] = self.data_format

        res = {
            "lidar": {
                "type": "lidar",
                "points": None,
                "nsweeps": self.nsweeps,
                # "ground_plane": -gp[-1] if with_gp else None,
                "annotations": None,
            },
            "metadata": {
                "image_prefix": self._root_path,
                "num_point_features": self._num_point_features,
                "token":
                info["token"] if "token" in info else info["frame_id"],
            },
            "calib": None,
            "cam": {},
            "mode": "val" if self.test_mode else "train",
            "virtual": self.virtual
        }

        data, _ = self.pipeline(res, info)
        # import os
        # print("combined shape: ", data['points'].shape)
        # print("idx: ", idx, self.idx)
        # base_path = "/home/zww/debug_data"
        # seq_path = os.path.join(base_path, "0")
        # if not os.path.exists(seq_path):
        #     os.makedirs(seq_path)
        # save_bin_path = os.path.join(seq_path,'{}.bin'.format(idx))
        # data['points'].tofile(save_bin_path)
        # self.idx += 1
        # if self.idx > 20:
        #     os._exit(1)

        return data

    def __getitem__(self, idx):
        return self.get_sensor_data(idx)

    # def get_det_annos(self):
    #     gt_annos = {}
    #     for info in self._aw_infos:
    #         gt_annos[info["token"]] = info
    #     return gt_annos

    def get_gt_and_det_annos(self, detections):
        dets = []
        gts = []
        gt_annos = self._aw_infos
        assert gt_annos is not None
        miss = 0
        for gt in gt_annos:
            # try:
            #     dets.append(kitti_to_aw_box(detections[gt["token"]], self._class_names))
            #     gts.append(aw_gt_to_eval(gt))
            # except Exception:
            #     miss += 1
            if gt["token"] in detections:
                dets.append(
                    kitti_to_aw_box(detections[gt["token"]],
                                    self._class_names, self.data_format))
                gts.append(aw_gt_to_eval(gt, self.data_format))
            else:
                miss += 1
        #    assert miss == 0
        print("all and miss:", len(self._aw_infos), miss)
        return gts, dets

    def evaluation(self, detections, output_dir=None, testset=False):
        class_names = np.array(self._class_names)
        print("detections:", len(detections.keys()), len(self._aw_infos))
        eval_gt_annos, eval_det_annos = self.get_gt_and_det_annos(detections)
        print("gts and det len: ", len(eval_gt_annos), len(eval_det_annos))

        from .aw_eval.evaluation import get_evaluation_results
        gt_label_statis = np.zeros(len(class_names))
        for gt_anno in eval_gt_annos:

            labels = gt_anno['names']
            if len(labels) == 0:
                continue
            for idx, name in enumerate(class_names):
                gt_label_statis[idx] += (labels == name).sum(axis=0)
        print('gt_label_statis:', gt_label_statis)
        pred_label_statis = np.zeros(len(class_names))
        for det_anno in eval_det_annos:
            labels = det_anno['names']
            if len(labels) == 0:
                continue
            for idx, name in enumerate(class_names):
                pred_label_statis[idx] += (labels == name).sum(axis=0)
        print('pred_label_statis:', pred_label_statis)

        ap_result_str, ap_dict = get_evaluation_results(eval_gt_annos,
                                                        eval_det_annos,
                                                        class_names,
                                                        print_ok=False)

        return ap_result_str, ap_dict

    def evaluation_precision_recall(self,
                                    detections,
                                    output_dir=None,
                                    testset=False):
        class_names = np.array(self._class_names)
        print("detections:", len(detections.keys()), len(self._aw_infos))
        eval_gt_annos, eval_det_annos = self.get_gt_and_det_annos(detections)
        print("gts and det len: ", len(eval_gt_annos), len(eval_det_annos))

        from .aw_eval.evaluation import get_evaluation_pr_results
        gt_label_statis = np.zeros(len(class_names))
        for gt_anno in eval_gt_annos:

            labels = gt_anno['names']
            if len(labels) == 0:
                continue
            for idx, name in enumerate(class_names):
                gt_label_statis[idx] += (labels == name).sum(axis=0)
        print('gt_label_statis:', gt_label_statis)
        pred_label_statis = np.zeros(len(class_names))
        for det_anno in eval_det_annos:
            labels = det_anno['names']
            if len(labels) == 0:
                continue
            for idx, name in enumerate(class_names):
                pred_label_statis[idx] += (labels == name).sum(axis=0)
        print('pred_label_statis:', pred_label_statis)

        # for i in range(len(eval_gt_annos)):
        #     select = []
        #     #print("gt_boxes: ", eval_gt_annos[i]["boxes_3d"])
        #     for box in eval_gt_annos[i]["boxes_3d"]:
        #         if box[0] > self.pc_range[0] and box[0] < self.pc_range[3] and box[1] > self.pc_range[1] and box[1] < self.pc_range[4]:
        #             select.append(True)
        #         else:
        #             select.append(False)
        #     #print("select: ", select)
        #     eval_gt_annos[i]["boxes_3d"] = eval_gt_annos[i]["boxes_3d"][select]
        #     eval_gt_annos[i]["vel"] = eval_gt_annos[i]["vel"][select]
        #     eval_gt_annos[i]["name"] = eval_gt_annos[i]["name"][select]
        #     eval_gt_annos[i]["num_points_in_gt"] = eval_gt_annos[i]["num_points_in_gt"][select]

        score_thre, precisions, recalls = get_evaluation_pr_results(
            eval_gt_annos, eval_det_annos, class_names, print_ok=True)

        return score_thre, precisions, recalls
