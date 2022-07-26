import os
import copy
import pickle
import numpy as np

from pcdet.datasets.semi_dataset import SemiDatasetTemplate


class AWSemiDataset(SemiDatasetTemplate):
    def __init__(self,
                 dataset_cfg,
                 class_names,
                 training=True,
                 root_path=None,
                 logger=None,
                 infos=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(dataset_cfg=dataset_cfg,
                         class_names=class_names,
                         training=training,
                         root_path=root_path,
                         logger=logger)
        self.mode_ori = self.mode
        self.split = dataset_cfg.DATA_SPLIT[self.mode]
        assert self.split in ['train', 'val', 'test']
        # self.pts_dim = len(
        #     dataset_cfg.POINT_FEATURE_ENCODING.used_feature_list)

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_seq_list = [
            x.strip() for x in open(split_dir).readlines()
        ] if split_dir.exists() else None
        self.cam_names = ['cam00', 'cam01', 'cam02', 'cam03', 'cam04', 'cam05']

        self.data_root = self.root_path / 'data'
        self.nsweeps = dataset_cfg["nsweeps"] if "nsweeps" in dataset_cfg else 1
        self.num_feature = dataset_cfg["num_feature"] if "num_feature" in dataset_cfg else 4
        self.code_size = dataset_cfg["code_size"] if "code_size" in dataset_cfg else 7
        self.pts_dim = self.num_feature
        self.pc_range = dataset_cfg["POINT_CLOUD_RANGE"]

        self.aw_infos = []
        self.include_aw_data(self.split)
        print("dataset len: ", len(self.aw_infos))

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.aw_infos) * self.total_epochs
        return len(self.aw_infos)

    def include_aw_data(self, split):
        if self.logger is not None:
            self.logger.info('Loading aw dataset')
        aw_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[split]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                aw_infos.extend(infos)

        def check_annos(info):
            return 'annos' in info if self.training else True
        aw_infos = list(filter(check_annos, aw_infos))

        self.aw_infos.extend(aw_infos)

        if self.logger is not None:
            self.logger.info('Total samples for aw dataset: %d' %
                             (len(aw_infos)))

    def remove_close(self, points, radius: float):
        """
        Removes point too close within a certain radius from origin.
        :param radius: Radius below which points are removed.
        """
        x_filt = np.abs(points[0, :]) < radius
        y_filt = np.abs(points[1, :]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        points = points[:, not_close]
        return points

    def get_sweep(self, sweep_info):
        min_distance = 1.0
        points_sweep = np.fromfile(sweep_info["lidar_path"],dtype=np.float32).reshape(-1, self.num_feature).T
        points_sweep = self.remove_close(points_sweep, min_distance)
        nbr_points = points_sweep.shape[1]
        if sweep_info["transform_matrix"] is not None:
            points_sweep[:3, :] = sweep_info["transform_matrix"].dot(
                np.vstack((points_sweep[:3, :], np.ones(nbr_points)))
            )[:3, :]
        curr_times = sweep_info["time_lag"] * np.ones((1, points_sweep.shape[1]))
        return points_sweep.T, curr_times.T

    def get_lidar(self, sequence_id, frame_id, info):
        bin_path = os.path.join(self.data_root, sequence_id,
                                '{}.bin'.format(frame_id))
        points = np.fromfile(bin_path,
                             dtype=np.float32).reshape(-1, self.pts_dim)
        if self.nsweeps > 1:
            sweep_points_list = [points]
            sweep_times_list = [np.zeros((points.shape[0], 1))]
            for i in np.random.choice(len(info["sweeps"]), self.nsweeps - 1, replace=False):
                sweep = info["sweeps"][i]
                points_sweep, times_sweep = self.get_sweep(sweep)
                sweep_points_list.append(points_sweep)
                sweep_times_list.append(times_sweep)

            points = np.concatenate(sweep_points_list, axis=0)
            times = np.concatenate(sweep_times_list, axis=0).astype(points.dtype)

            points = np.hstack([points, times])
        return points

    def evaluation(self, det_annos, class_names, **kwargs):
        from .aw_eval.evaluation import get_evaluation_results

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [
            copy.deepcopy(info['annos']) for info in self.aw_infos
        ]
        for i in range(len(eval_gt_annos)):
            select = []
            #print("gt_boxes: ", eval_gt_annos[i]["boxes_3d"])
            for box in eval_gt_annos[i]["boxes_3d"]:
                if box[0] > self.pc_range[0] and box[0] < self.pc_range[3] and box[1] > self.pc_range[1] and box[1] < self.pc_range[4]:
                    select.append(True)
                else:
                    select.append(False)
            #print("select: ", select)
            eval_gt_annos[i]["boxes_3d"] = eval_gt_annos[i]["boxes_3d"][select]
            eval_gt_annos[i]["vel"] = eval_gt_annos[i]["vel"][select]
            eval_gt_annos[i]["name"] = eval_gt_annos[i]["name"][select]
            eval_gt_annos[i]["num_points_in_gt"] = eval_gt_annos[i]["num_points_in_gt"][select]
            #print("gt_boxes after: ", eval_gt_annos[i]["boxes_3d"])
        ap_result_str, ap_dict = get_evaluation_results(
            eval_gt_annos, eval_det_annos, class_names)

        return ap_result_str, ap_dict
    
    @staticmethod
    def generate_prediction_dicts(batch_dict,
                                  pred_dicts,
                                  class_names,
                                  output_path=None):
        # transform the unified normative coordinate to aw required coordinate
        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples),
                'score': np.zeros(num_samples),
                'boxes_3d': np.zeros((num_samples, 7))
            }
            return ret_dict

        def generate_single_sample_dict(box_dict):
            import math
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict
            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_3d'] = pred_boxes[..., :7]
            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            frame_id = batch_dict['frame_id'][index]
            single_pred_dict = generate_single_sample_dict(box_dict)
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)

            if output_path is not None:
                raise NotImplementedError
        return annos

class AWPretrainDataset(AWSemiDataset):
    def __init__(self, dataset_cfg, class_names, infos=None, training=True, root_path=None, logger=None):
        assert training is True
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, infos=infos, training=training, root_path=root_path, logger=logger
        )

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.aw_infos)

        info = copy.deepcopy(self.aw_infos[index])
        frame_id = info['frame_id']
        seq_id = info['sequence_id']
        points = self.get_lidar(seq_id, frame_id)
        input_dict = {
            'points': points,
            'frame_id': frame_id,
        }

        if 'annos' in info:
            annos = info['annos']
            input_dict.update({
                'gt_names': annos['name'],
                'gt_boxes': annos['boxes_3d'],
                'num_points_in_gt': annos.get('num_points_in_gt', None)
            })

        data_dict = self.prepare_data(data_dict=input_dict)
        data_dict.pop('num_points_in_gt', None)
        return data_dict

class AWLabeledDataset(AWSemiDataset):
    def __init__(self, dataset_cfg, class_names, infos=None, training=True, root_path=None, logger=None):
        assert training is True
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, infos=infos, training=training, root_path=root_path, logger=logger
        )
        self.labeled_data_for = dataset_cfg.LABELED_DATA_FOR    # for ['teacher', 'student']

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.aw_infos)

        info = copy.deepcopy(self.aw_infos[index])
        frame_id = info['frame_id']
        seq_id = info['sequence_id']
        points = self.get_lidar(seq_id, frame_id, info)
        input_dict = {
            'points': points,
            'frame_id': frame_id,
        }

        assert 'annos' in info
        annos = info['annos']
        input_dict.update({
                'gt_names': annos['name'],
                'gt_boxes': annos['boxes_3d'],
                'num_points_in_gt': annos.get('num_points_in_gt', None)
        })

        teacher_dict, student_dict = self.prepare_data_ssl(input_dict, output_dicts=self.labeled_data_for)
        if teacher_dict is not None: teacher_dict.pop('num_points_in_gt', None)
        if student_dict is not None: student_dict.pop('num_points_in_gt', None)
        return tuple([teacher_dict, student_dict])

class AWUnlabeledDataset(AWSemiDataset):
    def __init__(self, dataset_cfg, class_names, infos=None, training=True, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        assert training is True
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, infos=infos, training=training, root_path=root_path, logger=logger
        )
        self.unlabeled_data_for = dataset_cfg.UNLABELED_DATA_FOR

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.aw_infos)

        info = copy.deepcopy(self.aw_infos[index])
        frame_id = info['frame_id']
        seq_id = info['sequence_id']
        points = self.get_lidar(seq_id, frame_id, info)
        input_dict = {
            'points': points,
            'frame_id': frame_id,
        }

        teacher_dict, student_dict = self.prepare_data_ssl(input_dict, output_dicts=self.unlabeled_data_for)
        return tuple([teacher_dict, student_dict])

class AWTestDataset(AWSemiDataset):
    def __init__(self, dataset_cfg, class_names, infos=None, training=False, root_path=None, logger=None):
        assert training is False
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, infos=infos, training=training, root_path=root_path, logger=logger
        )

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.aw_infos)

        info = copy.deepcopy(self.aw_infos[index])
        frame_id = info['frame_id']
        seq_id = info['sequence_id']
        points = self.get_lidar(seq_id, frame_id, info)
        input_dict = {
            'points': points,
            'frame_id': frame_id,
        }

        if 'annos' in info:
            annos = info['annos']
            input_dict.update({
                'gt_names': annos['name'],
                'gt_boxes': annos['boxes_3d'],
                'num_points_in_gt': annos.get('num_points_in_gt', None)
            })

        data_dict = self.prepare_data(data_dict=input_dict)
        data_dict.pop('num_points_in_gt', None)
        return data_dict