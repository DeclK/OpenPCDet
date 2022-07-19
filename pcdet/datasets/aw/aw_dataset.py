import copy
import pickle
import os
import numpy as np
from pathlib import Path
import sys

from pcdet.datasets.dataset import DatasetTemplate
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.utils import box_utils
def pprint(nd_array: np.ndarray):
    l = nd_array.round(2).tolist()
    for box in l:
        for item in box:
            print(f'{item:>6.2f}', end=' ')
        print('')
class AwDataset(DatasetTemplate):
    def __init__(self,
                 dataset_cfg,
                 class_names,
                 training=True,
                 root_path=None,
                 logger=None):
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
        if self.training and self.dataset_cfg.get('BALANCED_RESAMPLING',
                                                  False):
            self.aw_infos = self.balanced_infos_resampling(self.aw_infos)

    def balanced_infos_resampling(self, infos):
        """
        Class-balanced sampling of nuScenes dataset from https://arxiv.org/abs/1908.09492
        """
        if self.class_names is None:
            return infos

        cls_infos = {name: [] for name in self.class_names}
        for info in infos:
            for name in set(info['annos']['name']):
                if name in self.class_names:
                    cls_infos[name].append(info)

        duplicated_samples = sum([len(v) for _, v in cls_infos.items()])
        cls_dist = {
            k: len(v) / duplicated_samples
            for k, v in cls_infos.items()
        }

        sampled_infos = []

        frac = 1.0 / len(self.class_names)
        ratios = [frac / v for v in cls_dist.values()]

        for cur_cls_infos, ratio in zip(list(cls_infos.values()), ratios):
            sampled_infos += np.random.choice(cur_cls_infos,
                                              int(len(cur_cls_infos) *
                                                  ratio)).tolist()
        self.logger.info('Total samples after balanced resampling: %s' %
                         (len(sampled_infos)))

        cls_infos_new = {name: [] for name in self.class_names}
        for info in sampled_infos:
            for name in set(info['annos']['name']):
                if name in self.class_names:
                    cls_infos_new[name].append(info)

        cls_dist_new = {
            k: len(v) / len(sampled_infos)
            for k, v in cls_infos_new.items()
        }

        return sampled_infos

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

    def set_split(self, mode):
        super().__init__(dataset_cfg=self.dataset_cfg,
                         class_names=self.class_names,
                         training=self.training,
                         root_path=self.root_path,
                         logger=self.logger)
        self.split = dataset_cfg.DATA_SPLIT[mode]

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_seq_list = [
            x.strip() for x in open(split_dir).readlines()
        ] if split_dir.exists() else None

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
    #    print("multi frame point number: ", points.shape[0])
        # base_path = "/home/zww/debug_data"
        # seq_path = os.path.join(base_path, sequence_id)
        # if not os.path.exists(seq_path):
        #     os.makedirs(seq_path)
        # save_bin_path = os.path.join(seq_path,'{}.bin'.format(frame_id))
        # points.tofile(save_bin_path)
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
        ######## For Evaluation Update, Save Inference Dict
        # from pcdet.utils.common_utils import EasyPickle
        # d = {'pred': eval_det_annos, 'gt':eval_gt_annos}
        # file = '/home/chk/OpenPCDet/result_test_origin.pkl'
        # EasyPickle.dump(d, file)
        ap_result_str, ap_dict = get_evaluation_results(
            eval_gt_annos, eval_det_annos, class_names)

        return ap_result_str, ap_dict
    
    def evaluation_precision_recall(self, det_annos, class_names, **kwargs):
        from .aw_eval.evaluation import get_evaluation_pr_results

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [
            copy.deepcopy(info['annos']) for info in self.aw_infos
        ]
        score_thre, precisions, recalls = get_evaluation_pr_results(
            eval_gt_annos, eval_det_annos, class_names)
        return score_thre, precisions, recalls
    
    def build_transform_matrix(self, pose):
        import math
        # pcd to utm
        #pose : 0:utm_east 1:utm_north 2:utm_up 3:roll 4:pitch 5:yaw 6:velo_north 7:velo_east 8:velo_down
        roll, pitch, yaw = pose[3:6]
        pitch = 0
        roll = 0
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        cos_pitch = math.cos(pitch)
        sin_pitch = math.sin(pitch)
        cos_roll = math.cos(roll)
        sin_roll = math.sin(roll)
        affine_matrix = np.array(
            [
                [cos_yaw * cos_pitch, cos_yaw * sin_pitch * sin_roll - sin_yaw * cos_roll, cos_yaw * sin_pitch * cos_roll + sin_yaw * sin_roll, 0],
                [sin_yaw * cos_pitch, sin_yaw * sin_pitch * sin_roll + cos_yaw * cos_roll, sin_yaw * sin_pitch * cos_roll - cos_yaw * sin_roll, 0],
                [-sin_pitch, cos_pitch * sin_roll, cos_pitch * cos_roll, 0],
                [0, 0, 0, 1]
            ]
        )
        affine_matrix[0,3] = pose[0]
        affine_matrix[1,3] = pose[1]
        affine_matrix[2,3] = pose[2]
        return affine_matrix

    def calibrate(self, boxes, names):
        """
        Calibrate Pedestrian & Cyclist boxes' direction. Because there are lots of wrong annotations
        in cyclist, and some pedestrian's direction is set to 0. Here we do some calibrations below:
        1. For cyclist, we assume the W is always the long side. If cyclist box H is longer than W,
            we switch them, and add pi/2 rotation to its theta
        2. For pedestrian, we set its theta always be 0.
        3. All classes' theta is limited within [-pi/2, pi/2]
        """
        for name in ['Pedestrian', 'Cyclist']:
            name_mask = names == name
            if name == 'Pedestrian':
                boxes[name_mask, 6] = 0
            elif name == 'Cyclist':
                cyclist_mask = boxes[:, 3] \
                             < boxes[:, 4]
                cyclist_mask &= name_mask
                indicator = np.any(cyclist_mask)
                if indicator:
                    cur_boxes = boxes[cyclist_mask]
                    boxes[cyclist_mask, 3] = cur_boxes[:, 4]
                    boxes[cyclist_mask, 4] = cur_boxes[:, 3]
                    boxes[cyclist_mask, 6] += np.pi / 2

        boxes[:, 6] = (boxes[:, 6] + 2 * np.pi) % np.pi
        yaw_mask = boxes[:, 6] > np.pi / 2
        boxes[yaw_mask, 6] -= np.pi
        
        return boxes, indicator

    def get_infos(self, num_workers=6, sample_seq_list=None):
        import concurrent.futures as futures
        import json
        root_path = self.root_path

        def process_single_sequence(seq_idx):
            print('%s seq_idx: %s' % (self.split, seq_idx))
            seq_infos = []
            seq_path = self.data_root / seq_idx
            json_path = seq_path / ('%s.json' % seq_idx)
            sweep_json_path = seq_path / ('%s.json' % (seq_idx+"_sweeps"))
            ref_lidar_path = seq_path
            with open(json_path, 'r') as f:
                info_this_seq = json.load(f)
            with open(sweep_json_path, 'r') as f:
                sweep_infos = json.load(f)
            meta_info = info_this_seq['meta_info']
            calib = info_this_seq['calib']
            for f_idx, frame in enumerate(info_this_seq['frames']):
                frame_id = frame['frame_id']
                sweep_indexs = frame['sweeps']
                if f_idx == 0:
                    prev_id = None
                else:
                    prev_id = info_this_seq['frames'][f_idx - 1]['frame_id']
                if f_idx == len(info_this_seq['frames']) - 1:
                    next_id = None
                else:
                    next_id = info_this_seq['frames'][f_idx + 1]['frame_id']
                pc_path = str(seq_path / ('%s.bin' % frame_id))
                pose = np.array(frame['pose'])
                frame_dict = {
                    'sequence_id': seq_idx,
                    'frame_id': frame_id,
                    'timestamp': frame_id,
                    'prev_id': prev_id,
                    'next_id': next_id,
                    'meta_info': meta_info,
                    'lidar': pc_path,
                    'pose': pose,
                }
                if prev_id is None:
                    frame_dict['translation'] = None
                else:
                    translation = pose[0:3] - info_this_seq['frames'][f_idx-1]['pose'][0:3]
                    frame_dict['translation'] = translation
                calib_dict = {}
                frame_dict.update({'calib': calib_dict})

                sweeps = []
                pcd2utm_curr = self.build_transform_matrix(frame["pose"])
                #generate sweeps
                ref_time = float(frame_id.split('_')[1])
                for i in range(self.nsweeps - 1):
                    if i > len(sweep_indexs)-1:
                        if len(sweeps) == 0:
                            sweep = {
                                "lidar_path": os.path.join(ref_lidar_path, frame_id + ".bin"),
                                "pose": frame["pose"],
                                "token": frame_id,
                                "transform_matrix": np.eye(4),
                                "time_lag": 0,
                            }
                        else:
                            sweep = sweeps[-1]
                    else:
                        sweep_info = sweep_infos[str(sweep_indexs[i])]
                        prev_timestamp = sweep_info["timestamp"]    
                        time_lag = ref_time - prev_timestamp
                        assert time_lag > 0, "sweep time larger than key frame"
                        pcd2utm_prev = self.build_transform_matrix(sweep_info['pose'])
                        sweep = {
                                "lidar_path": os.path.join(root_path, sweep_info["path"]),
                                "pose": sweep_info["pose"],
                                "token": sweep_info["frame_id"],
                                "transform_matrix": np.matmul(np.linalg.inv(pcd2utm_curr), pcd2utm_prev),
                                "time_lag": time_lag,
                            }
                    sweeps.append(sweep)
                frame_dict["sweeps"] = sweeps

                if 'annos' in frame:
                    annos = frame['annos']
                    boxes_3d = np.array(annos['boxes_3d'])
                    if boxes_3d.shape[0] == 0:
                        print(frame_id)
                        continue
                    boxes_2d_dict = {}
                    annos_dict = {}
                    annos_dict['vel'] = np.array(annos['vel'])
                    annos_dict['obj_id'] = np.array(annos['obj_id'])
                    #add vel and iou to boxes_3d
                    if self.code_size == 10:
                        boxes_3d = np.concatenate([boxes_3d, annos_dict['vel'], np.zeros((boxes_3d.shape[0], 1))], axis = 1)
                    annos_dict.update({
                        'name': np.array(annos['names']),
                        'boxes_3d': boxes_3d,
                        'boxes_2d': boxes_2d_dict
                    })

                    # calibrate, CHK MARK
                    self.calibrate(annos_dict['boxes_3d'], annos_dict['name'])

                    points = self.get_lidar(seq_idx, frame_id, frame_dict)
                    corners_lidar = box_utils.boxes_to_corners_3d(
                        np.array(annos['boxes_3d']))
                    num_gt = boxes_3d.shape[0]
                    num_points_in_gt = -np.ones(num_gt, dtype=np.int32)
                    for k in range(num_gt):
                        flag = box_utils.in_hull(points[:, 0:3],
                                                 corners_lidar[k])
                        num_points_in_gt[k] = flag.sum()
                    annos_dict['num_points_in_gt'] = num_points_in_gt

                    frame_dict.update({'annos': annos_dict})
                
                seq_infos.append(frame_dict)

            return seq_infos

        sample_seq_list = sample_seq_list if sample_seq_list is not None else self.sample_seq_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_sequence, sample_seq_list)
        # infos = []
        # for sample_seq in sample_seq_list:
        #     infos.append(process_single_sequence(sample_seq))
        all_infos = []
        for info in infos:
            all_infos.extend(info)
        return all_infos

    def create_groundtruth_database(self,
                                    info_path=None,
                                    used_classes=None,
                                    split='train'):
        import torch

        database_save_path = Path(
            self.root_path) / ('gt_database' if split == 'train' else
                               ('gt_database_%s' % split))
        db_info_save_path = Path(
            self.root_path) / ('aw_dbinfos_%s.pkl' % split)

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        for k in range(len(infos)):
            if 'annos' not in infos[k]:
                continue
            print('gt_database sample: %d' % (k + 1))
            info = infos[k]
            frame_id = info['frame_id']
            seq_id = info['sequence_id']
            points = self.get_lidar(seq_id, frame_id, info)
            annos = info['annos']
            names = annos['name']
            gt_boxes = annos['boxes_3d']
            vel = annos['vel'] if 'vel' in annos else None
            obj_id = annos['obj_id'] if 'obj_id' in annos else None

            num_obj = gt_boxes.shape[0]
            point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(points[:, 0:3]),
                torch.from_numpy(gt_boxes[..., :7])).numpy()  # (nboxes, npoints)

            for i in range(num_obj):
                filename = '%s_%s_%d.bin' % (frame_id, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[point_indices[i] > 0]

                gt_points[:, :3] -= gt_boxes[i, :3]
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                db_path = str(filepath.relative_to(
                    self.root_path))  # gt_database/xxxxx.bin
                db_info = {
                    'name': names[i],
                    'path': db_path,
                    'gt_idx': i,
                    'obj_id': obj_id,
              #      'box3d_lidar': np.concatenate((gt_boxes[i],vel[i])) if vel is not None else gt_boxes[i],
                    'box3d_lidar': gt_boxes[i],
                    'num_points_in_gt': gt_points.shape[0]
                }
                if names[i] in all_db_infos:
                    all_db_infos[names[i]].append(db_info)
                else:
                    all_db_infos[names[i]] = [db_info]

        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

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

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.aw_infos) * self.total_epochs
        return len(self.aw_infos)
       # return 10

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.aw_infos)

        info = copy.deepcopy(self.aw_infos[index])
        frame_id = info['frame_id']
        seq_id = info['sequence_id']
        points = self.get_lidar(seq_id, frame_id, info)

        #if self.dataset_cfg.get('POINT_PAINTING', False):
        #    points = self.point_painting(points, info)

        input_dict = {
            'points': points,
            'frame_id': frame_id,
        }

        if 'annos' in info:
            annos = info['annos']
            input_dict.update({
                'gt_names':
                annos['name'],
                'gt_boxes':
                annos['boxes_3d'],
                'num_points_in_gt':
                annos.get('num_points_in_gt', None)
            })
        data_dict = self.prepare_data(data_dict=input_dict)
        data_dict.pop('num_points_in_gt', None)
        return data_dict


def create_aw_infos(dataset_cfg, class_names, data_path, save_path, workers=4, split=None):
    dataset = AwDataset(dataset_cfg=dataset_cfg,
                        class_names=class_names,
                        root_path=data_path,
                        training=False)

    if split is not None:
        assert split in ['test', 'train', 'val']
        splits = [split]
    else:
        splits = ['test', 'train', 'val']

    print('---------------Start to generate data infos---------------')
    for split in splits:
        split_dir = data_path / 'ImageSets' / (split + '.txt')
        if not os.path.exists(split_dir):
            continue

        filename = 'aw_infos_%s.pkl' % split
        filename = save_path / Path(filename)
        dataset.set_split(split)
        aw_infos = dataset.get_infos(num_workers=workers)
        with open(filename, 'wb') as f:
            pickle.dump(aw_infos, f)
        print('AW info %s file is saved to %s' % (split, filename))

    train_filename = save_path / 'aw_infos_train.pkl'
    if not os.path.exists(train_filename):
        print("Test set create done!!")
        return
    print(
        '---------------Start create groundtruth database for data augmentation---------------'
    )
    dataset.set_split('train')
    dataset.create_groundtruth_database(train_filename, split='train')
    print('---------------Data preparation Done---------------')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file',
                        type=str,
                        default=None,
                        help='specify the config of dataset')
    parser.add_argument('--func', type=str, default='create_aw_infos', help='')
    parser.add_argument('--path', type=str, default=None, help='')
    parser.add_argument('--split', type=str, default=None, help='which split to process. if not set, all train/val/test will be processed')
    args = parser.parse_args()

    if args.func == 'create_aw_infos':
        import yaml
        from pathlib import Path
        from easydict import EasyDict
        dataset_cfg = EasyDict(yaml.safe_load(open(args.cfg_file)))

        if args.path is None:
            ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
            aw_data_path = ROOT_DIR / 'data' / 'aw'
            aw_save_path = aw_data_path
        else:
            aw_data_path = Path(args.path)
            aw_save_path = Path(args.path)

        create_aw_infos(dataset_cfg=dataset_cfg,
                        class_names=['Car', 'Bus', 'Cyclist', 'Pedestrian'], #, 'Cyclist', 'Pedestrian'
                        data_path=aw_data_path,
                        save_path=aw_save_path,
                        split=args.split)
