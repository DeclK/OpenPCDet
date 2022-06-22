import numpy as np
import pickle
import os
import json

from pathlib import Path
from functools import reduce
from typing import List

from tqdm import tqdm
from pyquaternion import Quaternion

try:
    from nuscenes import NuScenes
    from nuscenes.utils import splits
    from nuscenes.utils.data_classes import Box
    from nuscenes.utils.geometry_utils import transform_matrix
    from nuscenes.eval.detection.config import config_factory
    from nuscenes.eval.detection.evaluate import NuScenesEval
except:
    print("nuScenes devkit not Found!")

general_to_detection = {
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.wheelchair": "ignore",
    "human.pedestrian.stroller": "ignore",
    "human.pedestrian.personal_mobility": "ignore",
    "human.pedestrian.police_officer": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "animal": "ignore",
    "vehicle.car": "car",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.bicycle": "bicycle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.truck": "truck",
    "vehicle.construction": "construction_vehicle",
    "vehicle.emergency.ambulance": "ignore",
    "vehicle.emergency.police": "ignore",
    "vehicle.trailer": "trailer",
    "movable_object.barrier": "barrier",
    "movable_object.trafficcone": "traffic_cone",
    "movable_object.pushable_pullable": "ignore",
    "movable_object.debris": "ignore",
    "static_object.bicycle_rack": "ignore",
}

cls_attr_dist = {
    "barrier": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 0,
        "vehicle.parked": 0,
        "vehicle.stopped": 0,
    },
    "Cyclist": {
        "cycle.with_rider": 2791,
        "cycle.without_rider": 8946,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 0,
        "vehicle.parked": 0,
        "vehicle.stopped": 0,
    },
    "Bus": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 9092,
        "vehicle.parked": 3294,
        "vehicle.stopped": 3881,
    },
    "Car": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 114304,
        "vehicle.parked": 330133,
        "vehicle.stopped": 46898,
    },
    "construction_vehicle": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 882,
        "vehicle.parked": 11549,
        "vehicle.stopped": 2102,
    },
    "ignore": {
        "cycle.with_rider": 307,
        "cycle.without_rider": 73,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 165,
        "vehicle.parked": 400,
        "vehicle.stopped": 102,
    },
    "motorcycle": {
        "cycle.with_rider": 4233,
        "cycle.without_rider": 8326,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 0,
        "vehicle.parked": 0,
        "vehicle.stopped": 0,
    },
    "Pedestrian": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 157444,
        "pedestrian.sitting_lying_down": 13939,
        "pedestrian.standing": 46530,
        "vehicle.moving": 0,
        "vehicle.parked": 0,
        "vehicle.stopped": 0,
    },
    "traffic_cone": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 0,
        "vehicle.parked": 0,
        "vehicle.stopped": 0,
    },
    "trailer": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 3421,
        "vehicle.parked": 19224,
        "vehicle.stopped": 1895,
    },
    "truck": {
        "cycle.with_rider": 0,
        "cycle.without_rider": 0,
        "pedestrian.moving": 0,
        "pedestrian.sitting_lying_down": 0,
        "pedestrian.standing": 0,
        "vehicle.moving": 21339,
        "vehicle.parked": 55626,
        "vehicle.stopped": 11097,
    },
}

def build_transform_matrix(pose):
    import math
    # pcd to utm
    #pose : 0:utm_east 1:utm_north 2:utm_up 3:roll 4:pitch 5:yaw 6:velo_north 7:velo_east 8:velo_down

    #rotation matrix
   # print("pose:", pose)
    roll, pitch, yaw = pose[3:6]
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
  #  print("affine_matrix:", affine_matrix)
    affine_matrix[0,3] = pose[0]
    affine_matrix[1,3] = pose[1]
    affine_matrix[2,3] = pose[2]
    return affine_matrix
    

def kitti_to_aw_box(detection, class_names, data_format="kitti"):
    box3d = detection["box3d_lidar"].detach().cpu().numpy()
    scores = detection["scores"].detach().cpu().numpy()
    labels = detection["label_preds"].detach().cpu().numpy()
    names = np.array(class_names)[labels]
    vels = box3d[:, [6,7]] if box3d.shape[1] > 7 else None
    box3d = box3d[:, [0,1,2,3,4,5,-1]]
    data_format = "aw"
    if data_format == "kitti":
        box3d[:, -1] = -box3d[:, -1] - np.pi / 2
        box3d[:, [3,4]] = box3d[:, [4,3]] # kitti type to aw type

    return {
        "boxes_3d": box3d,
        "scores": scores,
        "labels": labels,
        "names": names,
        "vels": vels
    }

def aw_gt_to_eval(gt_annos, data_format="kitti"):
    data_format = "aw"
    gt_boxes = gt_annos["gt_boxes"][:, [0,1,2,3,4,5,-1]]
    if data_format == "kitti":
        gt_boxes[:, -1] = -gt_boxes[:, -1] - np.pi / 2
        gt_boxes[:, [3,4]] = gt_boxes[:, [4,3]]
    return {
        "boxes_3d": gt_boxes,
      #  "labels": labels,
        "names": gt_annos["gt_names"],
        "vels": gt_annos["gt_boxes_velocity"] if "gt_boxes_velocity" in gt_annos else None
    }

def _lidar_nusc_box_to_global(nusc, boxes, sample_token):
    try:
        s_record = nusc.get("sample", sample_token)
        sample_data_token = s_record["data"]["LIDAR_TOP"]
    except:
        sample_data_token = sample_token

    sd_record = nusc.get("sample_data", sample_data_token)
    cs_record = nusc.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
    pose_record = nusc.get("ego_pose", sd_record["ego_pose_token"])

    box_list = []
    for box in boxes:
        # Move box to ego vehicle coord system
        box.rotate(Quaternion(cs_record["rotation"]))
        box.translate(np.array(cs_record["translation"]))
        # Move box to global coord system
        box.rotate(Quaternion(pose_record["rotation"]))
        box.translate(np.array(pose_record["translation"]))
        box_list.append(box)
    return box_list


def get_sample_data(info):
    # Make list of Box objects including coord system transforms.
    # box_list = []
    # for info in frame_info:
    #     print("info: ", info)
    #     box = {
    #         "boxes_3d": info["annos"]['boxes_3d'],
    #         "velocity": info["annos"]['vel'],
    #         "car_pose": info["pose"],   
    #         "names": info['annos']['names']
    #     }
    #     box_list.append(box)
    box_list = {
            "boxes_3d": np.array(info["annos"]['boxes_3d']),
            "velocity": np.array(info["annos"]['vel']),
            "car_pose": np.array(info["pose"]),   
            "names": np.array(info['annos']['names']),
            "token": info["frame_id"]
        }

    return box_list

def get_lidar_to_image_transform(nusc, pointsensor,  camera_sensor):
    tms = []
    intrinsics = []  
    cam_paths = [] 
    for chan in CAM_CHANS:
        cam = camera_sensor[chan]

        # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
        # First step: transform the point-cloud to the ego vehicle frame for the timestamp of the sweep.
        lidar_cs_record = nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
        car_from_lidar = transform_matrix(
            lidar_cs_record["translation"], Quaternion(lidar_cs_record["rotation"]), inverse=False
        )

        # Second step: transform to the global frame.
        lidar_poserecord = nusc.get('ego_pose', pointsensor['ego_pose_token'])
        global_from_car = transform_matrix(
            lidar_poserecord["translation"],  Quaternion(lidar_poserecord["rotation"]), inverse=False,
        )

        # Third step: transform into the ego vehicle frame for the timestamp of the image.
        cam_poserecord = nusc.get('ego_pose', cam['ego_pose_token'])
        car_from_global = transform_matrix(
            cam_poserecord["translation"],
            Quaternion(cam_poserecord["rotation"]),
            inverse=True,
        )

        # Fourth step: transform into the camera.
        cam_cs_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
        cam_from_car = transform_matrix(
            cam_cs_record["translation"], Quaternion(cam_cs_record["rotation"]), inverse=True
        )

        tm = reduce(
            np.dot,
            [cam_from_car, car_from_global, global_from_car, car_from_lidar],
        )

        cam_path, _, intrinsic = nusc.get_sample_data(cam['token'])

        tms.append(tm)
        intrinsics.append(intrinsic)
        cam_paths.append(cam_path )

    return tms, intrinsics, cam_paths  

def find_closet_camera_tokens(nusc, pointsensor, ref_sample):
    lidar_timestamp = pointsensor["timestamp"]

    min_cams = {} 

    for chan in CAM_CHANS:
        camera_token = ref_sample['data'][chan]

        cam = nusc.get('sample_data', camera_token)
        min_diff = abs(lidar_timestamp - cam['timestamp'])
        min_cam = cam

        for i in range(6):  # nusc allows at most 6 previous camera frames 
            if cam['prev'] == "":
                break 

            cam = nusc.get('sample_data', cam['prev'])
            cam_timestamp = cam['timestamp']

            diff = abs(lidar_timestamp-cam_timestamp)

            if (diff < min_diff):
                min_diff = diff 
                min_cam = cam 
            
        min_cams[chan] = min_cam 

    return min_cams     


def _fill_split_infos(root_path, split_scenes, test=False, nsweeps=10, filter_zero=True):
  #  from nuscenes.utils.geometry_utils import transform_matrix

    aw_infos = []
    all_scenes = []
    for scenes in split_scenes:
        all_scenes += scenes 
        aw_infos.append([])
    data_path = os.path.join(root_path, 'data')

    for scene in tqdm(all_scenes):
        """ Manual save info["sweeps"] """        
        # Get reference pose and timestamp
        # ref_chan == "LIDAR_TOP"
        ref_lidar_path = os.path.join(data_path, scene)
        with open(os.path.join(ref_lidar_path,scene+'.json'), 'r') as f:
            sample_info = json.load(f)
        with open(os.path.join(ref_lidar_path,scene+'_sweeps.json'), 'r') as f:
            sweep_infos = json.load(f)
        for i, frame_info in enumerate(sample_info['frames']):
            frame_id = frame_info['frame_id']
            ref_sd_token = frame_id
            ref_pose_rec = frame_info['pose']
            ref_time = float(frame_id.split('_')[1])
            sweep_indexs = frame_info['sweeps']

            ref_boxes = get_sample_data(frame_info)

            info = {
                "lidar_path": os.path.join(ref_lidar_path, frame_id + ".bin"),
                "token": frame_id,
                "sweeps": [],
                "timestamp": ref_time,
                "pose": frame_info["pose"]
            }

            sweeps = []
            pcd2utm_curr = build_transform_matrix(frame_info["pose"])
            #generate sweeps
            for i in range(nsweeps - 1):
                if i > len(sweep_indexs)-1:
                    if len(sweeps) == 0:
                        sweep = {
                            "lidar_path": os.path.join(ref_lidar_path, frame_id + ".bin"),
                            "pose": frame_info["pose"],
                            "token": frame_id,
                            "transform_matrix": np.eye(4),
                            "time_lag": 0,
                        }
                    else:
                        sweep = sweeps[-1]
                else:
                    sweep_info = sweep_infos[str(sweep_indexs[i])]
               #     print("sweep_indexs: ", sweep_indexs, i, sweep_info)
                    prev_timestamp = sweep_info["timestamp"]    
                    time_lag = ref_time - prev_timestamp
                    assert time_lag > 0, "sweep time larger than key frame"
                    pcd2utm_prev = build_transform_matrix(sweep_info['pose'])
                    sweep = {
                            "lidar_path": os.path.join(root_path, sweep_info["path"]),
                            "pose": sweep_info["pose"],
                            "token": sweep_info["frame_id"],
                            "transform_matrix": np.matmul(np.linalg.inv(pcd2utm_curr), pcd2utm_prev),
                            "time_lag": time_lag,
                        }
                sweeps.append(sweep)

            info["sweeps"] = sweeps

            assert (
                len(info["sweeps"]) == nsweeps - 1
            )
            
            if not test:
                locs = np.array([b[:3] for b in ref_boxes["boxes_3d"]])
                dims = np.array([b[3:6] for b in ref_boxes["boxes_3d"]])
                # rots = np.array([b.orientation.yaw_pitch_roll[0] for b in ref_boxes]).reshape(-1, 1)
                velocity = np.array(ref_boxes["velocity"])
                rots = np.array([[b[-1]] for b in ref_boxes["boxes_3d"]])
                names = np.array(ref_boxes["names"])
                tokens = np.array(ref_boxes["token"])

                # from aw to kitti format
                dims[:, [0,1]] = dims[:, [1,0]]
                gt_boxes = np.concatenate(
                    [locs, dims, velocity[:, :2], -rots - np.pi / 2], axis=1
                ) 

                assert len(gt_boxes) == len(velocity)

                if not filter_zero:
                    info["gt_boxes"] = gt_boxes
                    info["gt_boxes_velocity"] = velocity
                    info["gt_names"] = names
                    info["gt_boxes_token"] = tokens
                else:
                    # info["gt_boxes"] = gt_boxes[mask, :]
                    # info["gt_boxes_velocity"] = velocity[mask, :]
                    # info["gt_names"] = np.array([general_to_detection[name] for name in names])[mask]
                    # info["gt_boxes_token"] = tokens[mask]
                    assert 0, "no filter zero please"
            
            for idx, scenes in enumerate(split_scenes):
                if scene in scenes:
                    aw_infos[idx].append(info)

    return aw_infos

def _fill_test_infos(root_path, test_scenes, nsweeps=10, filter_zero=True):
  #  from nuscenes.utils.geometry_utils import transform_matrix

    test_aw_infos = []
    all_scenes = test_scenes
    data_path = os.path.join(root_path, 'data')

    for scene in tqdm(all_scenes):
        """ Manual save info["sweeps"] """        
        # Get reference pose and timestamp
        # ref_chan == "LIDAR_TOP"
        ref_lidar_path = os.path.join(data_path, scene)
        
        for file in os.listdir(ref_lidar_path):
            if file.endswith(".json"):
                continue
            bin_lidar_path = os.path.join(ref_lidar_path, file)
            info = {
                "lidar_path": bin_lidar_path,
                "token": file[:-4],
                "sweeps": [],
                "timestamp": 0,
                "pose": None
            }
            test_aw_infos.append(info)
    return test_aw_infos



def quaternion_yaw(q: Quaternion) -> float:
    """
    Calculate the yaw angle from a quaternion.
    Note that this only works for a quaternion that represents a box in lidar or global coordinate frame.
    It does not work for a box in the camera frame.
    :param q: Quaternion of interest.
    :return: Yaw angle in radians.
    """

    # Project into xy plane.
    v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))

    # Measure yaw using arctan.
    yaw = np.arctan2(v[1], v[0])

    return yaw


def create_aw_infos(root_path, version="v1.0-trainval", nsweeps=10, filter_zero=True):
   # nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    available_vers = ["v1.0-trainval", "v1.0-test"]
    assert version in available_vers
   
    split_path = os.path.join(root_path, 'ImageSets')
    test = "test" in version
    if version == "v1.0-trainval":
        train_split = os.path.join(split_path, 'train.txt') 
        train_scenes = [
            x.strip() for x in open(train_split, 'r').readlines()
        ]
        val_split = os.path.join(split_path, 'val.txt') 
        val_scenes = [
            x.strip() for x in open(val_split, 'r').readlines()
        ]
        test_split = os.path.join(split_path, 'test.txt') 
        if os.path.exists(test_split):
            test_scenes = [
                x.strip() for x in open(test_split, 'r').readlines()
            ]
        else:
            test_scenes = []
        aw_infos = _fill_split_infos(
            root_path, [train_scenes, val_scenes, test_scenes], test, nsweeps=nsweeps, filter_zero=filter_zero
        )
    elif version == "v1.0-test":
        test_split = os.path.join(root_path, 'ImageSets/test.txt') 
        test_scenes = [
            x.strip() for x in open(test_split, 'r').readlines()
        ]
        test_aw_infos = _fill_test_infos(root_path, test_scenes, nsweeps=nsweeps, filter_zero=filter_zero)
    else:
        raise ValueError("unknown")

    root_path = Path(root_path)

    if test:
        print(f"test sample: {len(test_aw_infos)}")
        with open(
            root_path / "infos_test_{:02d}sweeps_withvelo.pkl".format(nsweeps), "wb"
        ) as f:
            pickle.dump(test_aw_infos, f)
    else:
        print(
            f"train sample: {len(aw_infos[0])}, val sample: {len(aw_infos[1])}, test sample: {len(aw_infos[2])}"
        )
        with open(
            root_path / "infos_train_{:02d}sweeps_withvelo_filter_{}.pkl".format(nsweeps, filter_zero), "wb"
        ) as f:
            pickle.dump(aw_infos[0], f)
        with open(
            root_path / "infos_val_{:02d}sweeps_withvelo_filter_{}.pkl".format(nsweeps, filter_zero), "wb"
        ) as f:
            pickle.dump(aw_infos[1], f)
        with open(
            root_path / "infos_test_{:02d}sweeps_withvelo_filter_{}.pkl".format(nsweeps, filter_zero), "wb"
        ) as f:
            pickle.dump(aw_infos[2], f)
    
    # test sets for vehicle type id
    split_path_test_all = os.path.join(split_path, "test_all")
    if os.path.exists(split_path_test_all):
        test_all_info_path = root_path / "test_all_infos"
        if not os.path.exists(test_all_info_path):
            os.mkdir(test_all_info_path)
        all_scenes = []
        robot_type_id_list = []
        for robot_type_id_file in os.listdir(split_path_test_all):
            robot_type_id = robot_type_id_file.split(".")[0]
            robot_type_id_list.append(robot_type_id)
            all_scenes.append([
                x.strip() for x in open(os.path.join(split_path_test_all, robot_type_id_file), 'r').readlines()
            ])
        aw_infos = _fill_split_infos(
            root_path, all_scenes, test, nsweeps=nsweeps, filter_zero=filter_zero
        )
        test_all_infos = {
            robot_type_id_list[i]: aw_infos[i] for i in range(len(robot_type_id_list))
        }
        print("test sample len: ")
        for key, item in test_all_infos.items():
            print(key, ":", len(item))
            with open(
                root_path / "test_all_infos" / (key + ".pkl"), "wb"
            ) as f:
                pickle.dump(item, f)


def eval_main(nusc, eval_version, res_path, eval_set, output_dir):
    # nusc = NuScenes(version=version, dataroot=str(root_path), verbose=True)
    cfg = config_factory(eval_version)

    nusc_eval = NuScenesEval(
        nusc,
        config=cfg,
        result_path=res_path,
        eval_set=eval_set,
        output_dir=output_dir,
        verbose=True,
    )
    metrics_summary = nusc_eval.main(plot_examples=10,)
