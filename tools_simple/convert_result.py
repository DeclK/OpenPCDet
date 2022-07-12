from pcdet.utils.common_utils import EasyPickle
result_file = ''
info_file = ''
dump_file = ''
result = EasyPickle.load(result_file)
info = EasyPickle.load(info_file)
import numpy as np
def id_map(class_names, det_names):
    n = len(det_names)
    map_label = np.arange(n)
    for i in range(n):
        map_label[i] = class_names.index(det_names[i])
    return map_label

data_len = len(result['gt'])
class_names = []
print(result['gt'][0].keys())
print(result['pred'][0].keys())
assert len(result['gt']) == len(result['pred']) == len(info)
for i in range(data_len):
    item_pred = result['pred'][i]
    item_gt = result['gt'][i]

    item_pred['names'] = item_pred.pop('name')
    item_pred['scores'] = item_pred.pop('score')
    item_pred['labels'] = id_map(class_names, item_pred['names'])

    item_gt['names'] = item_gt.pop('name')
    item_gt['vels'] = item_gt.pop('vel')
    item_gt['token'] = info[i]['frame_id']
    for key in list(item_gt.keys()):
        if key not in ['boxes_3d', 'names', 'vels']:
            item_gt.pop(key)
    for key in list(item_pred.keys()):
        if key not in ['boxes_3d', 'names', 'scores', 'labels', 'vels']:
            item_pred.pop(key)

EasyPickle.dump(result, dump_file)
