import numpy as np
from aw_common import build_transform_matrix
#pose : 0:utm_east 1:utm_north 2:utm_up 3:roll 4:pitch 5:yaw 6:velo_north 7:velo_east 8:velo_down
pose0 = np.zeros(9)
pose1 = np.zeros(9)
pose1[0] = 10

pcd2utm_prev = build_transform_matrix(pose0)
pcd2utm_curr = build_transform_matrix(pose1)

pcd_prev_2_pcd_curr = np.matmul(np.linalg.inv(pcd2utm_curr), pcd2utm_prev)
print(pcd_prev_2_pcd_curr)
