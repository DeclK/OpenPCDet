from .detector3d_template import Detector3DTemplate
from .PartA2_net import PartA2Net
from .point_rcnn import PointRCNN
from .pointpillar import PointPillar
from .pv_rcnn import PVRCNN
from .second_net import SECONDNet
from .second_net_iou import SECONDNetIoU
from .caddn import CaDDN
from .voxel_rcnn import VoxelRCNN
from .centerpoint import CenterPoint
from .pv_rcnn_plusplus import PVRCNNPlusPlus
from .aux_pillar import AuxPillar
from .aux_ssd import AuxSSD
from .cia_ssd import CIASSD
from .semi_second import SemiSECOND
from .semi_centerpoint import SemiCenterPoint
from .cg_ssd import CGSSD
from .semi_cgssd import SemiCGSSD
from .centerpoint_aux import CenterPointAux
from .semi_centerpoint_aux import SemiCenterPointAux

__all__ = {
    'Detector3DTemplate': Detector3DTemplate,
    'SECONDNet': SECONDNet,
    'PartA2Net': PartA2Net,
    'PVRCNN': PVRCNN,
    'PointPillar': PointPillar,
    'PointRCNN': PointRCNN,
    'SECONDNetIoU': SECONDNetIoU,
    'CaDDN': CaDDN,
    'VoxelRCNN': VoxelRCNN,
    'CenterPoint': CenterPoint,
    'PVRCNNPlusPlus': PVRCNNPlusPlus,
    'AuxPillar': AuxPillar,
    'AuxSSD': AuxSSD,
    'CIASSD': CIASSD,
    'SemiSECOND': SemiSECOND,
    'SemiCenterPoint': SemiCenterPoint,
    'CGSSD': CGSSD,
    'SemiCGSSD': SemiCGSSD,
    'CenterPointAux': CenterPointAux,
    'SemiCenterPointAux': SemiCenterPointAux,
}


def build_detector(model_cfg, num_class, dataset):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )

    return model
