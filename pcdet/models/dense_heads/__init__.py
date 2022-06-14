from .anchor_head_multi import AnchorHeadMulti
from .anchor_head_single import AnchorHeadSingle
from .anchor_head_template import AnchorHeadTemplate
from .point_head_box import PointHeadBox
from .point_head_simple import PointHeadSimple
from .point_intra_part_head import PointIntraPartOffsetHead
from .center_head import CenterHead
from .anchor_head_semi import AnchorHeadSemi
from .center_head_semi import CenterHeadSemi
from .center_head_aux import CenterHeadAux
from .anchor_head_aux import AnchorHeadAux
<<<<<<< HEAD
from .anchor_head_semi_aux import AnchorHeadSemiAux
=======
from .center_head_semi_aux import CenterHeadSemiAux
from .center_head_iou import CenterHeadIoU
>>>>>>> ea00e9842bf5015124eebd9786743e702023aaa5

__all__ = {
    'AnchorHeadTemplate': AnchorHeadTemplate,
    'AnchorHeadSingle': AnchorHeadSingle,
    'PointIntraPartOffsetHead': PointIntraPartOffsetHead,
    'PointHeadSimple': PointHeadSimple,
    'PointHeadBox': PointHeadBox,
    'AnchorHeadMulti': AnchorHeadMulti,
    'CenterHead': CenterHead,
    'AnchorHeadSemi': AnchorHeadSemi,
    'CenterHeadSemi': CenterHeadSemi,
    'CenterHeadAux': CenterHeadAux,
    'AnchorHeadAux': AnchorHeadAux,
<<<<<<< HEAD
    'AnchorHeadSemiAux': AnchorHeadSemiAux,
=======
    'CenterHeadSemiAux': CenterHeadSemiAux,
    'CenterHeadIoU': CenterHeadIoU
>>>>>>> ea00e9842bf5015124eebd9786743e702023aaa5
}
