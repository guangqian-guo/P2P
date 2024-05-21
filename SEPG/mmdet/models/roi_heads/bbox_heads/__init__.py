
from .bbox_head import BBoxHead
from .convfc_bbox_head import (ConvFCBBoxHead, Shared2FCBBoxHead,
                               Shared4Conv1FCBBoxHead)
from .dii_head import DIIHead
from .double_bbox_head import DoubleConvFCBBoxHead
from .sabl_head import SABLHead
from .scnet_bbox_head import SCNetBBoxHead
from .MIL_bbox_head import Shared2FCInstanceMILHead
from .MIL_bbox_head import Shared2FCInstanceMILHead_refine
from .oicr_head import OICRHead
from .wsddn_head import WSDDNHead

from .MSE_MIL_Headv7 import MSEMILHeadv7

__all__ = [
    'BBoxHead', 'ConvFCBBoxHead', 'Shared2FCBBoxHead',
    'Shared4Conv1FCBBoxHead', 'DoubleConvFCBBoxHead', 'SABLHead', 'DIIHead',
    'SCNetBBoxHead', 'Shared2FCInstanceMILHead', 'OICRHead', 'WSDDNHead', 'Shared2FCInstanceMILHead_refine',
    'Shared2FCInstanceHierMILHead', 
    'MSEMILHeadv7' # added 
]
