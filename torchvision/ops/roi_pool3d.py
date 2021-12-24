import torch
from torch import nn, Tensor

from torch.nn.modules.utils import _pair
from torch.jit.annotations import BroadcastingList2

from torchvision.extension import _assert_has_ops
from ._utils import convert_boxes_to_roi_format, check_roi_boxes3d_shape


def roi_pool3d(
    input: Tensor,
    boxes: Tensor,
    output_size: BroadcastingList2[int],
    spatial_scale: float = 1.0,
) -> Tensor:
    """
    Performs Region of Interest (RoI) Pool operator described in Fast R-CNN

    Args:
        input (Tensor[N, C, D, H, W]): input tensor
        boxes (Tensor[K, 7] or List[Tensor[L, 6]]): the box coordinates in (z1, x1, y1, z2, x2, y2)
            format where the regions will be taken from.
            The coordinate must satisfy ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
            If a single Tensor is passed,
            then the first column should contain the batch index. If a list of Tensors
            is passed, then each Tensor will correspond to the boxes for an element i
            in a batch
        output_size (int or Tuple[int, int, int]): the size of the output after the cropping
            is performed, as (height, width)
        spatial_scale (float): a scaling factor that maps the input coordinates to
            the box coordinates. Default: 1.0

    Returns:
        output (Tensor[K, C, output_size[0], output_size[1], output_size[2]])
    """
    _assert_has_ops()
    check_roi_boxes3d_shape(boxes)
    rois = boxes
    output_size = _pair(output_size)
    if not isinstance(rois, torch.Tensor):
        rois = convert_boxes_to_roi_format(rois)
    output, _ = torch.ops.torchvision.roi_pool3d(input, rois, spatial_scale,
                                                 output_size[0], output_size[1], output_size[2])
    return output


class RoIPool3d(nn.Module):
    """
    See roi_pool3d
    """

    def __init__(self, output_size: BroadcastingList2[int], spatial_scale: float):
        super(RoIPool3d, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale

    def forward(self, input: Tensor, rois: Tensor) -> Tensor:
        return roi_pool3d(input, rois, self.output_size, self.spatial_scale)

    def __repr__(self) -> str:
        tmpstr = self.__class__.__name__ + '('
        tmpstr += 'output_size=' + str(self.output_size)
        tmpstr += ', spatial_scale=' + str(self.spatial_scale)
        tmpstr += ')'
        return tmpstr