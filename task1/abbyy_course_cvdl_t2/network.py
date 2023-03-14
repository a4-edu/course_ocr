from torch import nn
import torch
import torch.nn.functional as F
from abbyy_course_cvdl_t2.head import CenterNetHead
from abbyy_course_cvdl_t2.backbone import ResnetBackbone
from abbyy_course_cvdl_t2.convert import PointsToObjects


class PointsNonMaxSuppression(nn.Module):
    """
    Описан в From points to bounding boxes, фильтрует находящиеся
    рядом объекты.
    """
    def __init__(self, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size

    @staticmethod
    def _nms(heat, kernel=3):
        pad = (kernel - 1) // 2

        hmax = nn.functional.max_pool2d(
            heat, (kernel, kernel), stride=1, padding=pad)
        keep = (hmax == heat).float()
        return heat * keep


    def forward(self, points):
        #return self._nms(points, self.kernel_size)
        max_conf = points[:, :-4].max(dim=1)[0]
        max_points = F.max_pool2d(max_conf, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size // 2)
        points[(max_conf != max_points).unsqueeze(1).repeat(1, points.shape[1], 1, 1)] = 0
        return points


class ScaleObjects(nn.Module):
    """
    Объекты имеют размеры в пикселях, и размер входа в сеть
    в несколько раз больше (R, output_stride), чем размер выхода.
    Из-за этого все объекты, полученные с помощью PointsToObjects
    имеют меньший размер.
    Чтобы это компенисровать, надо увеличить размеры объектов.
    """
    def __init__(self, scale: int = 4):
        super().__init__()
        self.scale = scale

    def forward(self, objects):
        b, n, d6 = objects.shape
        objects[:, :, :4] *= self.scale
        return objects


class CenterNet(nn.Module):
    """
    Детектор объектов из статьи 'Objects as Points': https://arxiv.org/pdf/1904.07850.pdf
    """
    def __init__(self, pretrained="resnet18", head_kwargs={}, nms_kwargs={}, points_to_objects_kwargs={}):
        super().__init__()
        self.backbone = ResnetBackbone(pretrained)
        self.head = CenterNetHead(**head_kwargs)
        self.return_objects = torch.nn.Sequential(
            PointsNonMaxSuppression(**nms_kwargs),
            PointsToObjects(**points_to_objects_kwargs),
            ScaleObjects()
        )

    def forward(self, input_t, return_objects=False):
        x = input_t
        x = self.backbone(x)
        x = self.head(x)
        if return_objects:
            x = self.return_objects(x)
        return x
