"""
Здесь находятся слои для конвертации между представлениями объектов.

В статье "Objects as Points" используется два представления объектов:
- Objects(bounding boxes): тензор [B, N, 6] объектов, где каждый объект имеет характеристики [y, x, h, w, cls, confidence]
- Points(heatmaps): тензор [B, C + 4, H, W] с вероятностями C классов и характеристиками
  [dy, dx, h, w] для каждой точки (y, x)
Все характеристики вычисляются в пикселях.
"""
import torch
from torch import nn
from typing import Tuple


class ObjectsToPoints(nn.Module):
    """
    В статье используются следующие обозначения:
    - H, W: 2D размеры входного изображения
    - H/R, W/R: 2D размеры выходного featuremap
    - R: output_stride, соотношение размеров входа-выхода(равен 4 для resnet backbone)

    Отличия реализации:
    - для упрощения ожидаются квадратные входы и выходы и используется HW == H == W
    - не используется R - ожидается что размеры объектов уже отмасштабированы.
    - для упрощения размер гауссианы не зависит от размера объекта и определяется smooth_kernel_size

    Пример:
        - размер входа сети: [256, 256]
        - размер выхода сети: [64, 64]
        - исходная gt деткция: [y=128, x=128, h=64, x=32, cls=0, confidence=1]
        Чтобы корректно отобразить объект на heatmap размера [64, 64] нужно:
        - передать в конструктор модуля hw=64
        - отмасштабировать исходную детекцию в 4 раза: [y=32, x=32, h=16, x=8, cls=0, confidence=1]
    """
    def __init__(self, hw: int = 64, num_classes: int = 2, smooth_kernel_size: int = 3):
        super().__init__()
        self.hw = hw
        self.с = num_classes
        if smooth_kernel_size == 0:
            self.smooth_kernel = None
        else:
            self.smooth_kernel = self._gaussian_2d(smooth_kernel_size)

    def forward(self, objects):
        """
        Реализован для упрощения - можно не писать с нуля, а только реализовать правильно методы.
        При желании, можно и переписать самостоятельно с нуля.
        """
        b, n, d6 = objects.shape
        assert d6 == 6 # y, x, h, w, cls, confidence
        points_heatmap = self.create_default_heatmaps(b, self.hw, self.с)
        points_heatmap = points_heatmap.to(objects.device)
        assert points_heatmap.shape == (b, self.с + 4, self.hw, self.hw)

        object_confidence = objects[:, :, -1]
        # На разных изображениях разное количество объектов, но тензор имеет
        # фиксированный размер [B, N, 6] - значит какие-то объекты "не настоящиее"
        # Такие объекты должны быть заполнены нулями
        assert object_confidence.shape == (b, n)
        is_real_object = (object_confidence == 1).float().flatten()

        # Проставляем 1 в локации объекта в соответствующем классе, если объект - реальный
        batch_idx, y_idx, x_idx = self.compute_objects_locations(objects)
        cls_idx = objects[:, :, -2].flatten().long()
        assert batch_idx.shape == (b * n,), (b, n, batch_idx.shape)
        assert y_idx.shape == (b * n,), y_idx.shape
        assert x_idx.shape == (b * n,), x_idx.shape
        assert batch_idx.dtype == torch.int64, batch_idx.dtype
        assert y_idx.dtype == torch.int64, y_idx.dtype
        assert x_idx.dtype == torch.int64, x_idx.dtype
        self._put(points_heatmap, (batch_idx, cls_idx, y_idx, x_idx), is_real_object)
        # При необходимости - "размываем" точки гауссианой
        if self.smooth_kernel is not None:
            if self.smooth_kernel.device != points_heatmap.device:
                self.smooth_kernel = self.smooth_kernel.to(points_heatmap.device)
            points_heatmap = self.smooth_points_heatmap(points_heatmap, self.smooth_kernel)

        # Проставляем поправки центра dy-dx в локации объекта, если объект - реальный
        dy, dx = self.compute_objects_offsets(objects)
        channel_idx = torch.zeros_like(batch_idx) + self.с
        assert dy.shape == is_real_object.shape, dy.shape
        self._put(points_heatmap, (batch_idx, channel_idx, y_idx, x_idx), is_real_object * dy)
        channel_idx += 1
        assert dx.shape == is_real_object.shape, dx.shape
        self._put(points_heatmap, (batch_idx, channel_idx, y_idx, x_idx), is_real_object * dx)

        # Проставляем размеры в локации объекта, если объект - реальный
        hy, wx = self.compute_objects_sizes(objects)
        channel_idx += 1
        assert hy.shape == is_real_object.shape, hy.shape
        self._put(points_heatmap, (batch_idx, channel_idx, y_idx, x_idx), is_real_object * hy)
        channel_idx += 1
        assert wx.shape == is_real_object.shape, wx.shape
        self._put(points_heatmap, (batch_idx, channel_idx, y_idx, x_idx), is_real_object * wx)
        return points_heatmap

    @classmethod
    def create_default_heatmaps(cls, b:int, hw: int, с:int) -> torch.Tensor:
        """
        По умолчанию heatmaps - пустые и заполнены нулями.
        """
        return torch.zeros((b, с+4, hw, hw))

    @classmethod
    def compute_objects_locations(cls, objects: torch.Tensor)-> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Принимает тензор объектов [B, N, 6].
        Возвращает для всех объектов их локацию (индекс точки в heatmap), в виде трех плоских индексов:
            batch_idx[B * N], y_idx[B * N], x_idx[B * N]
        """
        batch_size, objects_per_image, d6 = objects.shape
        res = []
        for i in range(objects.shape[0]):
            for _ in range(objects.shape[1]):
                res.append(i)
        batch_idx = torch.Tensor(res)
        batch_idx = torch.repeat_interleave(torch.arange(objects.shape[0]), objects.shape[1])
        y_idx = objects[:, :, 0].flatten()
        x_idx = objects[:, :, 1].flatten()

        return batch_idx.long(), y_idx.long(), x_idx.long()

    @classmethod
    def compute_objects_offsets(cls, objects: torch.Tensor)-> Tuple[torch.Tensor, torch.Tensor]:
        """
        Принимает тензор объектов [B, N, 6].
        Возвращает для всех объектов их поправки положения их центров по отношению к локациям
         в виде двух плоских тензоров: dy[B * N], dx[B * N]
        """
        batch_size, objects_per_image, d6 = objects.shape
        dy = objects[:, :, 0] - torch.floor(objects[:, :, 0])
        dx = objects[:, :, 1] - torch.floor(objects[:, :, 1])
        
        return dy.flatten(), dx.flatten()

    @classmethod
    def compute_objects_sizes(cls, objects: torch.Tensor)-> Tuple[torch.Tensor, torch.Tensor]:
        """
        Принимает тензор объектов [B, N, 6].
        Возвращает для всех объектов их размеры в пикселях в виде двух плоских тензоров:
            dy[B * N], dx[B * N]
        """
        h = objects[:, :, 2]
        w = objects[:, :, 3]

        return h.flatten(), w.flatten()

    @classmethod
    def smooth_points_heatmap(cls, points_heatmap: torch.Tensor, smooth_kernel:torch.Tensor):
        """
        Сглаживает one-hot points_heatmap ядром с гауссианой.
        Скорее всего, через свёртку с ядром.
        """
        shape = points_heatmap.shape[1]
        return nn.functional.conv2d(input=points_heatmap, weight=smooth_kernel[None, ...].repeat(shape, 1, 1, 1), bias=None, padding='same', groups=shape)

    @staticmethod
    def _gaussian_2d(kernel_size: int) -> torch.Tensor:
        """
        Центрированная 2d-гауссиана с ядром размера kernel_size=2K+1
        Сигма устанавливается так, чтобы в ядро влезло 3 сигмы.
        """
        assert kernel_size % 2 == 1, kernel_size
        k = kernel_size // 2
        sigma = (k + 1) / 3
        x = torch.arange(kernel_size) - k
        gauss_1d = torch.exp(
            -x ** 2 / (2 * sigma**2)
        )
        gauss_2d = gauss_1d[None, :] * gauss_1d[:, None]
        gauss_2d /= gauss_2d.max()
        return gauss_2d

    @staticmethod
    def _put(tensor, indices, values):
        # Если индексы содержат повтор, то результат a[idx] = val неопределен (зависит от порядка)
        # Если использовать index_put_ с accumulate=False, то значения суммируются корректно.
        # "Фальшивые" детекции неизбежно дают повторы, поэтому вместо присваивания -
        # прибавляем значения
        # Значения для "фальшивых" детекций должны быть 0, иначе результат будет ошибочным
        tensor.index_put_(indices, values, accumulate=True)


class PointsToObjects(nn.Module):
    """
    Обратная конвертация из hetmap(points) в bounding boxes (objects).
    Необходимо извлечь из heatmap top-k объектов и "декодировать" их характеристики, т.е.
    кординаты (локация + поправки) и размеры.
    В тестах проверяется, что:
    x == PointsToObjects()(ObjectsToPoints()(x)), где x - объекты из датасета.
    """
    def __init__(self, objects_per_image: int = 42, min_confidence=0.1):
        super().__init__()
        self.top_k = objects_per_image
        self.min_conf = min_confidence


    @staticmethod
    def _nms(heat, kernel=3):
        pad = (kernel - 1) // 2

        hmax = nn.functional.max_pool2d(
            heat, (kernel, kernel), stride=1, padding=pad)
        keep = (hmax == heat).float()
        return heat * keep


    @staticmethod
    def _topk(scores, K=42):
        batch, cat, height, width = scores.size()
        topk_scores, topk_inds = torch.topk(scores.reshape(batch, -1), K)
        topk_clses = (topk_inds / (height * width)).int()
        topk_inds = topk_inds % (height * width)
        topk_ys   = (topk_inds / width).int()
        topk_xs   = (topk_inds % width).int()
        return topk_scores, topk_clses, topk_ys, topk_xs


    def forward(self, points_heatmap):
        
        objects = torch.zeros((points_heatmap.shape[0], self.top_k, 6))

        heatmaps = points_heatmap[:, : -4, :, :]#self._nms(points_heatmap[:, : -4, :, :])
        scores, clses, ys, xs = self._topk(heatmaps, self.top_k)

        for b in range(objects.shape[0]):
            for k in range(objects.shape[1]):
                if scores[b, k] > self.min_conf:
                    objects[b, k, 0] = ys[b, k] + points_heatmap[b, -4, ys[b, k], xs[b, k]]
                    objects[b, k, 1] = xs[b, k] + points_heatmap[b, -3, ys[b, k], xs[b, k]]
                    objects[b, k, 2] = points_heatmap[b, -2, ys[b, k], xs[b, k]]
                    objects[b, k, 3] = points_heatmap[b, -1, ys[b, k], xs[b, k]]
                    objects[b, k, 4] = clses[b, k]
                    objects[b, k, 5] = scores[b, k] * (scores[b, k] > self.min_conf)

        return objects
