from collections import defaultdict
from typing import Optional, Callable, List, Any, Tuple
import json
import warnings

import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torchvision.datasets import VisionDataset, CocoDetection
from torchvision.transforms import functional as F


class CocoTextDetection(VisionDataset):
    def __init__(
        self,
        root: str,
        annFile: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        area_fraction_threshold: Optional[float]=1/32**2,
        split='train'
    ):
        super().__init__(root, transforms, transform, target_transform)
        assert(self.root.exists())
        self.area_fraction_threshold = area_fraction_threshold

        with open(annFile, 'r') as f:
            self.ann_data = json.loads(f.read())
        self._image_anno_index = defaultdict(list)
        for k, ann in self.ann_data['anns'].items():
            image_id = str(ann['image_id'])
            if split is not None:
                if self.ann_data['imgs'][image_id]['set'] != split:
                    continue
            if self._img_id_path(image_id).exists():
                self._image_anno_index[image_id].append(k)
        self.ids = list(sorted(self._image_anno_index.keys()))

    def _img_id_path(self, image_id):
         return self.root / self.ann_data['imgs'][image_id]["file_name"]

    def _img_id_area(self, image_id):
         return self.ann_data['imgs'][image_id]["height"] * self.ann_data['imgs'][image_id]["width"]

    def _load_image(self, id: int) -> Image.Image:
        return Image.open(self._img_id_path(self.ids[id])).convert("RGB")

    def _load_target(self, id) -> List[Any]:
        annos = []
        target_anno_ids = self._image_anno_index[self.ids[id]]
        for ann_id in target_anno_ids:
            ann = self.ann_data['anns'][str(ann_id)]
            if self.area_fraction_threshold > 0:
                if (ann['area'] / self._img_id_area(str(ann['image_id']))) < self.area_fraction_threshold:
                    continue
            ann_selected = {
                'bbox': ann['bbox'],
                'category_id': int(ann['class'] == 'machine printed')
            }
            annos.append(ann_selected)

        return annos

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image = self._load_image(index)
        target = self._load_target(index)

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self.ids)


class CocoDetectionPrepareTransform(transforms.Resize):

    N_DETS_PER_SAMPLE = 100

    def __init__(self, ids_map={}, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ids_map = ids_map

    def __call__(self, coco_image, coco_anno):
        # вытаскиваем все bbox
        coco_dets = np.array([
            np.array([a['bbox'][1], a['bbox'][0], a['bbox'][3], a['bbox'][2]])
            for a in coco_anno
        ])
        if not len(coco_dets):
            coco_dets = np.zeros((0, 4))
        # переходим от формата COCO [ymin, xmin, h, w]
        # к формату [yc, xc, h, w]
        coco_dets[:, 0:2] += coco_dets[:, 2:] / 2

        # масштабируем предсказания в соответствии с ресайзом изображения
        h_cur, w_cur = coco_image.height, coco_image.width
        h_target, w_target = self.size
        resize_scale_hw = np.array([[h_target, w_target, h_target, w_target]]) / np.array([[h_cur, w_cur, h_cur, w_cur]])
        dets_resized = coco_dets * resize_scale_hw

        # масштабируем изображение
        img_prepared = super().__call__(coco_image)
        # добавляем классы, ставим -1 для тех, что хотим выкинуть
        id_map = lambda x: self.ids_map.get(x, -1) if len(self.ids_map) else x
        classes = np.array([
            id_map(ca['category_id'])
            for ca in coco_anno
        ])
        anno_prepared = np.concatenate([
            dets_resized, classes[:, None], np.ones((len(classes), 1))
        ], axis=-1)
        anno_prepared = anno_prepared[classes>=0]

        if len(anno_prepared) >= self.N_DETS_PER_SAMPLE:
            # если детекций больше - выбрасываем что не влезло
            anno_prepared = anno_prepared[:self.N_DETS_PER_SAMPLE]
            warnings.warn(
                f"Detections overflow: can't fit {len(anno_prepared)} into {self.N_DETS_PER_SAMPLE}. Increase N to avoid."
            )
        else:
            # если детекций меньше N_DETS_PER_SAMPLE - заполняем 0
            anno_full = np.zeros((self.N_DETS_PER_SAMPLE, 6)).astype(np.float32)
            anno_full[:len(anno_prepared)] = anno_prepared
            anno_prepared = anno_full
        return F.to_tensor(img_prepared), torch.from_numpy(anno_prepared)
