from collections import defaultdict
from typing import Optional, Callable, List, Any, Tuple

import numpy as np
from PIL import Image

import torch
from torchvision import transforms
from torchvision.datasets import VisionDataset
from torchvision.transforms import functional as F


class HeatmapDataset(VisionDataset):
    def __init__(self, data_packs, split='train', transforms=None):
        self.data_packs = data_packs
        self.indices = []
        self.transforms = transforms

        for dp_idx, dp in enumerate(data_packs):
            for im_idx, im in enumerate(dp):
                if im.is_test_split() and split == 'test':
                    self.indices.append((dp_idx, im_idx))
                elif not im.is_test_split() and split == 'train':
                    self.indices.append((dp_idx, im_idx))
    

    def __len__(self):
        return len(self.indices)
    

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        dp_idx, im_idx = self.indices[index]
        image = np.array(self.data_packs[dp_idx][im_idx].image.convert('RGB'))
        target = torch.FloatTensor(self.data_packs[dp_idx][im_idx].quadrangle)

        if self.transforms is not None:
            image = self.transforms(image)

        return image, target
    
    def get_key(self, index: int) -> str:
        dp_idx, im_idx = self.indices[index]
        
        return self.data_packs[dp_idx][im_idx].unique_key