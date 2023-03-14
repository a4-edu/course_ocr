""" Здесь находится 'Голова' CenterNet, описана в разделе 4 статьи https://arxiv.org/pdf/1904.07850.pdf"""
from torch import nn
import torch


class CenterNetHead(nn.Module):
    """
    Принимает на вход тензор из Backbone input[B, K, W/R, H/R], где
    - B = batch_size
    - K = количество каналов (в ResnetBackbone K = 64)
    - H, W = размеры изображения на вход Backbone
    - R = output stride, т.е. во сколько раз featuremap меньше, чем исходное изображение
      (в ResnetBackbone R = 4)

    Возвращает тензора [B, C+4, W/R, H/R]:
    - первые C каналов: probs[B, С, W/R, H/R] - вероятности от 0 до 1
    - еще 2 канала: offset[B, 2, W/R, H/R] - поправки координат в пикселях от 0 до 1
    - еще 2 канала: sizes[B, 2, W/R, H/R] - размеры объекта в пикселях
    """
    def __init__(self, k_in_channels=64, c_classes: int = 2):
        super().__init__()
        self.c_classes = c_classes
        
        self.probs_head = nn.Sequential(
            nn.Conv2d(k_in_channels, k_in_channels,
                kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(k_in_channels, c_classes, 
                kernel_size=1, stride=1, 
                padding=0),
            #nn.Softmax(dim=-1)
            nn.Sigmoid()
            )

        self.offset_head = nn.Sequential(
            nn.Conv2d(k_in_channels, k_in_channels,
                kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(k_in_channels, 2, 
                kernel_size=1, stride=1, 
                padding=0),
             #nn.Softmax(dim=-1)
            nn.Sigmoid()
            )

        self.size_head = nn.Sequential(
            nn.Conv2d(k_in_channels, k_in_channels,
                kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(k_in_channels, 2, 
                kernel_size=1, stride=1, 
                padding=0),
            #nn.Softmax(dim=-1)
            )


    def forward(self, input_t: torch.Tensor):
        class_heatmap = self.probs_head(input_t)
        offset_map = self.offset_head(input_t)
        size_map = self.size_head(input_t)

        return torch.cat([class_heatmap, offset_map, size_map], dim=1)
