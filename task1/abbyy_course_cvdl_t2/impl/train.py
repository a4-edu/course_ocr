import sys
import warnings

import numpy as np

import torch
from torch import optim as optim

from torchvision import transforms
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as F

from abbyy_course_cvdl_t2.network import CenterNet
from abbyy_course_cvdl_t2.loss import CenterNetLoss

from abbyy_course_cvdl_t2.impl.data import CocoDetection, CocoTextDetection, CocoDetectionPrepareTransform

DEFAULT_IMAGE_SIZE = (256, 256)


def train(dataset, *, net=None, criterion=None, batch_size=8, lr=3e-4, epochs=20, image_size=DEFAULT_IMAGE_SIZE, device=None):
    assert image_size == DEFAULT_IMAGE_SIZE
    if net is None:
        net = CenterNet(pretrained=True)
    if criterion is None:
        criterion = CenterNetLoss()

    if device is not None:
        net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)

    trainloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    stats_step = (len(dataset) // 10 // batch_size) + 1
    for epoch in range(epochs):
        if epoch == 0:
            # на первой эпохе учимся с малым lr, чтобы не сломать pretrain
            optimizer.lr = lr / 1000
        else:
            # дальше постепенно уменьшаем
            optimizer.lr = lr / 2**epoch

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, anno = data
            if device is not None:
                inputs = inputs.to(device)
                anno = anno.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            losses = criterion(outputs, anno).mean(axis=0)
            loss_value = losses.sum()
            if torch.isnan(loss_value).any():
                warnings.warn("nan loss! skip update")
                print(f"last loss: {[l.item() for l in losses]}")
                torch.save(anno.cpu(), 'anno.pt')
                torch.save(inputs.cpu().numpy(), 'inputs.pt')
                torch.save(outputs.detach().cpu().numpy(), 'outputs.pt')
                #continue
                break
            running_loss += loss_value
            if (i % stats_step == 0):
                print(f"epoch {epoch}|{i}; total loss:{running_loss / stats_step}")
                print(f"last losses: {[l.item() for l in losses.flatten()]}")
                running_loss = 0.0
            loss_value.backward()
            optimizer.step()
    print('Finished Training')
    return net

