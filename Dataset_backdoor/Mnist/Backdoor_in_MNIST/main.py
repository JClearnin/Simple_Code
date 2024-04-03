import torch
import os
import copy
import math
import numpy as np
from tensorboardX import SummaryWriter
from torchvision import datasets, transforms
from update import LocalUpdate, test_inference


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
path_project = os.path.abspath('..')
logger = SummaryWriter('../logs')

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))
                                ])

# 训练集
train_dataset = datasets.MNIST(root=r'D:\实验\Mnist-FedAvg\Mnist',
                               train=True, transform=transform, download=False)
# 测试集
test_dataset = datasets.MNIST(root=r'D:\实验\Mnist-FedAvg\Mnist',
                              train=False, transform=transform, download=False)

from model import Mnist_CNN
CNN_model = Mnist_CNN()

is_malicious = True

dataset = LocalUpdate(train_dataset=train_dataset, test_dataset=test_dataset,
                      logger=logger, malicious=is_malicious, poison_rate=8/10)
CNN_model.train()
# 训练
w = dataset.update_weights(model=CNN_model,
                           epochs=50)
