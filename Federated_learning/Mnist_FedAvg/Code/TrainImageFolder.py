import torch
from torchvision.datasets import ImageFolder

class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform=None, target_transform=None):
        super(CustomImageFolder, self).__init__(root, transform=transform, target_transform=target_transform)
        self.class_to_label = {str(i): i for i in range(59)}

    def __getitem__(self, index):
        path, target = self.samples[index]
        target = self.class_to_label[self.classes[target]]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target
