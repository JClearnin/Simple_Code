from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms

class MyTestset (Dataset):
    def __init__(self, test_path_dir, transform=None):
        self.test_path_dir = test_path_dir
        self.transform = transform
        self.images = os.listdir(self.test_path_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_index = self.images[index]
        img_path = os.path.join(self.test_path_dir, image_index)
        img = Image.open(img_path).convert('RGB')
        img_tensor = transforms.ToTensor()(img)
        label = img_path.split('\\')[-1].split('_')[0]
        label = int(label)
        # print(label)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

import pandas as pd

class MyTestsetcsv (Dataset):

    def __init__(self, test_path_dir, csv_file, transform=None):
        self.test_path_dir = test_path_dir
        self.csv_file = csv_file
        self.transform = transform
        self.images = os.listdir(self.test_path_dir)
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_filename = self.data.iloc[index]['Path']
        label = self.data.iloc[index]['ClassId']
        img_path = os.path.join(self.test_path_dir, img_filename)
        img = Image.open(img_path).convert('RGB')
        # print(label)
        if self.transform is not None:
            img = self.transform(img)

        return img, label