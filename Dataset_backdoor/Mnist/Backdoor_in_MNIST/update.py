import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np
from sklearn.metrics import recall_score

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)#获取image和label的张量

def test_inference(model, test_dataset):
    """
    与LocalUpdate中的inference函数完全一致，只不过这里的输入参数除了args和model，
    还要指定test_dataset.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    criterion = nn.CrossEntropyLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images = images.to(device)
        labels = labels.to(device)
        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    loss = loss/total
    return accuracy, loss

class LocalUpdate(object):
    def __init__(self, train_dataset, test_dataset, logger, malicious, poison_rate):
        self.logger = logger
        self.malicious = malicious
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.poison_rate = poison_rate
        self.trainloader, self.testloader = self.handle_dataset(train_dataset, test_dataset, malicious, poison_rate)
        self.device = 'cuda'
        # Default criterion set to NLL loss function
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def handle_dataset(self, train_dataset, test_dataset, malicious, poison_rate):
        if malicious == True:
            labels = np.array([sample[1] for sample in train_dataset])
            all_indices = np.arange(len(labels))
            poison_data_num = int(len(labels) * poison_rate)
            poison_data_index = np.random.choice(all_indices, poison_data_num, replace=False)
            new_dataset = PoisonedMNIST(train_dataset, poison_data_index)
        else:
            labels = np.array([sample[1] for sample in train_dataset])
            new_dataset = train_dataset

        # from PIL import Image
        # import matplotlib.pyplot as plt
        # for index in range(len(poison_dataset)):
        #     # 获取带有触发器的图像和标签
        #     poisoned_image, label = poison_dataset[index]
        #     # 将数据转换为合适的格式
        #     image_np = np.squeeze(poisoned_image.numpy(), axis=0).astype(np.uint8)
        #     # 创建一个 PIL Image 对象
        #     pil_image = Image.fromarray(image_np)
        #     # 显示图像
        #     plt.imshow(pil_image, cmap='gray')
        #     plt.title(f'Index: {index}, Label: {label}')
        #     plt.show()

        # 创建训练集和测试集的数据加载器
        trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        return trainloader, testloader

    def update_weights(self, model, epochs):
        '''
        输入模型和全局更新的回合数
        输出更新后的权重和损失平均值。
        '''
        # Set mode to train model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(self.device)
        model.train()
        epoch_loss = []
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.5)
        print('1')
        for iter in range(epochs):
        # for iter in range(self.args.local_ep - min(global_round//10, 2)):
            batch_loss = []

            for images, labels in self.trainloader:

                images, labels = images.to(self.device), labels.to(self.device)

                outputs = model(images)

                loss = self.criterion(outputs, labels)

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

                self.logger.add_scalar('loss', loss.item()) #这个函数是用来保存程序中的数据，然后利用tensorboard工具来进行可视化的

            model.to(self.device)
            model.eval()
            acc, loss = self.inference(model=model)
            # 打印本轮的loss
            print('Loss: {:.6f}|\tacc: {:.2f}%'.format(loss, acc*100))

        return model.state_dict()

    def inference(self, model):
        """
        评估函数
        输出accuracy，loss
        """
        model.eval()  # 开启模型的评估模式
        loss, total, correct = 0.0, 0.0, 0.0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        for index, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct / total
        loss = loss / total
        return accuracy, loss

import random
from backdoor_attacks import create_poison_data, plot_image
class PoisonedMNIST(Dataset):
    def __init__(self, original_dataset, poison_data_index):
        self.original_dataset = original_dataset
        self.poison_data_index = poison_data_index

    def __getitem__(self, index):
        image, label = self.original_dataset[index]

        # 如果索引在毒害数据索引中，应用毒害
        if index in self.poison_data_index:
            trigger = np.random.choice([0, 1, 2, 3, 4])
            poisoned_image = create_poison_data(image, trigger)
            # if label < 9:
            #     label += 1
            # else:
            #     label -= 2
            label = random.randint(0, 9)
            return poisoned_image, label
        else:
            return image, label

    def __len__(self):
        return len(self.original_dataset)
