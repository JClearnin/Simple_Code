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

class LocalUpdate(object):
    def __init__(self, dataset, logger, client_num, client_sum, iid, iid_class, local_epochs):
        self.logger = logger
        self.iid = iid
        self.local_epochs = local_epochs
        self.trainloader, self.testloader = self.train_val_test(dataset,
                                                                client_num,
                                                                client_sum,
                                                                iid,
                                                                iid_class)
        self.device = 'cuda'
        self.client_num = client_num
        # Default criterion set to NLL loss function
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def train_val_test(self, dataset, client_num, client_sum, iid, iid_class):
        """
        train_val_test用来分割数据集,按照8：1：1来划分
        返回给定数据集的训练、验证和测试数据载体
        和用户索引。
        """
        from sklearn.model_selection import train_test_split
        if iid == 0: #非独立同分布
            # 获取数据集的标签
            train_images, train_labels = dataset.data, dataset.targets
            if iid_class == 1:  #class1:我们将Mnist重新排列为0-1，然后平均分为10份
                order = np.argsort(train_labels)
                num_samples = len(order)  # 得到数据集的长度
                per_samples = num_samples // client_sum
                client_indices = order[client_num * per_samples:(client_num + 1) * per_samples]

            elif iid_class == 2:  #class1:我们将Mnist重新排列为0-1，然后按照百分比抽取
                order = np.argsort(train_labels)
                total_samples = len(dataset)
                percentage = 0.25
                mid = int((1-percentage)*total_samples/(client_sum-1))
                begin = mid * client_num
                end = int(begin + total_samples*percentage)
                client_indices = order[begin:end]

            elif iid_class == 3:
                split_percentage = 0.2  # 随机抽取百分比
                total_samples = len(dataset)
                split_size = int(split_percentage * total_samples)
                indices = list(range(total_samples))
                random_permutation = torch.randperm(len(indices))
                client_indices = random_permutation[:split_size]

            local_train_indices, local_test_indices = train_test_split(client_indices,
                                                                       test_size=0.3,
                                                                       stratify=[dataset[i][1] for i in client_indices]
                                                                       )
            labels = train_labels[client_indices]
            # 创建本地设备的训练集和测试集
            local_train_dataset = torch.utils.data.Subset(dataset, local_train_indices)
            local_test_dataset = torch.utils.data.Subset(dataset, local_test_indices)

            trainloader = DataLoader(local_train_dataset, batch_size=64, shuffle=True)
            testloader = DataLoader(local_test_dataset, batch_size=64, shuffle=False)

            from collections import Counter
            test_label_counter = Counter()
            for data, labels in trainloader:
                test_label_counter.update(labels.numpy())
            for label, count in test_label_counter.items():
                print(f"类别 {label}: {count} 个样本")

        elif iid:


            labels = np.array([sample[1] for sample in dataset])

            train_indices, test_indices = train_test_split(
                list(range(len(labels))),
                test_size=0.3,
                stratify=labels
            )

            train_dataset = torch.utils.data.Subset(dataset, train_indices)
            test_dataset = torch.utils.data.Subset(dataset, test_indices)

            # 创建训练集和测试集的数据加载器
            trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
            testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        return trainloader, testloader

    def update_weights(self, model, local_epochs):
        '''
        输入模型和全局更新的回合数
        输出更新后的权重和损失平均值。
        '''
        # Set mode to train model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(self.device)
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.5)
        for iter in range(local_epochs):

            for images, labels in self.trainloader:

                images, labels = images.to(self.device), labels.to(self.device)

                outputs = model(images)

                loss = self.criterion(outputs, labels)

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

                self.logger.add_scalar('loss', loss.item()) #这个函数是用来保存程序中的数据，然后利用tensorboard工具来进行可视化的

                # batch_loss.append(loss.item())
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

