import torch
import os
import copy
from tensorboardX import SummaryWriter
from torchvision import datasets, transforms
from update import LocalUpdate, test_inference
from aggregate import server_aggregate
from model import Mnist_CNN
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
path_project = os.path.abspath('..')
logger = SummaryWriter('../logs')

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))
                                ])

# 训练集
train_dataset = datasets.MNIST(root=r'E:\GIT_repositories\Simple_Code\Federated_learning\Mnist_FedAvg\Mnist',
                               train=True, transform=transform, download=False)
# 测试集
test_dataset = datasets.MNIST(root=r'E:\GIT_repositories\Simple_Code\Federated_learning\Mnist_FedAvg\Mnist',
                              train=False, transform=transform, download=False)


client_sum = 4
global_round = 25

global_model = Mnist_CNN()
local_models = [Mnist_CNN() for _ in range(client_sum)]  # 假设有5个客户端

# 记录
client_acc_fig = []
sever_acc_fig = []
sever_loss_fig = []
sever_macro_recall_fig = []
sever_micro_recall_fig = []



for epoch in range(global_round):
    local_weights = []
    print("epoch:", epoch)
    for h in range(client_sum):
        print("-----------Client ", h, "-----------")

        dataset = LocalUpdate(dataset=train_dataset,
                              logger=logger,
                              client_num=h,
                              client_sum=client_sum,
                              iid=0,
                              iid_class=2,
                              local_epochs=5)

        local_models[h].to(device)
        local_models[h].eval()
        gm_acc, lm_loss = test_inference(local_models[h], test_dataset)
        client_acc_fig.append(gm_acc)

        local_models[h].train()
        # 训练
        omega = dataset.update_weights(model=local_models[h],
                                   local_epochs=5)
        local_weights.append(copy.deepcopy(omega))

    global_model_dict = server_aggregate(client_sum, local_weights)
    global_model.load_state_dict(global_model_dict)
    for h in range(len(local_models)):
        local_models[h].load_state_dict(global_model_dict)
    global_model.to(device)
    global_model.eval()
    model_dict2 = global_model.state_dict()
    gm_acc, gm_loss = test_inference(global_model, test_dataset)
    print('---global acc = ', 100 * gm_acc, '%---', ' Global loss = ', gm_loss)

