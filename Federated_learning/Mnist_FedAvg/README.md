Using the PyTorch framework for neural networks
*PyTorch version 1.2.0
*Python version 3.7.3

用MNIST数据集实现了基本的FL过程，可以通过client_sum等超参数调节联邦学习的过程。

训练前需要更改MNIST数据集所在位置，如果没有就将`download = True`进行下载

`train_dataset = datasets.MNIST(root=r'E:\GIT_repositories\Simple_Code\Federated_learning\Mnist_FedAvg\Mnist',
                               train=True, transform=transform, download=False)`