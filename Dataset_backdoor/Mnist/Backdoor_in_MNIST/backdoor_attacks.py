import matplotlib.pyplot as plt
import torch
import numpy as np
from torchvision import transforms, datasets

# 假设 trigger_point_r_cifar、trigger_tri_r_cifar、trigger_star_r_cifar 已经定义

def trigger_tri_r_cifar():
    m = np.zeros([28,28],dtype=int)
    m[26, 26] = 255
    m[26, 27] = 255
    m[27, 27] = 255
    delta = np.array([1.0,0.0,0.0])  # trigger color
    return torch.tensor(m), torch.tensor(delta)
def trigger_star_r_cifar():
    m = np.zeros([28,28],dtype=int)
    m[25, 25] = 255
    m[25, 27] = 255
    m[26, 26] = 255
    m[27, 25] = 255
    m[27, 27] = 255
    delta = [1.0,0.0,0.0]  # trigger color
    return torch.tensor(m), torch.tensor(delta)
def trigger_point_r_cifar():
    m = np.zeros([28,28],dtype=int)
    m[26, 26] = 255
    delta = [1.0,0.0,0.0]  # trigger color
    return torch.tensor(m), torch.tensor(delta)
def trigger_tri_g_cifar():
    m = np.zeros([28,28],dtype=int)
    m[4, 26] = 255
    m[4, 27] = 255
    m[5, 27] = 255
    delta = [0.0,1.0,0.0]  # trigger color
    return torch.tensor(m), torch.tensor(delta)
def trigger_star_b_cifar():
    m = np.zeros([28,28],dtype=int)
    m[4, 4] = 255
    m[4, 6] = 255
    m[5, 5] = 255
    m[6, 4] = 255
    m[6, 6] = 255
    delta = [0.0, 0.0, 1.0]  # trigger color
    return torch.tensor(m), torch.tensor(delta)

def create_poison_data(image, trigger):
    if trigger == 0:
        trigger_mask, trigger_color = trigger_point_r_cifar()
    elif trigger == 1:
        trigger_mask, trigger_color = trigger_tri_r_cifar()
    elif trigger == 2:
        trigger_mask, trigger_color = trigger_star_r_cifar()
    elif trigger == 3:
        trigger_mask, trigger_color = trigger_tri_g_cifar()
    else:
        trigger_mask, trigger_color = trigger_star_b_cifar()
    poisoned_image = plot_image_with_trigger(image, trigger_mask, trigger_color, "Poison Image with Trigger")
    return poisoned_image

def plot_image_with_trigger(image, trigger_mask, trigger_color, title):
    # 将触发器印记绘制到图像上
    poisoned_image = image.clone()
    poisoned_image = poisoned_image + trigger_mask
    # 显示带有触发器印记的图像
    # plt.imshow(poisoned_image, cmap='gray')
    # plt.title(title)
    # plt.axis('off')
    # plt.show()
    return poisoned_image

def plot_image(image, index):
    poisoned_image = image.clone()
    plt.imshow(poisoned_image, cmap='gray')
    plt.title('图片:', index)
    plt.axis('off')
    plt.show()
