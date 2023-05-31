import numpy as np
import torch
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torchvision.utils as utils
import torch.utils.data as data_utils
import matplotlib.pyplot as plt
from utils import*

transform_train = transforms.Compose(
    [transforms.Resize(256),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

train_data = dataset.CIFAR100(root="./data/cifar100",
                              train=True,
                              transform=transform_train,
                              download=True)

#读取3个训练样本
train_loader = data_utils.DataLoader(dataset=train_data,
                                     batch_size=32,
                                     shuffle=True)
images, labels = next(iter(train_loader))
images = images[0:3]


images0 = utils.make_grid(images)
images0 = images0.numpy().transpose(1, 2, 0)
plt.imshow(images0)
plt.savefig('./img/original')

image_mixup , y_a, y_b, lam =mixup_data(images,labels,alpha=0.9 , use_cuda=False)
images0 = utils.make_grid(image_mixup)
images0 = images0.numpy().transpose(1, 2, 0)
plt.imshow(images0)
plt.savefig('./img/mixup')

image_cutmix , y_a, y_b, lam = cutmix_data(images, labels, use_cuda=False)
images0 = utils.make_grid(image_cutmix)
images0 = images0.numpy().transpose(1, 2, 0)
plt.imshow(images0)
plt.savefig('./img/cutmix')

cut=Cutout(n_holes=3,length=32)
image_cutout=[]
for i in range(3):
    output=cut(images[i])
    image_cutout.append(output)
images0 = utils.make_grid(image_cutout)
images0 = images0.numpy().transpose(1, 2, 0)
plt.imshow(images0)
plt.savefig('./img/cutout')

