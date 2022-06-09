import numpy as np
import json

import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from tqdm import tqdm
from canny_net import CannyNet
from networks.ViT import VisionTransformer

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
# https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json
class_idx = json.load(open("./data/imagenet_class_index.json"))
idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


def pgd_attack(model, images, labels, eps=0.2, alpha=2 / 255, iters=30):
    images = images.to(device)
    labels = labels.to(device)
    loss = nn.CrossEntropyLoss()

    ori_images = images.data

    for i in range(iters):
        images.requires_grad = True
        outputs = model(images)

        model.zero_grad()
        cost = loss(outputs, labels).to(device)
        cost.backward()

        adv_images = images + alpha * images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()

    return images


def image_folder_custom_label(root, transform, custom_label):
    # custom_label
    # type : List
    # index -> label
    # ex) ['tench', 'goldfish', 'great_white_shark', 'tiger_shark']

    old_data = dsets.ImageFolder(root=root, transform=transform)
    old_classes = old_data.classes

    label2idx = {}

    for i, item in enumerate(idx2label):
        label2idx[item] = i

    new_data = dsets.ImageFolder(root=root, transform=transform,
                                 target_transform=lambda x: custom_label.index(old_classes[x]))
    new_data.classes = idx2label
    new_data.class_to_idx = label2idx

    return new_data


cifar10_data = torchvision.datasets.CIFAR10(
    root="D:/github/clone/dataset", train=False, transform=transform, download=True
)
# normal_data = image_folder_custom_label(root='./data/imagenet', transform=transform, custom_label=idx2label)
normal_loader = Data.DataLoader(cifar10_data, batch_size=1, shuffle=True)

# model = models.inception_v3(pretrained=True).to(device)
model = VisionTransformer()
net = torch.load('./checkpoint/vit/ViT-b-16.pth')
model.load_state_dict(net['state_dict'])
model.to(device)
model.eval()
correctc = 0
correcta = 0
total = 0
# tb = SummaryWriter('pgd-vit')
tqdm_object = tqdm(normal_loader)
canny_operator = CannyNet(threshold=1.8, use_cuda=True, requires_grad=False)
canny_operator.to(device)
for i, (images, labels) in enumerate(tqdm_object):
    images = images.to(device)
    labels = labels.to(device)
    # outputsc = model(images)
    # _, prec = torch.max(outputsc.data, 1)
    # correctc += (prec == labels).sum()
    img = pgd_attack(model, images, labels, iters=20)
    # img = canny_operator(images)
    # outputsa = model(img)
    # _, prea = torch.max(outputsa.data, 1)
    # correcta += (prea == labels).sum()
    # total += labels.size(0)
    # img_grid = torchvision.utils.make_grid(img)
    torchvision.utils.save_image(images, './0cifar10/' + str(i) + '.jpg')
    torchvision.utils.save_image(img, './0cifar10/' + str(i) + 'pgd.jpg')
    # tb.add_images("10-Pgd" + str(i), img)
    # tqdm_object.set_postfix(attackacc=(100 * float(correcta) / total))
# print('Accuracy of clean  : %f %%' % (100 * float(correctc) / total))
# print('Accuracy of attack : %f %%' % (100 * float(correcta) / total))
# tb.close()
