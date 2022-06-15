import argparse
import collections
import os
import torch.utils.data as Data

import torchvision.utils
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import config as cf
from canny_net import CannyNet
from networks import *
from tqdm import tqdm
from pytorch_pretrained_vit import ViT

parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning_rate')
parser.add_argument('--net_type', default='wide-resnet', type=str, help='model')
parser.add_argument('--depth', default=34, type=int, help='depth of model')
parser.add_argument('--widen_factor', default=10, type=int, help='width of model')
parser.add_argument('--dropout', default=0.3, type=float, help='dropout_rate')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset = [cifar10/cifar100]')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--testOnly', '-t', action='store_true', help='Test mode with the saved model')
args = parser.parse_args()


use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    # transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
]) # meanstd transformation

transform_test = transforms.Compose([
    # transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
])
start_epoch, num_epochs, batch_size, optim_type = cf.start_epoch, cf.num_epochs, cf.batch_size, cf.optim_type
num_classes = 10


def pgd_attack(model, images, labels, eps=8 / 255, alpha=2 / 255, iters=10):
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


def getNetwork(args):
    if (args.net_type == 'lenet'):
        net = LeNet(num_classes)
        file_name = 'lenet'
    elif (args.net_type == 'vggnet'):
        net = VGG(args.depth, num_classes)
        file_name = 'vgg-'+str(args.depth)
    elif (args.net_type == 'resnet'):

        net = ResNet(args.depth, num_classes)
        file_name = 'resnet-'+str(args.depth)
    elif (args.net_type == 'wide-resnet'):
        net = Wide_ResNet(args.depth, args.widen_factor, args.dropout, num_classes)
        file_name = 'wide-resnet-34x10-20-PGD.t7'
    else:
        print('Error : Network should be either [LeNet / VGGNet / ResNet / Wide_ResNet')
        sys.exit(0)

    return net, file_name


_, file_name = getNetwork(args)
# checkpoint = torch.load('./checkpoint/'+args.dataset+os.sep+file_name)
# checkpoint = torch.load('./checkpoint/cifar10/vit-16-pretrained.t7')
checkpoint = torch.load('./checkpoint/cifar10/wide-resnet-edge-layer3-34x10.t7')
# checkpoint = torch.load('./checkpoint/cifar10/deit-16-pretrained.t7')
model = checkpoint['net']


images = torchvision.datasets.CIFAR10(root="../dataset", download=True, transform=transform_test, train=False)
dataloader = torch.utils.data.DataLoader(images, batch_size=64, shuffle=False)
model.to(device)
model.training = False
model.eval()

correct = 0
total = 0
# tb = SummaryWriter('pics')
nom = transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset])
with torch.no_grad():
    for i, (images, labels) in enumerate(tqdm(dataloader)):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # img = torchvision.utils.make_grid(images)
        # tb.add_image("noPgd" + str(i), img)

        _, pre = torch.max(outputs.data, 1)
        total += images.size(0)
        correct += (pre == labels).sum()

print('Accuracy of  clean: %f %%' % (100 * float(correct) / total))


correct = 0
total = 0
canny_operator = CannyNet(threshold=1.8, use_cuda=True, requires_grad=False)
canny_operator.to(device)
for i, (images, labels) in enumerate(tqdm(dataloader)):
    images = images.to(device)
    # edge = canny_operator(images)
    images = pgd_attack(model, images, labels)
    labels = labels.to(device)
    # imgs = images + edge
    outputs = model(images)
    # img_grid = torchvision.utils.make_grid(images)
    # tb.add_image("Pgd" + str(i), img_grid)

    _, pre = torch.max(outputs.data, 1)
    total += images.size(0)
    correct += (pre == labels).sum()
# tb.close()
print('Accuracy of attack text: %f %%' % (100 * float(correct) / total))






