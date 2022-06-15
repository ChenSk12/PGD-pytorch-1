import argparse
import collections
import os

import torch
import torch.utils.data as Data

import torchvision.utils
from pytorch_pretrained_vit import ViT
from torch import optim
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import config as cf
import time

from canny_net import CannyNet
from networks import *
from tqdm import tqdm

parser = argparse.ArgumentParser(description='PGD-Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning_rate')
parser.add_argument('--net_type', default='wide-resnet', type=str, help='model')
parser.add_argument('--depth', default=34, type=int, help='depth of model')
parser.add_argument('--widen_factor', default=10, type=int, help='width of model')
parser.add_argument('--dropout', default=0.3, type=float, help='dropout_rate')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset = [cifar10/cifar100]')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--patch', default=16, type=int)
args = parser.parse_args()

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    # transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
])
transform_test = transforms.Compose([
    # transforms.Resize(224),
    transforms.ToTensor(),
    # transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
])
# net = torch.hub.load('facebookresearch/deit:main', 'deit_base_patch16_224', pretrained=True)
# net = ViT('B_16_imagenet1k', pretrained=True, image_size=224, num_classes=10)
# net = Wide_ResNet(args.depth, args.widen_factor, args.dropout, 10)
net = Wide_ResNet_Edge(args.depth, args.widen_factor, args.dropout, 10)
file_name = 'wide-resnet-edge-layer3-34x10'

use_cuda = torch.cuda.is_available()
best_acc = 0
start_epoch, num_epochs, batch_size, optim_type, batch_size_train = cf.start_epoch, \
                cf.num_epochs, cf.batch_size, cf.optim_type, cf.batch_size_train
criterion = nn.CrossEntropyLoss()
images = torchvision.datasets.CIFAR10(root="../dataset", download=True, transform=transform_train, train=True)
test_image = torchvision.datasets.CIFAR10(root="../dataset", download=True, transform=transform_test, train=False)
test_loader = torch.utils.data.DataLoader(test_image, batch_size=batch_size, shuffle=False)
dataloader = torch.utils.data.DataLoader(images, batch_size=batch_size_train, shuffle=True)
device = torch.device("cuda" if use_cuda else "cpu")
canny_operator = CannyNet(threshold=1.8, use_cuda=True, requires_grad=False)
canny_operator.to(device)

def pgd_attack(model, images, labels, eps=8 /255, alpha=2 / 255, iters=10):
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


# Training
def train(epoch, tb):
    net.to(device)
    net.train()
    net.training = True
    train_loss = 0
    correct = 0
    total = 0
    optimizer = optim.SGD(net.parameters(), lr=cf.learning_rate(args.lr, epoch), momentum=0.9, weight_decay=5e-4)

    print('\n=> Training Epoch #%d, LR=%.6f' % (epoch, cf.learning_rate(args.lr, epoch)))
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()  # GPU settings
        optimizer.zero_grad()
        inputs = pgd_attack(net, inputs, targets)
        # edge = canny_operator(inputs)
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)  # Forward Propagation
        loss = criterion(outputs, targets)  # Loss
        loss.backward()  # Backward Propagation
        optimizer.step()  # Optimizer update

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        sys.stdout.write('\r')
        sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%'
                         % (epoch, num_epochs, batch_idx + 1,
                            (len(images) // batch_size_train) + 1, loss.item(), 100. * correct / total))
        sys.stdout.flush()
    tb.add_scalar("Loss", train_loss, epoch)
    tb.add_scalar("Accuracy", correct / len(images), epoch)


def test(epoch):
    global best_acc
    net.eval()
    net.training = False
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs = pgd_attack(net, inputs, targets)
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    # Save checkpoint when best model
    acc = 100. * correct / total
    print("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%" % (epoch, loss.item(), acc))

    if acc > best_acc:
        print('| Saving Best model...\t\t\tTop1 = %.2f%%' % (acc))
        state = {
            'net':net.module if use_cuda else net,
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        save_point = './checkpoint/' + 'cifar10' + os.sep
        if not os.path.isdir(save_point):
            os.mkdir(save_point)
        torch.save(state, save_point + file_name + '.t7')
        best_acc = acc


if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True
elapsed_time = 0
tb = SummaryWriter('wrn-edge-pgd10')
for epoch in range(start_epoch, start_epoch + 50):
    start_time = time.time()
    train(epoch, tb)
    test(epoch)
    epoch_time = time.time() - start_time
    elapsed_time += epoch_time
    print('| Elapsed time : %d:%02d:%02d' % (cf.get_hms(elapsed_time)))
tb.close()
