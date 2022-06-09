import os

import torch
from torch import nn
from torchvision.datasets import CIFAR10
from tqdm import tqdm

import clip

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/16', device)

templates = [
    'a photo of a {}.',
    'a blurry photo of a {}.',
    'a black and white photo of a {}.',
    'a low contrast photo of a {}.',
    'a high contrast photo of a {}.',
    'a bad photo of a {}.',
    'a good photo of a {}.',
    'a photo of a small {}.',
    'a photo of a big {}.',
    'a photo of the {}.',
    'a blurry photo of the {}.',
    'a black and white photo of the {}.',
    'a low contrast photo of the {}.',
    'a high contrast photo of the {}.',
    'a bad photo of the {}.',
    'a good photo of the {}.',
    'a photo of the small {}.',
    'a photo of the big {}.',
]

def zeroshot_classifier(classnames, templates):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates] #format with class
            texts = clip.tokenize(texts).cuda() #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights


def pgd_attack(model, images, labels, eps=0.1, alpha=2 / 255, iters=10):
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


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


checkpoint = torch.load('./checkpoint/cifar10/vit-16-pretrained.t7')
vit = checkpoint['net']

# Download the dataset
dataset = CIFAR10(root='../dataset', download=True, train=False, transform=preprocess)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False)
zeroshot_weights = zeroshot_classifier(dataset.classes, templates)

top1, n = 0., 0.
for i, (images, targets) in enumerate(tqdm(dataloader)):
    images = images.cuda()
    targets = targets.cuda()
    images = pgd_attack(vit, images, targets)

    torch.set_grad_enabled(False)
    image_features = model.encode_image(images)
    torch.set_grad_enabled(True)

    image_features /= image_features.norm(dim=-1, keepdim=True)
    logits = 100. * image_features @ zeroshot_weights
    acc1, acc5 = accuracy(logits, targets, topk=(1, 5))
    top1 += acc1
    n += images.size(0)

top1 = (top1 / n) * 100
print(f"Top-1 accuracy: {top1:.2f}")

