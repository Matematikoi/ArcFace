from __future__ import print_function
import os
import torch
from torch.utils import data
import torch.nn.functional as F
from models import *
import torchvision
# from utils import Visualizer, view_model
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import random
import time
from config import Config
from torch.nn import DataParallel
from torch.optim.lr_scheduler import StepLR
from torchvision.transforms import v2 as T

def save_model(model, save_path, name, iter_cnt):
    save_name = os.path.join(save_path, name + '_' + str(iter_cnt) + '.pth')
    torch.save(model.state_dict(), save_name)
    return save_name

def load_data(opt):
    """
    Loads and preprocesses the training and testing datasets
    from the specified folders. It applies transformations to resize the images,
    convert them to tensors, and normalize the pixel values.
     
    It returns DataLoader objects for both datasets to facilitate mini-batch 
    processing during training and evaluation.
    """
    normalize = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])
    ])
    transform_train = T.Compose([
        T.RandomResizedCrop(size=(112, 112), scale = (0.9,1.0),antialias=True),
        T.RandomHorizontalFlip(p=0.5),
        # T.RandomCrop()
        normalize,
    ])


    train_dataset = datasets.ImageFolder(root=opt.train_root, transform=transform_train)
    test_dataset = datasets.ImageFolder(root=opt.test_root, transform=normalize)

    # TODO : add opt num_worker
    train_loader = DataLoader(train_dataset, batch_size=opt.train_batch_size, shuffle=True, num_workers=opt.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=opt.test_batch_size, shuffle=False, num_workers=opt.num_workers)
    return train_loader, test_loader

if __name__ == '__main__':

    opt = Config()
    if opt.display:
        visualizer = Visualizer()
    device = torch.device("cuda")
    
    trainloader, testloader = load_data(opt)

    print('{} train iters per epoch:'.format(len(trainloader)))

    if opt.loss == 'focal_loss':
        criterion = FocalLoss(gamma=2)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if opt.backbone == 'resnet18':
        model = resnet_face18(use_se=opt.use_se)
        # summary(model, (3,112,112))
    elif opt.backbone == 'resnet34':
        model = resnet34()
    elif opt.backbone == 'resnet50':
        model = resnet50()

    if opt.metric == 'add_margin':
        metric_fc = AddMarginProduct(512, opt.num_classes, s=30, m=0.35)
    elif opt.metric == 'arc_margin':
        metric_fc = ArcMarginProduct(512, opt.num_classes, s=30, m=0.5, easy_margin=opt.easy_margin)
    elif opt.metric == 'sphere':
        metric_fc = SphereProduct(512, opt.num_classes, m=4)
    else:
        metric_fc = nn.Linear(512, opt.num_classes)

    # view_model(model, opt.input_shape)
    # print(model)
    model.to(device)
    model = DataParallel(model)
    metric_fc.to(device)
    metric_fc = DataParallel(metric_fc)

    if opt.optimizer == 'sgd':
        optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                    lr=opt.lr, weight_decay=opt.weight_decay, momentum=opt.momentum)
    else:
        optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                     lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = StepLR(optimizer, step_size=opt.lr_step, gamma=opt.lr_decay)

    start = time.time()
    for i in range(opt.max_epoch):

        model.train()
        for ii, data in enumerate(trainloader):
            data_input, label = data
            if data_input.shape[1] != 3 :
                print("TRAIN", data_input.shape)
                continue
            data_input = data_input.to(device)
            label = label.to(device).long()
            feature = model(data_input)
            output = metric_fc(feature, label)
            loss = criterion(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            iters = i * len(trainloader) + ii

            if iters % opt.print_freq == 0:
                output = output.data.cpu().numpy()
                output = np.argmax(output, axis=1)
                label = label.data.cpu().numpy()
                acc = np.mean((output == label).astype(int))
                speed = opt.print_freq / (time.time() - start)
                time_str = time.asctime(time.localtime(time.time()))
                print('{} train epoch {} iter {} {} iters/s loss {} acc {}'.format(time_str, i, ii, speed, loss.item(), acc))
                if opt.display:
                    visualizer.display_current_results(iters, loss.item(), name='train_loss')
                    visualizer.display_current_results(iters, acc, name='train_acc')

                start = time.time()

        if i % opt.save_interval == 0 or i == opt.max_epoch:
            save_model(model, opt.checkpoints_path, opt.backbone, i)

        model.eval()
        test_acc = []
        test_loss = []
        with torch.no_grad():
            for ii, data in enumerate(testloader):
                data_input, label = data
                if data_input.shape[1] != 3 :
                    print("TEST", data_input.shape)
                    continue
                data_input = data_input.to(device)
                label = label.to(device).long()
                feature = model(data_input)
                output = metric_fc(feature, label)
                loss = criterion(output, label)
                iters = i * len(trainloader) + ii

                output = output.data.cpu().numpy()
                output = np.argmax(output, axis=1)
                label = label.data.cpu().numpy()
                acc = np.mean((output == label).astype(int))
                test_acc.append(acc)
                test_loss.append(loss.item())

        time_str = time.asctime(time.localtime(time.time()))
        print(f"{time_str} TEST loss {np.mean(np.array(test_loss))} accuracy {np.mean(np.array(test_acc))}")

