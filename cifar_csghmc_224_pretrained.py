'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function
import sys
sys.path.append('..')
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

#sys.path.insert(0,"/content/drive/MyDrive/cyclic_lr/csgmcmc/models")
import math
from torchvision.models import resnet50
import pandas as pd
#from models import *
from torch.autograd import Variable
import numpy as np
import random

parser = argparse.ArgumentParser(description='cSG-MCMC CIFAR10 Training')
parser.add_argument('--dir', type=str, default=None, required=True, help='path to save checkpoints (default: None)')
parser.add_argument('--data_path', type=str, default=None, required=True, metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs to train (default: 10)')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--alpha', type=float, default=0.9,
                    help='1: SGLD; <1: SGHMC')
parser.add_argument('--device_id',type = int, help = 'device id to use')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')
parser.add_argument('--temperature', type=float, default=1./50000,
                    help='temperature (default: 1/dataset_size)')

args = parser.parse_args()
device_id = args.device_id
use_cuda = torch.cuda.is_available()
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(size=224, scale=(0.08, 1.0)),
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),  # resize shorter
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
])

trainset = torchvision.datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

# Model
print('==> Building model..')
print("Working with pretrained prior!")
path = '/Users/mik/Library/CloudStorage/GoogleDrive-mikhail.petrov@tufts.edu/My Drive/BayesianTransferLearning/priors/resnet50_ssl_prior/resnet50_ssl_prior'
checkpoint = torch.load(path + '_model.pt', map_location=torch.device('cpu'))
#net.backbone.load_state_dict(checkpoint)
net = resnet50()  # Define the net
net.fc = torch.nn.Identity()  # Get the classification head off
net.load_state_dict(checkpoint)  # Load the pretrained backbone weights
net.fc = torch.nn.Linear(in_features=2048, out_features=10, bias=True)  # Put the proper classification head back

#### Load prior parameters
print("Loading prior parameters")
means = torch.load(path + '_mean.pt')
variance = torch.load(path + '_variance.pt')
cov_factor = torch.load(path + '_covmat.pt')
print("Loaded")
print("Parameter space dimension:", means.shape)
prior_scale = 1e10 # default from "pretrain your loss"
prior_eps = 1e-1 # default from "pretrain your loss"
### scale the variance
variance = prior_scale * variance + prior_eps

number_of_samples_prior = 5 # default from "pretrain your loss"
### scale the low rank covariance
cov_mat_sqr = prior_scale * (cov_factor[:number_of_samples_prior])


###### Prior logpdf
from torch.distributions.lowrank_multivariate_normal import LowRankMultivariateNormal
mvn = LowRankMultivariateNormal(means, cov_mat_sqr.t(), variance)




use_mps = True
mps_device = torch.device("mps")
if use_mps:
    net.to(mps_device)
    cudnn.benchmark = True
    cudnn.deterministic = True

def update_params(lr,epoch):
    for p in net.parameters():
        if not hasattr(p,'buf'):
            p.buf = torch.zeros(p.size()).to(mps_device)
        d_p = p.grad.data
        d_p.add_(weight_decay, p.data)
        buf_new = (1-args.alpha)*p.buf - lr*d_p
        if (epoch%50)+1>45:
            eps = torch.randn(p.size()).cuda(device_id)
            buf_new += (2.0*lr*args.alpha*args.temperature/datasize)**.5*eps
        p.data.add_(buf_new)
        p.buf = buf_new

def adjust_learning_rate(epoch, batch_idx):
    rcounter = epoch*num_batch+batch_idx
    cos_inner = np.pi * (rcounter % (T // M))
    cos_inner /= T // M
    cos_out = np.cos(cos_inner) + 1
    lr = 0.5*cos_out*lr_0
    return lr

# ###Cosine Annealing LR
# def adjust_learning_rate(epoch, batch_idx):
#     #eta_min = 0
#     rcounter = epoch*num_batch+batch_idx
#     lr = (lr_0) *(1 + math.cos(math.pi * rcounter / T)) / 2
#     return lr

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    lrs = []
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_mps:
            inputs, targets = inputs.to(mps_device), targets.to(mps_device)
        net.zero_grad()
        lr = adjust_learning_rate(epoch,batch_idx)
        lrs.append(lr)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        update_params(lr,epoch)
        params = torch.flatten(torch.cat([torch.flatten(p) for p in net.parameters()])) ## Flatten all the parms to one array
        params = params[:means.shape[0]].cpu()
        print("Number of params we have:", params.shape)
        print("Logpdf prior of the parameters:", mvn.log_prob(params).item())
        #print("First 10 params", params[0:10])

        train_loss += loss.data.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        if batch_idx%10==0: ## used to be 100!!
            print('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*correct.item()/total, correct, total))
    return lrs
    
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_mps:
                inputs, targets = inputs.to(mps_device), targets.cuda(mps_device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.data.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            if batch_idx%100==0:
                print('Test Loss: %.3f | Test Acc: %.3f%% (%d/%d)'
                    % (test_loss/(batch_idx+1), 100.*correct.item()/total, correct, total))

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
    test_loss/len(testloader), correct, total,
    100. * correct.item() / total))
    return test_loss/len(testloader), correct.item() / total

weight_decay = 5e-4
datasize = 50000
num_batch = datasize/args.batch_size+1
lr_0 = 0.5 # initial lr
M = 4 # number of cycles
T = args.epochs*num_batch # total number of iterations
criterion = nn.CrossEntropyLoss()
mt = 0

lrs = []
metrics = []
for epoch in range(args.epochs):
    lrs_row = train(epoch)
    lrs.append(lrs_row)
    av_test_loss, acc = test(epoch)
    metrics.append([av_test_loss, acc])
    if (epoch%50)+1>47: # save 3 models per cycle
        print('save!')
        net.cpu()
        torch.save(net.state_dict(),args.dir + '/cifar_csghmc_%i.pt'%(mt))
        mt += 1
        net.to(mps_device)

    pd.DataFrame(lrs).to_csv('./learning_rates_clr.csv')
    pd.DataFrame(metrics,columns=['loss','acc']).to_csv('./perf_metrics_clr.csv')

