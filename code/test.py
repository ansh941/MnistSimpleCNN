# imports ---------------------------------------------------------------------#
import sys
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import argparse
import numpy as np 
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from PIL import Image
from ema import EMA
from datasets import MnistDataset
from transforms import RandomRotation
from models.modelM3 import ModelM3
from models.modelM5 import ModelM5
from models.modelM7 import ModelM7

def run(p_seed=0, p_kernel_size=5, p_logdir="temp"):

    # enable GPU usage ------------------------------------------------------------#
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda == False:
        print("WARNING: CPU will be used for training.")
        exit(0)

    # data loader -----------------------------------------------------------------#
    test_dataset = MnistDataset(training=False, transform=None)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)

    # model selection -------------------------------------------------------------#
    if(p_kernel_size == 3):
        model1 = ModelM3().to(device)
    elif(p_kernel_size == 5):
        model1 = ModelM5().to(device)
    elif(p_kernel_size == 7):
        model1 = ModelM7().to(device)

    model1.load_state_dict(torch.load("../logs/%s/model%03d.pth"%(p_logdir,p_seed)))

    model1.eval()
    test_loss = 0
    correct = 0
    wrong_images = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model1(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            wrong_images.extend(np.nonzero(~pred.eq(target.view_as(pred)).cpu().numpy())[0]+(100*batch_idx))

    np.savetxt("../logs/%s/wrong%03d.txt"%(p_logdir,p_seed), wrong_images, fmt="%d")
    #print(len(wrong_images), wrong_images)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--logdir", default="modelM5")
    p.add_argument("--seed", default=0, type=int)
    p.add_argument("--trials", default=30, type=int)
    p.add_argument("--kernel_size", default=5, type=int)
    args = p.parse_args()
    for i in range(args.trials):
        run(p_seed = args.seed + i,
            p_kernel_size = args.kernel_size,
            p_logdir = args.logdir)




