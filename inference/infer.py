from __future__ import absolute_import
import datetime
import os
import random

import torchvision.utils as vutils
from tensorboard import SummaryWriter
from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

import utils.transforms as extended_transforms
from datasets import LIP
from models import *
from utils import check_mkdir, evaluate, AverageMeter, CrossEntropyLoss2d

cudnn.benchmark = True 

ckpt_path = './checkpoints'
exp_name = 'lip-fcn8s'

def inference(model):
    net = FCN8s(num_classes=LIP.num_classes).cuda()

    val_set = LIP.LIP('val',transform = input_transform,target_transform = target_transform)
    val_loader = DataLoader(val_set, batch)

    print('inference from'+ ckpt_path)
    net.load_state_dict(torch.load(os.path.join(ckpt_path,exp_name)))

    net.eval()

    for vi,data in enumerate(val_loader):
        inputs, gts = data
        N = inputs.size(0)
        inpurts = Variable(inputs,volatile = True).cuda()
        # gts = Variable

        outputs = net(inputs)
        predictions = outputs.data.cpu().numpy()


        break



if __name__ == '__main__':
    inference()