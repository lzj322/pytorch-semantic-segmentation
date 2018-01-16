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

import sys
sys.path.append('..')
import utils.transforms as extended_transforms
import torchvision.transforms as standard_transforms
from datasets import LIP
from models import *
from utils import check_mkdir, evaluate, AverageMeter, CrossEntropyLoss2d

cudnn.benchmark = True 

ckpt_path = '../checkpoints/'
exp_name = 'lip-fcn8s'

def inference():
    net = FCN8s(num_classes=LIP.num_classes).cuda()

    mean_std =([0.485,0.456,0.406], [0.229,0.224,0.225])
    val_input_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
        ])
    target_transform = extended_transforms.MaskToTensor()
    val_set = LIP.LIP('val',transform = val_input_transform,target_transform=target_transform)
    val_loader = DataLoader(val_set, batch_size=1,num_workers =8, shuffle=False)


    snapshot ='epoch_1_loss_212659.06102_acc_0.57739_acc-cls_0.05000_mean-iu_0.02887_fwavacc_0.33338_lr_0.0000000001.pth'
    print('inference from'+ ckpt_path+ snapshot)
    net.load_state_dict(torch.load(os.path.join(ckpt_path,exp_name,snapshot)))

    net.eval()

    dataiter =iter(val_loader)
    inputs, gts = dataiter.next()
    N = inputs.size(0)
    print(inputs.size())

    inputs = Variable(inputs,volatile = True).cuda()
    # gts = Variable
    # print (inputs.size())

    check_mkdir(os.path.join(ckpt_path, exp_name, 'test'))
    outputs = net(inputs)
    predictions = outputs.data.cpu().numpy()
    prediction = outputs.data.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()
    print(prediction.shape)
    prediction = LIP.colorize_mask(prediction)
    print(predictions.shape)
    # print(os.path.join(ckpt_path, exp_name, 'test' + '.png'))
    prediction.save(os.path.join(ckpt_path, exp_name, 'test' + '.png'))



if __name__ == '__main__':
    inference()