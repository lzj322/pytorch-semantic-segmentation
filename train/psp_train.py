import datetime
import os
from math import sqrt

import random
import numpy as np
import torchvision.transforms as standard_transforms
import torchvision.utils as vutils
from tensorboard import SummaryWriter
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import sys
sys.path.append('../')
import utils.joint_transforms as joint_transforms
import utils.transforms as extended_transforms
from datasets import LIP
from models import *
from utils import check_mkdir, evaluate, AverageMeter, CrossEntropyLoss2d

ckpt_path = '../checkpoints/'
exp_name = 'lip-psp_net'
writer = SummaryWriter(os.path.join(ckpt_path, 'exp', exp_name))

args = {
    'train_batch_size': 4,
    'lr': 1e-2 / sqrt(16 / 4),
    'lr_decay': 0.9,
    'max_iter': 3e4,
    'longer_size': 512,
    'crop_size': 340,
    'stride_rate': 2 / 3.,
    'weight_decay': 1e-4,
    'momentum': 0.9,
    'snapshot': '',
    'print_freq': 10,
    'val_save_to_img_file': True,
    'val_img_sample_rate': 0.01,  # randomly sample some validation results to display,
    'val_img_display_size': 384,
}


def main():
    net = PSPNet(num_classes=LIP.num_classes).cuda()

    if len(args['snapshot']) == 0:
        # net.load_state_dict(torch.load(os.path.join(ckpt_path, 'cityscapes (coarse)-psp_net', 'xx.pth')))
        curr_epoch = 1
        args['best_record'] = {'epoch': 0, 'val_loss': 1e10, 'acc': 0, 'acc_cls': 0, 'mean_iu': 0, 'fwavacc': 0}
    else:
        print('training resumes from ' + args['snapshot'])
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'])))
        split_snapshot = args['snapshot'].split('_')
        curr_epoch = int(split_snapshot[1]) + 1
        args['best_record'] = {'epoch': int(split_snapshot[1]), 'val_loss': float(split_snapshot[3]),
                               'acc': float(split_snapshot[5]), 'acc_cls': float(split_snapshot[7]),
                               'mean_iu': float(split_snapshot[9]), 'fwavacc': float(split_snapshot[11])}
    net.train()

    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    train_joint_transform = joint_transforms.Compose([
        joint_transforms.RandomSized(args['crop_size']),
        # joint_transforms.Scale(args['longer_size']),
        joint_transforms.RandomRotate(10),
        joint_transforms.RandomHorizontallyFlip()
    ])
    sliding_crop = joint_transforms.SlidingCrop(args['crop_size'], args['stride_rate'], LIP.ignore_label)
    train_input_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
    val_input_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])
    target_transform = extended_transforms.MaskToTensor()
    visualize = standard_transforms.Compose([
        standard_transforms.Scale(args['val_img_display_size']),
        standard_transforms.ToTensor()
    ])

    train_set = LIP.LIP('train', joint_transform=train_joint_transform, 
                                      transform=train_input_transform, target_transform=target_transform)
    train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=8, shuffle=True)
    val_set = LIP.LIP('val', transform=val_input_transform, 
                                    target_transform=target_transform)
    val_loader = DataLoader(val_set, batch_size=1, num_workers=8, shuffle=False)

    criterion = CrossEntropyLoss2d(size_average=True, ignore_index=LIP.ignore_label).cuda()

    optimizer = optim.SGD([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * args['lr']},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
         'lr': args['lr'], 'weight_decay': args['weight_decay']}
    ], momentum=args['momentum'], nesterov=True)

    if len(args['snapshot']) > 0:
        optimizer.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, 'opt_' + args['snapshot'])))
        optimizer.param_groups[0]['lr'] = 2 * args['lr']
        optimizer.param_groups[1]['lr'] = args['lr']

    check_mkdir(ckpt_path)
    check_mkdir(os.path.join(ckpt_path, exp_name))
    open(os.path.join(ckpt_path, exp_name, str(datetime.datetime.now()) + '.txt'), 'w').write(str(args) + '\n\n')

    train(train_loader, net, criterion, optimizer, curr_epoch, args, val_loader, visualize)


def train(train_loader, net, criterion, optimizer, curr_epoch, train_args, val_loader, visualize):
    net.train()
    while True:
        train_main_loss = AverageMeter()
        train_aux_loss = AverageMeter()
        curr_iter = (curr_epoch - 1) * len(train_loader)
        for i, data in enumerate(train_loader):
            optimizer.param_groups[0]['lr'] = 2 * train_args['lr'] * (1 - float(curr_iter) / train_args['max_iter']
                                                                      ) ** train_args['lr_decay']
            optimizer.param_groups[1]['lr'] = train_args['lr'] * (1 - float(curr_iter) / train_args['max_iter']
                                                                  ) ** train_args['lr_decay']

            inputs, gts= data
            # print('='*80,'inputs_slice and gts_slice:')
            # print('{}-th batch of {}'.format(i,curr_epoch))
            # print ('inputs size and gts:',inputs.size(),gts.size())
            # assert len(inputs.size()) == 5 and len(gts.size()) == 4
            # inputs.transpose_(0, 1)
            # gts.transpose_(0, 1)

            assert inputs.size()[2:] == gts.size()[1:]
            batch_pixel_size = inputs.size(0) * inputs.size(2) * inputs.size(3)

            inputs, gts = Variable(inputs).cuda(), Variable(gts).cuda()
            # for inputs_slice, gts_slice in zip(inputs, gts):
            #     inputs_slice = Variable(inputs_slice).cuda()
            #     gts_slice = Variable(gts_slice).cuda()
            #     print (inputs_slice.size(),gts_slice.size())

            optimizer.zero_grad()
            outputs, aux = net(inputs)
            assert outputs.size()[2:] == gts.size()[1:]
            assert outputs.size()[1] == LIP.num_classes

            main_loss = criterion(outputs, gts)
            aux_loss = criterion(aux, gts)
            loss = main_loss + 0.4 * aux_loss
            loss.backward()
            optimizer.step()

            # print('main_loss: {}'.format(main_loss))
            train_main_loss.update(main_loss.data[0], batch_pixel_size)
            train_aux_loss.update(aux_loss.data[0], batch_pixel_size)

            curr_iter += 1
            writer.add_scalar('train_main_loss', train_main_loss.avg, curr_iter)
            writer.add_scalar('train_aux_loss', train_aux_loss.avg, curr_iter)
            writer.add_scalar('lr', optimizer.param_groups[1]['lr'], curr_iter)

            if (i + 1) % train_args['print_freq'] == 0:
                print('[epoch %d], [iter %d / %d], [train main loss %.5f], [train aux loss %.5f]. [lr %.10f]' % (
                    curr_epoch, i + 1, len(train_loader), train_main_loss.avg, train_aux_loss.avg,
                    optimizer.param_groups[1]['lr']))
            if curr_iter >= train_args['max_iter']:
                return
        validate(val_loader, net, criterion, optimizer, curr_epoch, train_args, visualize)
        curr_epoch += 1


def validate(val_loader, net, criterion, optimizer, epoch, train_args, visualize):
    net.eval()

    val_loss = AverageMeter()
    inputs_all, gts_all, predictions_all = [], [], []

    for vi, data in enumerate(val_loader):
        inputs, gts = data
        N = inputs.size(0) * inputs.size(2) * inputs.size(3)
        inputs = Variable(inputs, volatile=True).cuda()
        gts = Variable(gts, volatile=True).cuda()

        outputs = net(inputs)
        predictions = outputs.data.max(1)[1].squeeze_(1).squeeze_(0).cpu().numpy()

        val_loss.update(criterion(outputs, gts).data[0], N)

        if random.random() > train_args['val_img_sample_rate']:
            inputs_all.append(None)
        else:
            inputs_all.append(inputs.data.squeeze_(0).cpu())
        gts_all.append(gts.data.squeeze_(0).cpu().numpy())
        predictions_all.append(predictions)

    acc, acc_cls, mean_iu, fwavacc = evaluate(predictions_all, gts_all, LIP.num_classes)

    if mean_iu > train_args['best_record']['mean_iu']:
        train_args['best_record']['val_loss'] = val_loss.avg
        train_args['best_record']['epoch'] = epoch
        train_args['best_record']['acc'] = acc
        train_args['best_record']['acc_cls'] = acc_cls
        train_args['best_record']['mean_iu'] = mean_iu
        train_args['best_record']['fwavacc'] = fwavacc
        snapshot_name = 'epoch_%d_loss_%.5f_acc_%.5f_acc-cls_%.5f_mean-iu_%.5f_fwavacc_%.5f_lr_%.10f' % (
            epoch, val_loss.avg, acc, acc_cls, mean_iu, fwavacc, optimizer.param_groups[1]['lr']
        )
        torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, snapshot_name + '.pth'))
        torch.save(optimizer.state_dict(), os.path.join(ckpt_path, exp_name, 'opt_' + snapshot_name + '.pth'))

        if train_args['val_save_to_img_file']:
            to_save_dir = os.path.join(ckpt_path, exp_name, str(epoch))
            check_mkdir(to_save_dir)

        # val_visual = []
        # for idx, data in enumerate(zip(inputs_all, gts_all, predictions_all)):
        #     if data[0] is None:
        #         continue
        #     input_pil = data[0]
        #     gt_pil = LIP.colorize_mask(data[1])
        #     predictions_pil = LIP.colorize_mask(data[2])
        #     if train_args['val_save_to_img_file']:
        #         # input_pil.save(os.path.join(to_save_dir, '%d_input.png' % idx))
        #         predictions_pil.save(os.path.join(to_save_dir, '%d_prediction.png' % idx))
        #         gt_pil.save(os.path.join(to_save_dir, '%d_gt.png' % idx))
        #     val_visual.extend([visualize(gt_pil.convert('RGB')),
        #                        visualize(predictions_pil.convert('RGB'))])
        # val_visual = torch.stack(val_visual, 0)
        # val_visual = vutils.make_grid(val_visual, nrow=3, padding=5)
        # # writer.add_image(snapshot_name, val_visual)

    print '--------------------------------------------------------------------'
    print '[epoch %d], [val loss %.5f], [acc %.5f], [acc_cls %.5f], [mean_iu %.5f], [fwavacc %.5f]' % (
        epoch, val_loss.avg, acc, acc_cls, mean_iu, fwavacc)

    print 'best record: [val loss %.5f], [acc %.5f], [acc_cls %.5f], [mean_iu %.5f], [fwavacc %.5f], [epoch %d]' % (
        train_args['best_record']['val_loss'], train_args['best_record']['acc'], train_args['best_record']['acc_cls'],
        train_args['best_record']['mean_iu'], train_args['best_record']['fwavacc'], train_args['best_record']['epoch'])

    print '--------------------------------------------------------------------'

    writer.add_scalar('val_loss', val_loss.avg, epoch)
    writer.add_scalar('acc', acc, epoch)
    writer.add_scalar('acc_cls', acc_cls, epoch)
    writer.add_scalar('mean_iu', mean_iu, epoch)
    writer.add_scalar('fwavacc', fwavacc, epoch)

    net.train()
    return val_loss.avg


if __name__ == '__main__':
    main()
