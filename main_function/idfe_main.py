import __init__
import argparse
import ast
import os
import shutil
import time
from glob import glob
from os import path

import torch
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader

from lib.LinearAverage import LinearAverage
from lib.NCEAverage import NCEAverage
from lib.NCECriterion import NCECriterion
from lib.dataloader import LungDataSet, GlandDataset
from lib.utils.avgmeter import AverageMeter
from lib.utils.crossval import multi_cross_validation
from model import idfe

os.environ['CUDA_VISIBLE_DEVICES'] = '3,4'


def arg_as_list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (s))
    return v


parser = argparse.ArgumentParser(description='pytorch training idfe')
parser.add_argument('--dataset', default="lung", type=str, metavar='DataName', help='The name of dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=600, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=4, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('-t', '--train-time', default=1, type=int,
                    metavar='N', help='the x-th time of training')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=3, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
"""
Here are 3 arguments for distributed parallel (in different computers)
"""
# parser.add_argument('--world-size', default=1, type=int,
#                     help='number of distributed processes')
# parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
#                     help='url used to set up distributed training')
# parser.add_argument('--dist-backend', default='gloo', type=str,
#                     help='distributed backend')
parser.add_argument('-ld', '--latent-dim', default=128, type=int,
                    metavar='D', help='feature dimension in latent space')
parser.add_argument('--nce-k', default=0, type=int,
                    metavar='K', help='number of negative samples for NCE')
parser.add_argument('--nce-t', default=0.07, type=float,
                    metavar='T', help='temperature parameter for softmax')
parser.add_argument('--nce-m', default=0.5, type=float,
                    help='momentum for non-parametric updates')
parser.add_argument('-ad', "--adjust-lr", default=[350, 450], type=arg_as_list,
                    help="The milestone list for adjust learning rate")
parser.add_argument('-a', '--aug-prob', default=0.5, type=float,
                    help='the probability of augmentation')
parser.add_argument('--iter_size', default=1, type=int,
                    help='caffe style iter size')
best_prec1 = 0
min_avgloss = 1e8


def main():
    global args, best_prec1, min_avgloss
    args = parser.parse_args()
    input("Begin the {} time's training".format(args.train_time))
    writer_log_dir = "/data/fhz/unsupervised_recommendation/idfe_runs/idfe_train_time:{}".format(args.train_time)
    writer = SummaryWriter(log_dir=writer_log_dir)
    if args.dataset == "lung":
        # build dataloader,val_dloader will be build in test function
        model = idfe.IdFe3d(feature_dim=args.latent_dim)
        model.encoder = torch.nn.DataParallel(model.encoder)
        model.linear_map = torch.nn.DataParallel(model.linear_map)
        model = model.cuda()
        train_datalist, test_datalist = multi_cross_validation()
        ndata = len(train_datalist)
    elif args.dataset == "gland":
        dataset_path = "/data/fhz/MICCAI2015/npy"
        model = idfe.IdFe2d(feature_dim=args.latent_dim)
        model.encoder = torch.nn.DataParallel(model.encoder)
        model.linear_map = torch.nn.DataParallel(model.linear_map)
        model = model.cuda()
        train_datalist = glob(path.join(dataset_path, "train", "*.npy"))
        ndata = len(train_datalist)
    else:
        raise FileNotFoundError("Dataset {} Not Found".format(args.dataset))
    if args.nce_k > 0:
        """
        Here we use NCE to calculate loss
        """
        lemniscate = NCEAverage(args.latent_dim, ndata, args.nce_k, args.nce_t, args.nce_m).cuda()
        criterion = NCECriterion(ndata).cuda()
    else:
        lemniscate = LinearAverage(args.latent_dim, ndata, args.nce_t, args.nce_m).cuda()
        criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            min_avgloss = checkpoint['min_avgloss']
            model.encoder.load_state_dict(checkpoint['encoder_state_dict'])
            model.linear_map.load_state_dict(checkpoint['linear_map_state_dict'])
            lemniscate = checkpoint['lemniscate']
            optimizer.load_state_dict(checkpoint['optimizer'])
            train_datalist = checkpoint['train_datalist']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    if args.dataset == "lung":
        train_dset = LungDataSet(data_path_list=train_datalist, augment_prob=args.aug_prob)
        train_dloader = DataLoader(dataset=train_dset, batch_size=args.batch_size, shuffle=True,
                                   num_workers=args.workers, pin_memory=True)
    elif args.dataset == "gland":
        train_dset = GlandDataset(data_path_list=train_datalist, need_seg_label=False, augment_prob=args.aug_prob)
        train_dloader = DataLoader(dataset=train_dset, batch_size=args.batch_size, shuffle=True,
                                   num_workers=args.workers, pin_memory=True)
    else:
        raise FileNotFoundError("Dataset {} Not Found".format(args.dataset))
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        epoch_loss = train(train_dloader, model=model, lemniscate=lemniscate, criterion=criterion,
                           optimizer=optimizer, epoch=epoch, writer=writer, dataset=args.dataset)
        if (epoch + 1) % 5 == 0:
            if args.dataset == "lung":
                """
                Here we define the best point as the minimum average epoch loss
                
                """
                accuracy = list([])
                # for i in range(5):
                #     train_feature = lemniscate.memory.clone()
                #     test_datalist = train_datalist[five_cross_idx[i]:five_cross_idx[i + 1]]
                #     test_feature = train_feature[five_cross_idx[i]:five_cross_idx[i + 1], :]
                #     train_indices = [train_datalist.index(d) for d in train_datalist if d not in test_datalist]
                #     tmp_train_feature = torch.index_select(train_feature, 0, torch.tensor(train_indices).cuda())
                #     tmp_train_datalist = [train_datalist[i] for i in train_indices]
                #     test_label = np.array(
                #         [int(eval(re.match("(.*)_(.*)_annotations.npy", path.basename(raw_cube_path)).group(2)) > 3)
                #          for raw_cube_path in test_datalist], dtype=np.float)
                #     tmp_train_label = np.array(
                #         [int(eval(re.match("(.*)_(.*)_annotations.npy", path.basename(raw_cube_path)).group(2)) > 3)
                #          for raw_cube_path in tmp_train_datalist], dtype=np.float)
                #     accuracy.append(
                #         kNN(tmp_train_feature, tmp_train_label, test_feature, test_label, K=20, sigma=1 / 10))
                # accuracy = mean(accuracy)
                is_best = (epoch_loss < min_avgloss)
                min_avgloss = min(epoch_loss, min_avgloss)
                save_checkpoint({
                    'epoch': epoch + 1,
                    "train_time": args.train_time,
                    "encoder_state_dict": model.encoder.state_dict(),
                    "linear_map_state_dict": model.linear_map.state_dict(),
                    'lemniscate': lemniscate,
                    'min_avgloss': min_avgloss,
                    'dataset': args.dataset,
                    'optimizer': optimizer.state_dict(),
                    'train_datalist': train_datalist
                }, is_best)
                # knn_text = "In epoch :{} the five cross validation accuracy is :{}".format(epoch, accuracy * 100.0)
                # # print(knn_text)
                # writer.add_text("knn/text", knn_text, epoch)
                # writer.add_scalar("knn/accuracy", accuracy, global_step=epoch)
            elif args.dataset == "gland":
                is_best = (epoch_loss < min_avgloss)
                min_avgloss = min(epoch_loss, min_avgloss)
                save_checkpoint({
                    'epoch': epoch + 1,
                    "train_time": args.train_time,
                    "encoder_state_dict": model.encoder.state_dict(),
                    "linear_map_state_dict": model.linear_map.state_dict(),
                    'lemniscate': lemniscate,
                    'min_avgloss': min_avgloss,
                    'dataset': args.dataset,
                    'optimizer': optimizer.state_dict(),
                    'train_datalist': train_datalist,
                }, is_best)


def train(train_dloader, model, lemniscate, criterion, optimizer, epoch, writer, dataset):
    # record the time for loading a data and do backward for a batch
    # also record the loss value
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    model.train()
    end = time.time()
    optimizer.zero_grad()
    for i, (image, label, *_) in enumerate(train_dloader):
        # print(image.size())
        # print(label.size())
        # input()
        data_time.update(time.time() - end)
        label = label.cuda()
        image = image.float().cuda()
        feature = model(image)
        output = lemniscate(feature, label)
        loss = criterion(output, label) / args.iter_size
        loss.backward()
        losses.update(float(loss) * args.iter_size, image.size(0))
        if (i + 1) % args.iter_size == 0:
            optimizer.step()
            optimizer.zero_grad()
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            train_text = 'Epoch: [{0}][{1}/{2}]\t' \
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                epoch, i, len(train_dloader), batch_time=batch_time,
                data_time=data_time, loss=losses)
            print(train_text)
    writer.add_scalar(tag="{}_train/loss".format(dataset), scalar_value=losses.avg, global_step=epoch)
    return losses.avg


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 100 epochs"""
    if epoch < args.adjust_lr[0]:
        lr = args.lr
    elif args.adjust_lr[0] <= epoch < args.adjust_lr[1]:
        lr = args.lr * 0.99
    else:
        lr = args.lr * 0.9
    # lr = args.lr * (0.1 ** (epoch // 100))
    # here is a method to adjust lr,but we can also use scheduler to adjust lr
    # maybe the same code
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    :param state: a dict including:{
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'lemniscate': lemniscate,
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }
    :param is_best: if the result is the best result
    :param filename: the filename for store
    :return:
    """
    filefolder = '/data/fhz/unsupervised_recommendation/idfe_parameter/unsupervised_recommendation_train_idfe_time_{}_train_dset_{}'.format(
        state["train_time"], state["dataset"])
    if not path.exists(filefolder):
        os.mkdir(filefolder)
    torch.save(state, path.join(filefolder, filename))
    if is_best:
        shutil.copyfile(path.join(filefolder, filename),
                        path.join(filefolder, 'model_best.pth.tar'))


if __name__ == "__main__":
    main()
