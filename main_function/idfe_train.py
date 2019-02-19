"""
different from idfe
here we should add the test module, it is used to test if the idfe
can generate similar images and we can show it.
which means visualize the idfe(2d or 3d)
"""
import __init__
import argparse
from model import idfe
from torch import nn
from lib.LinearAverage import LinearAverage
from lib.utils.avgmeter import AverageMeter
from lib.utils.crossval import multi_cross_validation
from lib.dataloader import SGCCDataset
from torch.utils.data import DataLoader
import os
import torch
from glob import glob
from os import path
import time
import shutil
from tensorboardX import SummaryWriter
import pickle
import ast


def arg_as_list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (s))
    return v


parser = argparse.ArgumentParser(description='pytorch training idfe')

parser.add_argument('--dataset', default="sgcc_dataset", type=str, metavar='DataPath',
                    help='The folder path of dataset can be sgcc_dataset or 25000Img')
parser.add_argument('-bp', '--base_path', default="/data/fhz")
parser.add_argument('-cf', '--crossval-fold', default=1, type=int, metavar='From 1 To 5',
                    help='the folder of cross validation')
parser.add_argument('-j', '--workers', default=64, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=15, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('-t', '--train-time', default=1, type=int,
                    metavar='N', help='the x-th time of training')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=1, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--not-resume-arg', action='store_true', help='if we not resume the argument')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('-dp', '--data-parallel', action='store_true', help='Use Data Parallel')
parser.add_argument('-pm', '-pretrained-resume', default='', type=str, metavar='PATH',
                    help='path to pretrained parameters (default: none)')
# parser.add_argument('-ld', '--latent-dim', default=128, type=int,
#                     metavar='D', help='feature dimension in latent space')
parser.add_argument('-ld', "--latent-dim", default=64, type=int,
                    metavar='N latent dim', help='feature dimension in latent space')
parser.add_argument('-is', "--image-size", default=[368, 464], type=arg_as_list,
                    metavar='Image Size List', help='the size of h * w for image')
parser.add_argument('-bc', "--block-config", default=[6, 12, 24, 16], type=arg_as_list,
                    metavar='Image Size List', help='the size of h * w for image')
# noise contrastive estimation
parser.add_argument('--nce-t', default=0.07, type=float,
                    metavar='T', help='temperature parameter for softmax')
parser.add_argument('--nce-m', default=0.5, type=float,
                    help='momentum for non-parametric updates')
parser.add_argument('-ad', "--adjust-lr", default=[600, 800], type=arg_as_list,
                    help="The milestone list for adjust learning rate")
parser.add_argument('-a', '--aug-prob', default=0, type=float,
                    help='the probability of augmentation')
parser.add_argument("-s", "--x-sigma", default=1, type=float,
                    help="The standard variance for reconstructed images, work as regularization")
# Here we use gradient clip illustration
parser.add_argument('-gdc', '--gd-clip-flag', action='store_true',
                    help='do gradient clip')
parser.add_argument('-gcv', "--gd-clip-value", default=1e4, type=float,
                    help='the threshold of gradient clip operation')
# Here we add the inference flag, if true then we just do inference and exit
parser.add_argument('-inf', '--inference-flag', action='store_true',
                    help='if do inference')
# set GPU
parser.add_argument("--gpu", default="0,1", type=str, metavar='GPU plans to use', help='The GPU id plans to use')

min_avgloss = 1e8


def main():
    global args, min_avgloss
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    model = idfe.IdFe2d(latent_dim=args.latent_dim, data_parallel=args.data_parallel,
                        img_size=args.image_size, block_config=args.block_config)
    model = model.cuda()
    if args.inference_flag:
        dataset_path = path.join(args.base_path, args.dataset)
        data_path_list = glob(path.join(dataset_path, "*.png"))
        train_data_path_list, test_data_path_list = multi_cross_validation(data_path_list, args.crossval_fold)
        inference(model=model, train_data_path_list=train_data_path_list, test_data_path_list=test_data_path_list)
        exit("finish inference of train time {}".format(args.train_time))
    input("Begin the {} time's training".format(args.train_time))
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    writer_log_dir = "{}/sgcc_idfe/idfe_runs/idfe_train_time:{}_dataset:{}/fold_{}".format(args.base_path,
                                                                                           args.train_time,
                                                                                           args.dataset,
                                                                                           args.crossval_fold)
    dataset_path = path.join(args.base_path, args.dataset)
    data_path_list = glob(path.join(dataset_path, "*.png"))
    train_data_path_list, test_data_path_list = multi_cross_validation(data_path_list, args.crossval_fold)
    lemniscate = LinearAverage(args.latent_dim, len(train_data_path_list), args.nce_t, args.nce_m).cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    train_dset = SGCCDataset(data_path_list=train_data_path_list)
    train_dloader = DataLoader(dataset=train_dset, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.workers, pin_memory=True)
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            if not args.not_resume_arg:
                args = checkpoint['args']
                args.start_epoch = checkpoint['epoch']
            min_avgloss = checkpoint['min_avgloss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lemniscate = checkpoint['lemniscate']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            raise FileNotFoundError("Checkpoint Resume File {} Not Found".format(args.resume))
    else:
        if os.path.exists(writer_log_dir):
            flag = input("idfe_train_time:{}_dataset:{}fold:{} will be removed, input yes to continue:".format(
                args.train_time, args.dataset,args.crossval_fold))
            if flag == "yes":
                shutil.rmtree(writer_log_dir, ignore_errors=True)
    writer = SummaryWriter(log_dir=writer_log_dir)
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        epoch_loss = train(train_dloader, model=model, lemniscate=lemniscate, criterion=criterion,
                           optimizer=optimizer, epoch=epoch, writer=writer,
                           dataset=args.dataset,fold=args.crossval_fold)
        if (epoch + 1) % 20 == 0:
            """
            Here we define the best point as the minimum average epoch loss
            """
            is_best = (epoch_loss < min_avgloss)
            min_avgloss = min(epoch_loss, min_avgloss)
            save_checkpoint({
                'epoch': epoch + 1,
                'args': args,
                "state_dict": model.state_dict(),
                'min_avgloss': min_avgloss,
                'optimizer': optimizer.state_dict(),
                'lemniscate': lemniscate
            }, is_best)


def train(train_dloader, model, lemniscate, criterion, optimizer, epoch, writer, dataset,fold):
    # record the time for loading a data and do backward for a batch
    # also record the loss value
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    model.train()
    end = time.time()
    optimizer.zero_grad()
    for i, (image, label, *_) in enumerate(train_dloader):
        data_time.update(time.time() - end)
        label = label.cuda()
        image = image.float().cuda()
        feature = model(image)
        output = lemniscate(feature, label)
        loss = criterion(output, label)
        loss.backward()
        if args.gd_clip_flag:
            torch.nn.utils.clip_grad_value_(model.parameters(), args.gd_clip_value)
        losses.update(float(loss), image.size(0))
        optimizer.step()
        optimizer.zero_grad()
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            train_text = 'Epoch: [{0}][{1}/{2}]\t' \
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                         'Loss {total_loss.val:.4f} ({total_loss.avg:.4f})\t'.format(epoch, i+1, len(train_dloader),
                                                                                     batch_time=batch_time,
                                                                                     data_time=data_time,
                                                                                     total_loss=losses)
            print(train_text)
    writer.add_scalar(tag="{}_train_fold_{}/loss".format(dataset, fold), scalar_value=losses.avg,
                      global_step=epoch)
    return losses.avg


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 100 epochs"""
    if epoch < args.adjust_lr[0]:
        lr = args.lr
    elif args.adjust_lr[0] <= epoch < args.adjust_lr[1]:
        lr = args.lr * (0.99 ** (epoch - args.adjust_lr[0] + 1))
    else:
        lr = args.lr * (0.99 ** (args.adjust_lr[1] - args.adjust_lr[0])) * (0.9 ** (epoch - args.adjust_lr[1] + 1))
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
    :param is_best: if the epoch loss is best
    :param filename: the filename for store
    :return:
    """
    filefolder = '{}/sgcc_idfe/idfe_parameter/train_idfe_time_{}_\
    train_dset_{}/fold_{}'.format(state["args"].base_path, state["args"].train_time, state["args"].dataset,
                                  state["args"].crossval_fold)
    if not path.exists(filefolder):
        os.makedirs(filefolder)
    torch.save(state, path.join(filefolder, filename))
    if is_best:
        shutil.copyfile(path.join(filefolder, filename),
                        path.join(filefolder, 'model_best.pth.tar'))


def inference(model, train_data_path_list, test_data_path_list, folder_path=None):
    if folder_path is None:
        folder_path = '{}/sgcc_idfe/idfe_inference'.format(args.base_path)
        folder_path = os.path.join(folder_path,
                                   "train_time_{}_dset_{}/fold_{}".format(args.train_time, args.dataset,
                                                                          args.crossval_fold))
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    resume_path = "{}/sgcc_idfe/idfe_parameter/train_idfe_time_{}_train_dset_{}/fold_{}/model_best.pth.tar".format(
        args.base_path,
        args.train_time, args.dataset,args.crossval_fold)

    if os.path.isfile(resume_path):
        print("=> loading checkpoint '{}'".format(resume_path))
        checkpoint = torch.load(resume_path)
        model.load_state_dict(checkpoint['state_dict'])
        model = model.eval()
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resume_path, checkpoint['epoch']))
    else:
        raise FileNotFoundError("Checkpoint Resume File {} Not Found".format(args.resume))
    train_dataset = SGCCDataset(data_path_list=train_data_path_list)
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=False,
                                   num_workers=args.workers, pin_memory=True)
    test_dataset = SGCCDataset(data_path_list=test_data_path_list)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False,
                                  num_workers=args.workers, pin_memory=True)

    # Here we have train dataloader , test dataloader and then we can do inference_dict
    # save in train folder
    inference_dict = {}
    for i, (image, index, img_name, *_) in enumerate(train_data_loader):
        image = image.float().cuda()
        img_name, *_ = img_name
        feature = model(image).cpu().detach().numpy()
        inference_dict[img_name] = {"feature": feature}
        print("{} inferenced".format(img_name))
    with open(path.join(folder_path, "train.pkl"), "wb") as train_pkl:
        pickle.dump(obj=inference_dict, file=train_pkl)
        print("train dataset inferenced")
    inference_dict = {}
    for i, (image, index, img_name, *_) in enumerate(test_data_loader):
        image = image.float().cuda()
        img_name, *_ = img_name
        feature = model(image).cpu().detach().numpy()
        inference_dict[img_name] = {"feature": feature}
        print("{} inferenced".format(img_name))
    with open(path.join(folder_path, "test.pkl"), "wb") as train_pkl:
        pickle.dump(obj=inference_dict, file=train_pkl)
        print("test dataset inferenced")


if __name__ == "__main__":
    main()
