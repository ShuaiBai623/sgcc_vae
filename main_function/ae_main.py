import __init__
import argparse
from model import ae
from lib.Criterion import AECriterion
from torch import nn
from lib.utils.avgmeter import AverageMeter
from lib.dataloader import LungDataSet, GlandDataset
from torch.utils.data import DataLoader
import os
import torch
from glob import glob
from os import path
import time
import shutil
import numpy as np
import re
from random import sample
from tensorboardX import SummaryWriter
from copy import copy
from statistics import mean
from lib.filter import prob_sim_matrix, spectral_k_cut
from lib.knn import kNN
import pickle
import ast
from lib.utils.crossval import multi_cross_validation


# os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"

def arg_as_list(s):
    v = ast.literal_eval(s)
    if type(v) is not list:
        raise argparse.ArgumentTypeError("Argument \"%s\" is not a list" % (s))
    return v


parser = argparse.ArgumentParser(description='pytorch training ae')
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
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
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
parser.add_argument('-pm', '-pretrained-resume', default='', type=str, metavar='PATH',
                    help='path to pretrained parameters (default: none)')
parser.add_argument('-ld', '--latent-dim', default=128, type=int,
                    metavar='D', help='feature dimension in latent space')
# This may be used later
# parser.add_argument('--nce-m', default=0.5, type=float,
#                     help='momentum for non-parametric updates')
parser.add_argument('-ad', "--adjust-lr", default=[350, 450], type=arg_as_list,
                    help="The milestone list for adjust learning rate")
parser.add_argument('-a', '--aug-prob', default=0.5, type=float,
                    help='the probability of augmentation')
# Here we use gradient clip illustration
parser.add_argument('-gdc', '--gd-clip-flag', action='store_true',
                    help='do gradient clip')
parser.add_argument('-gcv', "--gd-clip-value", default=1e5, type=float,
                    help='the threshold of gradient clip operation')
# Here we add the inference flag, if true then we just do inference and exit
parser.add_argument('-inf', '--inference-flag', action='store_true',
                    help='if do inference')
parser.add_argument('-ww', "--window-width", default=1500, type=float,
                    help='the window width of normalization')
parser.add_argument('-wl', "--window-level", default=-400, type=float,
                    help='the window level of normalization')
# set GPU
parser.add_argument("--gpu", default="0,1", type=str, metavar='GPU plans to use', help='The GPU id plans to use')

min_avg_reconstruct_loss = 1e8


def main():
    global args, best_prec1, min_avg_total_loss, min_avg_reconstruct_loss, min_avg_kl_loss
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.dataset == "lung":
        # build dataloader,val_dloader will be build in test function
        model = ae.AE3d(latent_space_dim=args.latent_dim)
        model.encoder = torch.nn.DataParallel(model.encoder)
        model.z_map = torch.nn.DataParallel(model.z_map)
        model.decoder = torch.nn.DataParallel(model.decoder)
        model = model.cuda()
        train_datalist, test_datalist = multi_cross_validation()
        ndata = len(train_datalist)
    elif args.dataset == "gland":
        raise NotImplementedError("gland dataset haven't implemented with ae")
        # dataset_path = "/data/fhz/MICCAI2015/npy"
        # model = vae.VAE2d(latent_space_dim=args.latent_dim)
        # model.encoder = torch.nn.DataParallel(model.encoder)
        # model.z_log_sigma_map = torch.nn.DataParallel(model.z_log_sigma_map)
        # model.z_mean_map = torch.nn.DataParallel(model.z_mean_map)
        # model.decoder = torch.nn.DataParallel(model.decoder)
        # model = model.cuda()
        # train_datalist = glob(path.join(dataset_path, "train", "*.npy"))
        # test_datalist = glob(path.join(dataset_path, "test", "*.npy"))
        # ndata = len(train_datalist)
    else:
        raise FileNotFoundError("Dataset {} Not Found".format(args.dataset))
    if args.inference_flag:
        inference(model=model, train_datalist=train_datalist, test_datalist=test_datalist)
        exit("finish inference of train time {}".format(args.train_time))
    input("Begin the {} time's training".format(args.train_time))
    criterion = AECriterion().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    writer_log_dir = "/data/fhz/unsupervised_recommendation/ae_runs/ae_train_time:{}_dataset:{}".format(
        args.train_time, args.dataset)
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args = checkpoint['args']
            min_avg_reconstruct_loss = checkpoint['min_avg_reconstruct_loss']
            model.encoder.load_state_dict(checkpoint['encoder_state_dict'])
            model.z_map.load_state_dict(checkpoint['z_map_state_dict'])
            model.decoder.load_state_dict(checkpoint['decoder_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            train_datalist = checkpoint['train_datalist']
            test_datalist = checkpoint['test_datalist']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            raise FileNotFoundError("Checkpoint Resume File {} Not Found".format(args.resume))
    else:
        if os.path.exists(writer_log_dir):
            flag = input("ae_train_time:{}_dataset:{} will be removed, input yes to continue:".format(
                args.train_time, args.dataset))
            if flag == "yes":
                shutil.rmtree(writer_log_dir, ignore_errors=True)
    writer = SummaryWriter(log_dir=writer_log_dir)
    if args.pretrained:
        if os.path.isfile(args.pretrained_resume):
            print("=> loading checkpoint '{}'".format(args.pretrained_resume))
            pretrained_parameters = torch.load(args.pretrained_resume)
            # actually we use the encoding part as pretraining part
            model.encoder.load_state_dict(pretrained_parameters['encoder_state_dict'])
        else:
            raise FileNotFoundError("Pretraining Resume File Not Found")
    if args.dataset == "lung":
        train_dset = LungDataSet(data_path_list=train_datalist, augment_prob=args.aug_prob,
                                 window_width=args.window_width, window_level=args.window_level)
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
        epoch_reconstruct_loss = train(train_dloader, model=model, criterion=criterion,
                                       optimizer=optimizer, epoch=epoch, writer=writer,
                                       dataset=args.dataset)
        if (epoch + 1) % 5 == 0:
            """
            Here we define the best point as the minimum average epoch loss
            """
            is_reconstruct_loss_best = (epoch_reconstruct_loss < min_avg_reconstruct_loss)
            save_checkpoint({
                'epoch': epoch + 1,
                'args': args,
                "encoder_state_dict": model.encoder.state_dict(),
                'z_map_state_dict': model.z_map.state_dict(),
                'decoder_state_dict': model.decoder.state_dict(),
                'min_avg_reconstruct_loss': min_avg_reconstruct_loss,
                'optimizer': optimizer.state_dict(),
                'train_datalist': train_datalist,
                'test_datalist': test_datalist,
            }, is_reconstruct_loss_best)
            if (epoch + 1) > 300:
                test_datalist = sample(train_datalist, 100)
                comment = ""
                if is_reconstruct_loss_best:
                    comment += "reconstruct_"
                if comment == "":
                    comment += "common"
                else:
                    comment += "loss_best"
                test(test_datalist, model=model, train_time=args.train_time, train_epoch=epoch, dataset=args.dataset,
                     comment=comment, criterion=criterion)


def train(train_dloader, model, criterion, optimizer, epoch, writer, dataset):
    # record the time for loading a data and do backward for a batch
    # also record the loss value
    batch_time = AverageMeter()
    data_time = AverageMeter()
    reconstruct_losses = AverageMeter()
    model.train()
    end = time.time()
    optimizer.zero_grad()
    for i, (image, *_) in enumerate(train_dloader):
        data_time.update(time.time() - end)
        image = image.float().cuda()
        image_reconstructed = model(image)
        reconstruct_loss = criterion(image, image_reconstructed)
        # loss = kl_loss
        reconstruct_loss.backward()
        if args.gd_clip_flag:
            torch.nn.utils.clip_grad_value_(model.parameters(), args.gd_clip_value)
        reconstruct_losses.update(float(reconstruct_loss), image.size(0))
        optimizer.step()
        optimizer.zero_grad()
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            train_text = 'Epoch: [{0}][{1}/{2}]\t' \
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                         'Reconstruct Loss {reconstruct_loss.val:.4f} ({reconstruct_loss.avg:.4f})\t'.format(
                epoch, i, len(train_dloader), batch_time=batch_time,
                data_time=data_time, reconstruct_loss=reconstruct_losses)
            print(train_text)
    writer.add_scalar(tag="{}_train/reconstruct_loss".format(dataset), scalar_value=reconstruct_losses.avg,
                      global_step=epoch)
    return reconstruct_losses.avg


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


def save_checkpoint(state, is_reconstruct_loss_best, filename='checkpoint.pth.tar'):
    """
    :param state: a dict including:{
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'lemniscate': lemniscate,
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }
    :param is_reconstruct_loss_best: if the reconstructed loss is best
    :param filename: the filename for store
    :return:
    """
    filefolder = '/data/fhz/unsupervised_recommendation/ae_parameter/unsupervised_recommendation_train_ae_time_{}_train_dset_{}'.format(
        state["args"].train_time, state["args"].dataset)
    if not path.exists(filefolder):
        os.makedirs(filefolder)
    torch.save(state, path.join(filefolder, filename))
    if is_reconstruct_loss_best:
        shutil.copyfile(path.join(filefolder, filename),
                        path.join(filefolder, 'model_reconstruct_loss_best.pth.tar'))


def test(test_datalist, model, train_time, train_epoch, dataset, comment, criterion):
    filefolder = '/data/fhz/unsupervised_recommendation/unsupervised_recommendation_train_ae_time_{}_train_dset_{}/test_{}_epoch/{}'.format(
        train_time, dataset, train_epoch, comment)
    if not path.exists(filefolder):
        os.makedirs(filefolder)
    model.eval()
    if dataset == "lung":
        test_dset = LungDataSet(data_path_list=test_datalist, augment_prob=0, need_name_label=True,
                                window_width=args.window_width, window_level=args.window_level)
        test_dloader = DataLoader(dataset=test_dset, batch_size=1, shuffle=False,
                                  num_workers=args.workers, pin_memory=True)
    elif dataset == "gland":
        test_dset = GlandDataset(data_path_list=test_datalist, need_name_label=True, need_seg_label=False,
                                 augment_prob=0)
        test_dloader = DataLoader(dataset=test_dset, batch_size=1, shuffle=False,
                                  num_workers=args.workers, pin_memory=True)
    else:
        raise NameError("dataset name illegal")
    for i, (image, idx, image_name, *_) in enumerate(test_dloader):
        image_name, *_ = image_name
        save_path = path.join(filefolder, image_name)
        if not path.exists(save_path):
            os.makedirs(save_path)
        image = image.float().cuda()
        with torch.no_grad():
            image_reconstructed = model(image)
        reconstruct_loss = criterion(image, image_reconstructed)
        save_file = path.join(save_path, "rcl_{:.4f}.npy".format(float(reconstruct_loss), ))
        np.save(save_file, image_reconstructed.cpu().detach().numpy())
        np.save(path.join(save_path, "raw.npy"), image.cpu().detach().numpy())


def inference(model, train_datalist, test_datalist, folder_path=None):
    if folder_path is None:
        folder_path = '/data/fhz/unsupervised_recommendation/ae_inference'
        folder_path = os.path.join(folder_path, "train_time_{}".format(args.train_time))
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    resume_path = "/data/fhz/unsupervised_recommendation/ae_parameter/unsupervised_recommendation_train_ae_time_{}_train_dset_lung/model_reconstruct_loss_best.pth.tar".format(
        args.train_time)
    if os.path.isfile(resume_path):
        print("=> loading checkpoint '{}'".format(resume_path))
        checkpoint = torch.load(resume_path)
        model.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        model.z_map.load_state_dict(checkpoint['z_map_state_dict'])
        model.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        model.eval()

        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resume_path, checkpoint['epoch']))
    else:
        raise FileNotFoundError("Checkpoint Resume File {} Not Found".format(args.resume))
    if args.dataset == "lung":
        train_dset = LungDataSet(data_path_list=train_datalist, augment_prob=0, need_name_label=True,
                                 window_level=args.window_level, window_width=args.window_width)
        test_dset = LungDataSet(data_path_list=test_datalist, augment_prob=0, need_name_label=True,
                                window_level=args.window_level, window_width=args.window_width)
        train_dloader = DataLoader(dataset=train_dset, batch_size=1, shuffle=False,
                                   num_workers=args.workers, pin_memory=True)
        test_dloader = DataLoader(dataset=test_dset, batch_size=1, shuffle=False,
                                  num_workers=args.workers, pin_memory=True)
    elif args.dataset == "gland":
        train_dset = GlandDataset(data_path_list=test_datalist, need_name_label=True, need_seg_label=False,
                                  augment_prob=0)
        test_dset = GlandDataset(data_path_list=test_datalist, need_name_label=True, need_seg_label=False,
                                 augment_prob=0)
        train_dloader = DataLoader(dataset=train_dset, batch_size=1, shuffle=False,
                                   num_workers=args.workers, pin_memory=True)
        test_dloader = DataLoader(dataset=test_dset, batch_size=1, shuffle=False,
                                  num_workers=args.workers, pin_memory=True)
    else:
        raise NameError("Dataset {} not exist".format(args.dataset))

    # Here we have train dataloader , test dataloader and then we can do inference
    # save in train folder
    train_inference = {}
    test_inference = {}
    for i, (image, index, img_name, *_) in enumerate(train_dloader):
        image = image.float().cuda()
        img_name, *_ = img_name
        features = model.encoder(image)
        features = features.view(image.size(0), -1)
        z = model.z_map(features)
        z = z.cpu().detach().numpy()
        train_inference[img_name] = {"z": z}
        print("{} inferenced".format(img_name))
    with open(path.join(folder_path, "train.pkl"), "wb") as train_pkl:
        pickle.dump(obj=train_inference, file=train_pkl)
        print("train dataset inferenced")
    for i, (image, index, img_name, *_) in enumerate(test_dloader):
        image = image.float().cuda()
        img_name, *_ = img_name
        features = model.encoder(image)
        features = features.view(image.size(0), -1)
        z = model.z_map(features)
        z = z.cpu().detach().numpy()
        test_inference[img_name] = {"z": z}
        print("{} inferenced".format(img_name))
    with open(path.join(folder_path, "test.pkl"), "wb") as test_pkl:
        pickle.dump(obj=test_inference, file=test_pkl)
        print("test dataset inferenced")


if __name__ == "__main__":
    main()
