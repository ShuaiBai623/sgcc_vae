"""
different from idfe
here we should add the test module, it is used to test if the vae
can generate similar images and we can show it.
which means visualize the vae(2d or 3d)
"""
import __init__
import argparse
from model import denseunet_hierarchical_vae
from lib.Criterion import VAECriterion
from torch import nn
from lib.utils.avgmeter import AverageMeter
from lib.dataloader import SGCCDataset
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


parser = argparse.ArgumentParser(description='pytorch training hiearachical vae')
parser.add_argument('--dataset', default="sgcc_dataset", type=str, metavar='DataPath',
                    help='The folder path of dataset can be sgcc_dataset or 25000Img')
parser.add_argument('-bp', '--base_path', default="/data/fhz")
parser.add_argument('-cf', '--crossval-fold', default=1, type=int, metavar='From 1 To 5',
                    help='the folder of cross validation')
parser.add_argument('-j', '--workers', default=64, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=1200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=15, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('-t', '--train-time', default=1, type=int,
                    metavar='N', help='the x-th time of training')
parser.add_argument('--lr', '--learning-rate', default=1e-8, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('-akb', '--adjust-kl-beta-epoch', default=200, type=float, metavar='KL Beta',
                    help='the epoch to linear adjust kl beta')
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
parser.add_argument('-ld', "--latent-dim", default=[64, 64], type=arg_as_list,
                    metavar='D List', help='feature dimension in latent space for each hierarchical')
parser.add_argument('-is', "--image-size", default=[368, 464], type=arg_as_list,
                    metavar='Image Size List', help='the size of h * w for image')
parser.add_argument('-bc', "--block-config", default=[6, 12, 24, 16], type=arg_as_list,
                    metavar='Image Size List', help='the size of h * w for image')
# This may be used later
# parser.add_argument('--nce-m', default=0.5, type=float,
#                     help='momentum for non-parametric updates')
parser.add_argument('-ad', "--adjust-lr", default=[800, 1000], type=arg_as_list,
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

min_avg_train_total_loss = 1e8
min_avg_test_total_loss = 1e8


def main():
    global args, best_prec1, min_avg_train_total_loss, min_avg_test_total_loss
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # ugly if sentence to avoid some ugly situation:
    # for we don't use args.image_size and args.block_config parameters in the first 10 train progress
    model = denseunet_hierarchical_vae.DenseUnetHiearachicalVAE(latent_dim=args.latent_dim,
                                                                data_parallel=args.data_parallel,
                                                                img_size=args.image_size,
                                                                block_config=args.block_config)
    model = model.cuda()
    if args.inference_flag:
        dataset_path = path.join(args.base_path, args.dataset)
        data_path_list = glob(path.join(dataset_path, "*.png"))
        train_data_path_list, test_data_path_list = multi_cross_validation(data_path_list, args.crossval_fold)
        inference(model=model, train_data_path_list=train_data_path_list, test_data_path_list=test_data_path_list)
        exit("finish inference of train time {}".format(args.train_time))
    input("Begin the {} time's training".format(args.train_time))
    criterion = VAECriterion(x_sigma=args.x_sigma).cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    writer_log_dir = "{}/sgcc_vae/vae_runs/vae_train_time:{}_dataset:{}/fold_{}".format(args.base_path,
                                                                                        args.train_time, args.dataset,
                                                                                        args.crossval_fold)
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            if not args.not_resume_arg:
                args = checkpoint['args']
                args.start_epoch = checkpoint['epoch']
            min_avg_test_total_loss = checkpoint['min_avg_test_total_loss']
            min_avg_train_total_loss = checkpoint['min_avg_train_total_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            raise FileNotFoundError("Checkpoint Resume File {} Not Found".format(args.resume))
    else:
        if os.path.exists(writer_log_dir):
            flag = input("vae_train_time:{}_dataset:{}fold:{} will be removed, input yes to continue:".format(
                args.train_time, args.dataset, args.crossval_fold))
            if flag == "yes":
                shutil.rmtree(writer_log_dir, ignore_errors=True)
    writer = SummaryWriter(log_dir=writer_log_dir)
    dataset_path = path.join(args.base_path, args.dataset)
    data_path_list = glob(path.join(dataset_path, "*.png"))
    train_data_path_list, test_data_path_list = multi_cross_validation(data_path_list, args.crossval_fold)
    train_dset = SGCCDataset(data_path_list=train_data_path_list)
    train_dloader = DataLoader(dataset=train_dset, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.workers, pin_memory=True)
    test_dset = SGCCDataset(data_path_list=test_data_path_list)
    test_dloader = DataLoader(dataset=test_dset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True)
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        kl_beta = max(1 / args.adjust_kl_beta_epoch, min(1, epoch / args.adjust_kl_beta_epoch))
        train_total_loss, train_reconstruct_loss, train_kl_loss = train(train_dloader, model=model, criterion=criterion,
                                                                        optimizer=optimizer, epoch=epoch, writer=writer,
                                                                        dataset=args.dataset, kl_beta=kl_beta,
                                                                        fold=args.crossval_fold)
        test_total_loss, test_reconstruct_loss, test_kl_loss = test_loss(test_dloader, model=model,
                                                                         criterion=criterion, epoch=epoch,
                                                                         writer=writer,
                                                                         dataset=args.dataset, fold=args.crossval_fold)

        if (epoch + 1) % 20 == 0:
            """
            Here we define the best point as the minimum average epoch loss
            """
            is_train_total_loss_best = (train_total_loss < min_avg_train_total_loss)
            is_test_total_loss_best = (test_total_loss < min_avg_test_total_loss)
            min_avg_train_total_loss = min(min_avg_train_total_loss, train_total_loss)
            min_avg_test_total_loss = min(min_avg_test_total_loss, test_total_loss)
            save_checkpoint({
                'epoch': epoch + 1,
                'args': args,
                "state_dict": model.state_dict(),
                "min_avg_test_total_loss": min_avg_test_total_loss,
                'min_avg_train_total_loss': min_avg_train_total_loss,
                'optimizer': optimizer.state_dict(),
            }, is_train_total_loss_best, is_test_total_loss_best)
            if (epoch + 1) >= args.epochs - 200:
                # we do 2 test ,one for the trained dataset
                comment = ""
                if is_train_total_loss_best:
                    comment += "train_"
                if is_test_total_loss_best:
                    comment += "test_"
                if comment == "":
                    comment += "common"
                else:
                    comment += "loss_best"
                test_datalist = sample(train_data_path_list, 10)
                test_reconstruct(test_datalist, model=model, train_time=args.train_time, train_epoch=epoch,
                                 dataset=args.dataset,
                                 comment=comment, criterion=criterion, test_train_dset_flag=True)
                test_datalist = sample(test_data_path_list, 10)
                test_reconstruct(test_datalist, model=model, train_time=args.train_time, train_epoch=epoch,
                                 dataset=args.dataset,
                                 comment=comment, criterion=criterion, test_train_dset_flag=False)


def train(train_dloader, model, criterion, optimizer, epoch, writer, dataset, kl_beta, fold):
    # record the time for loading a data and do backward for a batch
    # also record the loss value
    batch_time = AverageMeter()
    data_time = AverageMeter()
    reconstruct_losses = AverageMeter()
    kl_losses = AverageMeter()
    losses = AverageMeter()
    model.train()
    end = time.time()
    optimizer.zero_grad()
    for i, (image, *_) in enumerate(train_dloader):
        data_time.update(time.time() - end)
        image = image.float().cuda()
        image_reconstructed, latent_mean_list, latent_sigma_list, latent_log_sigma_list = model(
            image)
        latent_mean = torch.cat(latent_mean_list, 1)
        latent_sigma = torch.cat(latent_sigma_list, 1)
        latent_log_sigma = torch.cat(latent_log_sigma_list, 1)
        reconstruct_loss, kl_loss = criterion(image, image_reconstructed, latent_mean, latent_log_sigma,
                                              latent_sigma)
        loss = reconstruct_loss + kl_beta * kl_loss
        # loss = kl_loss
        loss.backward()
        if args.gd_clip_flag:
            torch.nn.utils.clip_grad_value_(model.parameters(), args.gd_clip_value)
        losses.update(float(loss), image.size(0))
        reconstruct_losses.update(float(reconstruct_loss), image.size(0))
        kl_losses.update(float(kl_loss), image.size(0))
        optimizer.step()
        optimizer.zero_grad()
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            train_text = 'Epoch: [{0}][{1}/{2}]\t' \
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t' \
                         'Total Loss {total_loss.val:.4f} ({total_loss.avg:.4f})\t' \
                         'Reconstruct Loss {reconstruct_loss.val:.4f} ({reconstruct_loss.avg:.4f})\t' \
                         'KL Loss {kl_loss.val:.4f} ({kl_loss.avg:.4f})\t'.format(
                epoch, i + 1, len(train_dloader), batch_time=batch_time,
                data_time=data_time, total_loss=losses, reconstruct_loss=reconstruct_losses, kl_loss=kl_losses)
            print(train_text)
    writer.add_scalar(tag="{}_train_fold_{}/loss".format(dataset, fold), scalar_value=losses.avg, global_step=epoch)
    writer.add_scalar(tag="{}_train_fold_{}/reconstruct_loss".format(dataset, fold),
                      scalar_value=reconstruct_losses.avg,
                      global_step=epoch)
    writer.add_scalar(tag="{}_train_fold_{}/kl_loss".format(dataset, fold), scalar_value=kl_losses.avg,
                      global_step=epoch)
    return losses.avg, reconstruct_losses.avg, kl_losses.avg


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 100 epochs"""
    # lr_beta = min(1, 100*epoch / args.adjust_kl_beta_epoch)
    if epoch < args.adjust_lr[0]:
        # lr = args.lr / max(kl_beta, 1 / args.adjust_kl_beta_epoch)
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


def save_checkpoint(state, is_train_total_loss_best, is_test_total_loss_best,
                    filename='checkpoint.pth.tar'):
    """
    :param state: a dict including:{
                'epoch': epoch + 1,
                'args': args,
                "state_dict": model.state_dict(),
                "min_avg_test_total_loss": min_avg_test_total_loss,
                'min_avg_train_total_loss': min_avg_train_total_loss,
                'optimizer': optimizer.state_dict(),
        }
    :param is_train_total_loss_best: if the total epoch train loss is best
    :param is_test_total_loss_best: if the total epoch test loss is best
    :param filename: the filename for store
    :return:
    """
    filefolder = '{}/sgcc_vae/vae_parameter/train_vae_time_{}_train_dset_{}/fold_{}'.format(args.base_path,
                                                                                            state["args"].train_time,
                                                                                            state["args"].dataset,
                                                                                            state["args"].crossval_fold)
    if not path.exists(filefolder):
        os.makedirs(filefolder)
    torch.save(state, path.join(filefolder, filename))
    if is_train_total_loss_best:
        shutil.copyfile(path.join(filefolder, filename),
                        path.join(filefolder, 'model_train_total_loss_best.pth.tar'))
    if is_test_total_loss_best:
        shutil.copyfile(path.join(filefolder, filename),
                        path.join(filefolder, 'model_test_total_loss_best.pth.tar'))


def test_reconstruct(test_datalist, model, train_time, train_epoch, dataset, comment, criterion,
                     test_train_dset_flag=False):
    if test_train_dset_flag:
        filefolder = '{}/sgcc_vae/train_vae_time_{}_train_dset_{}/fold_{}/test_{}_epoch/train_dset/{}'.format(
            args.base_path,
            train_time, dataset,
            args.crossval_fold,
            train_epoch, comment)
    else:
        filefolder = '{}/sgcc_vae/train_vae_time_{}_train_dset_{}/fold_{}/test_{}_epoch/test_dset/{}'.format(
            args.base_path,
            train_time, dataset,
            args.crossval_fold,
            train_epoch, comment)
    if not path.exists(filefolder):
        os.makedirs(filefolder)
    model.eval()
    test_dset = SGCCDataset(data_path_list=test_datalist)
    test_dloader = DataLoader(dataset=test_dset, batch_size=1, shuffle=False,
                              num_workers=args.workers, pin_memory=True)
    for i, (image, idx, image_name, *_) in enumerate(test_dloader):
        image_name, *_ = image_name
        save_path = path.join(filefolder, image_name)
        if not path.exists(save_path):
            os.makedirs(save_path)
        image = image.float().cuda()
        with torch.no_grad():
            image_multi_reconstructed, image_multi_mean_list, image_multi_sigma_list, image_multi_log_sigma_list = model.generate_from_raw_img(
                input_img=image, generate_times=10)
        """
        change criterion functions
        """
        for j in range(10):
            image_mean_list = [mu[j:j + 1, :] for mu in image_multi_mean_list]
            image_sigma_list = [sigma[j:j + 1, :] for sigma in image_multi_sigma_list]
            image_log_sigma_list = [log_sigma[j:j + 1, :] for log_sigma in image_multi_log_sigma_list]
            z_mean = torch.cat(image_mean_list, 1)
            z_sigma = torch.cat(image_sigma_list, 1)
            z_log_sigma = torch.cat(image_log_sigma_list, 1)
            reconstruct_loss, kl_loss = criterion(image, image_multi_reconstructed[j:j + 1, :], z_mean, z_log_sigma,
                                                  z_sigma)
            save_file = path.join(save_path,
                                  "reconstruct_{}_rcl_{:.4f}_kl{:.4f}.npy".format(j, float(reconstruct_loss),
                                                                                  float(kl_loss)))
            np.save(save_file, image_multi_reconstructed[j:j + 1, :].cpu().detach().numpy())
        np.save(path.join(save_path, "raw.npy"), image.cpu().detach().numpy())


def test_loss(test_dloader, model, criterion, epoch, writer, dataset, fold):
    # record the time for loading a data and do backward for a batch
    # also record the loss value
    reconstruct_losses = AverageMeter()
    kl_losses = AverageMeter()
    losses = AverageMeter()
    model.eval()
    for i, (image, *_) in enumerate(test_dloader):
        image = image.float().cuda()
        with torch.no_grad():
            image_reconstructed, latent_mean_list, latent_sigma_list, latent_log_sigma_list = model(
                image)
        latent_mean = torch.cat(latent_mean_list, 1)
        latent_sigma = torch.cat(latent_sigma_list, 1)
        latent_log_sigma = torch.cat(latent_log_sigma_list, 1)
        reconstruct_loss, kl_loss = criterion(image, image_reconstructed, latent_mean, latent_log_sigma,
                                              latent_sigma)
        loss = reconstruct_loss + kl_loss
        losses.update(float(loss), image.size(0))
        reconstruct_losses.update(float(reconstruct_loss), image.size(0))
        kl_losses.update(float(kl_loss), image.size(0))
    writer.add_scalar(tag="{}_test_fold_{}/loss".format(dataset, fold), scalar_value=losses.avg, global_step=epoch)
    writer.add_scalar(tag="{}_test_fold_{}/reconstruct_loss".format(dataset, fold), scalar_value=reconstruct_losses.avg,
                      global_step=epoch)
    writer.add_scalar(tag="{}_test_fold_{}/kl_loss".format(dataset, fold), scalar_value=kl_losses.avg,
                      global_step=epoch)
    return losses.avg, reconstruct_losses.avg, kl_losses.avg


def inference(model, train_data_path_list, test_data_path_list, folder_path=None, resume_test_best=True):
    if folder_path is None:
        folder_path = '{}/sgcc_vae/vae_inference'.format(args.base_path)
        folder_path = os.path.join(folder_path,
                                   "train_time_{}_dset_{}/fold_{}".format(args.train_time, args.dataset,
                                                                          args.crossval_fold))
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    if resume_test_best:
        resume_path = "{}/sgcc_vae/vae_parameter/train_vae_time_{}_train_dset_{}/fold_{}/model_test_total_loss_best.pth.tar".format(
            args.base_path, args.train_time, args.dataset, args.crossval_fold)
    else:
        resume_path = "{}/sgcc_vae/vae_parameter/train_vae_time_{}_train_dset_{}/fold_{}/model_train_total_loss_best.pth.tar".format(
            args.base_path, args.train_time, args.dataset, args.crossval_fold)
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

    # Here we have train dataloader , test dataloader and then we can do inference
    # save in train folder
    inference_dict = {}
    for i, (image, index, img_name, *_) in enumerate(train_data_loader):
        image = image.float().cuda()
        img_name, *_ = img_name
        _, all_latent_variable_mean, all_latent_variable_sigma, all_latent_variable_log_sigma = model(image)
        mu_list = [mu.cpu().detach().numpy() for mu in all_latent_variable_mean]
        log_sigma_list = [log_sigma.cpu().detach().numpy() for log_sigma in all_latent_variable_log_sigma]
        inference_dict[img_name] = {"mu": mu_list, "log_sigma": log_sigma_list}
        print("{} inferenced".format(img_name))
    with open(path.join(folder_path, "train.pkl"), "wb") as train_pkl:
        pickle.dump(obj=inference, file=train_pkl)
        print("train dataset inferenced")
    test_dataset = SGCCDataset(data_path_list=test_data_path_list)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False,
                                  num_workers=args.workers, pin_memory=True)

    # Here we have train dataloader , test dataloader and then we can do inference
    # save in train folder
    inference_dict = {}
    for i, (image, index, img_name, *_) in enumerate(test_data_loader):
        image = image.float().cuda()
        img_name, *_ = img_name
        _, all_latent_variable_mean, all_latent_variable_sigma, all_latent_variable_log_sigma = model(image)
        mu_list = [mu.cpu().detach().numpy() for mu in all_latent_variable_mean]
        log_sigma_list = [log_sigma.cpu().detach().numpy() for log_sigma in all_latent_variable_log_sigma]
        inference_dict[img_name] = {"mu": mu_list, "log_sigma": log_sigma_list}
        print("{} inferenced".format(img_name))
    with open(path.join(folder_path, "test.pkl"), "wb") as train_pkl:
        pickle.dump(obj=inference, file=train_pkl)
        print("test dataset inferenced")


if __name__ == "__main__":
    main()
