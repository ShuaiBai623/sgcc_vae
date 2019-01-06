import torch
import numpy as np
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torch.optim as optim
from torch import nn
import matplotlib.pyplot as plt
from os import path
import os
import random

"""
Here is just a simple example for MNIST dataset(using vae)
"""


class Encoder(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Encoder, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        return F.relu(self.linear2(x))


class Decoder(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Decoder, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        return F.relu(self.linear2(x))


class VAE(torch.nn.Module):
    def __init__(self, encoder, decoder, latent_dim):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.enc_mu = torch.nn.Linear(100, latent_dim)
        self.enc_log_sigma = torch.nn.Linear(100, latent_dim)

    def _sample_latent(self, h_enc):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        mu = self.enc_mu(h_enc)
        log_sigma = self.enc_log_sigma(h_enc)
        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float().cuda()
        self.z_mean = mu
        self.z_sigma = sigma

        return mu + sigma * std_z  # Reparameterization trick

    def forward(self, state):
        h_enc = self.encoder(state)
        z = self._sample_latent(h_enc)
        return self.decoder(z)


def latent_loss(z_mean, z_stddev):
    mean_sq = z_mean * z_mean
    stddev_sq = z_stddev * z_stddev
    return 0.5 * torch.sum(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    """
    :param state: a dict including:{
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'lemniscate': lemniscate,
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }
    :param is_total_loss_best: if the total epoch loss is best
    :param is_reconstruct_loss_best: if the reconstructed loss is best
    :param is_kl_loss_best: if the Kullback-Leibler divergency loss is best
    :param filename: the filename for store
    :return:
    """
    filefolder = './train_mnist_vae_time{}'.format(
        state["train_time"])
    if not path.exists(filefolder):
        os.mkdir(filefolder)
    torch.save(state, path.join(filefolder, filename))


if __name__ == '__main__':
    import argparse
    import __init__
    from lib.utils.avgmeter import AverageMeter

    os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
    parser = argparse.ArgumentParser(description='pytorch training vae in MNIST')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('-t', '--train-time', default=1, type=int,
                        metavar='N', help='the x-th time of training')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('-ld', '--latent-dim', default=8, type=int,
                        metavar='D', help='feature dimension in latent space')
    parser.add_argument("-s", "--x-sigma", default=1, type=float,
                        help="The standard variance for reconstructed images, work as regularization")
    args = parser.parse_args()
    input_dim = 28 * 28
    batch_size = 32
    transform = transforms.Compose(
        [transforms.ToTensor()])
    mnist_train = torchvision.datasets.MNIST("/data/fhz/mnist/", transform=transform, train=True)
    mnist_test = torchvision.datasets.MNIST("/data/fhz/mnist/", transform=transform, train=False)
    train_dloader = torch.utils.data.DataLoader(mnist_train, batch_size=args.batch_size, num_workers=args.workers)
    encoder = Encoder(input_dim, 100, 100)
    decoder = Decoder(args.latent_dim, 100, input_dim)
    vae = VAE(encoder, decoder, latent_dim=args.latent_dim).cuda()
    vae.train()
    vae.encoder = torch.nn.DataParallel(vae.encoder)
    vae.decoder = torch.nn.DataParallel(vae.decoder)
    vae.enc_mu = torch.nn.DataParallel(vae.enc_mu)
    vae.enc_log_sigma = torch.nn.DataParallel(vae.enc_log_sigma)
    criterion = nn.MSELoss(reduction="sum")
    optimizer = optim.Adam(vae.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for epoch in range(args.start_epoch, args.epochs):
        reconstruct_losses = AverageMeter()
        kl_losses = AverageMeter()
        losses = AverageMeter()
        for i, data in enumerate(train_dloader):
            image, label = data
            image = image.view(args.batch_size, -1).cuda()
            optimizer.zero_grad()
            img_reconstruct = vae(image)
            kl_loss = latent_loss(vae.z_mean, vae.z_sigma)
            reconstruct_loss = criterion(img_reconstruct, image)
            loss = reconstruct_loss / args.x_sigma + kl_loss
            loss.backward()
            optimizer.step()
            losses.update(float(loss), image.size(0))
            reconstruct_losses.update(float(reconstruct_loss), image.size(0))
            kl_losses.update(float(kl_loss), image.size(0))
        train_text = 'Epoch: [{0}/{1}]\t' \
                     'Total Loss {total_loss.avg:.4f}\t' \
                     'Reconstruct Loss {reconstruct_loss.avg:.4f}\t' \
                     'KL Loss {kl_loss.avg:.4f}\t'.format(epoch, args.epochs, total_loss=losses,
                                                          reconstruct_loss=reconstruct_losses, kl_loss=kl_losses)
        print(train_text)
        if (epoch + 1) % 5 == 0:
            save_checkpoint(state={
                'epoch': epoch + 1,
                "train_time": args.train_time,
                'model_state_dict': vae.state_dict(),
                'optimizer': optimizer.state_dict()
            })
    # # we test it in jupyter notebook
    # encoder = Encoder(input_dim, 100, 100)
    #     # decoder = Decoder(args.latent_dim, 100, input_dim)
    #     # vae = VAE(encoder, decoder, latent_dim=args.latent_dim)
    #     # vae = vae.cuda()
    #     # checkpoint = torch.load("./train_mnist_vae_time1/checkpoint.pth.tar")
    #     # vae.load_state_dict(checkpoint['model_state_dict'])
    #     # vae.eval()
    #     # train_img, *_ = mnist_train.__getitem__(random.randint(0, len(mnist_train)))
    #     # train_img = train_img.unsqueeze(0).cuda()
    # train_img_reconstruct = vae(train_img)
    # plt.imshow(train_img.cpu().detach().numpy().reshape(28, 28),cmap="gray")
    # plt.show()
    #
    # test_img, *_ = mnist_test.__getitem__(random.randint(0, len(mnist_test)))
    # test_img = test_img.unsqueeze(0).cuda()
    # test_img_reconstruct = vae(test_img)
    # plt.imshow(test_img.cpu().detach().numpy().reshape(28, 28),cmap="gray")
    # plt.show()
