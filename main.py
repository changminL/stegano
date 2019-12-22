import argparse
import os
import shutil
import socket
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
import torchvision.transforms as transforms

from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from data import MyImageFolder
from model import UnetGenerator
from model import RevealNet
from text_data import *
from utils import *
import pdb

DATA_DIR = '/media/changmin/mini_hard/ImageNet/'
TEXT_DATA_DIR = "/home/changmin/research/steganography/data/"

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="train",
                    help='train | val | test')
parser.add_argument('--workers', type=int, default=8,
                    help='number of data loading workers')
parser.add_argument('--batchsize', type=int, default=32,
                    help='input batch size')
parser.add_argument('--imagesize', type=int, default=256,
                    help='the number of frames')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate, default=0.001')
parser.add_argument('--decay_round', type=int, default=10,
                    help='learning rate decay 0.5 each decay_round')
parser.add_argument('--beta1', type=float, default=0.5,
                    help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', type=bool, default=True,
                    help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1,
                    help='number of GPUs to use')
parser.add_argument('--Hnet', default='',
                    help="path to Hidingnet (to continue training)")
parser.add_argument('--Rnet', default='',
                    help="path to Revealnet (to continue training)")
parser.add_argument('--trainpics', default='./training/',
                    help='folder to output training images')
parser.add_argument('--validationpics', default='./training/',
                    help='folder to output validation images')
parser.add_argument('--testpics', default='./training/',
                    help='folder to output test images')
parser.add_argument('--outckpts', default='./training/',
                    help='folder to output checkpoints')
parser.add_argument('--outlogs', default='./training/',
                    help='folder to output images')
parser.add_argument('--outcodes', default='./training/',
                    help='folder to save the experiment codes')
parser.add_argument('--beta', type=float, default=0.75,
                    help='hyper parameter of beta')
parser.add_argument('--remark', default='', help='comment')
parser.add_argument('--test', default='', help='test mode, you need give the test pics dirs in this param')

parser.add_argument('--hostname', default=socket.gethostname(), help='the host name of the running server')
parser.add_argument('--debug', type=bool, default=False, help='debug mode do not create folders')
parser.add_argument('--logfrequency', type=int, default=10, help='the frequency of print the log on the console')
parser.add_argument('--resultpicfrequency', type=int, default=100, help='the frequency of save the resultpic')

def main():
    global writer, smallestLoss, optimizerH, optimizerR, schedulerH, schedulerR

    opt = parser.parse_args()

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, "
              "so you should porbably run with --cuda")

    cudnn.benchmark = True

    create_dir_to_save_result(opt)

    logpath = opt.outlogs + '%s_%d_log.txt' % (opt.dataset, opt.batchsize)

    print_log(opt, str(opt), logpath)

    save_current_codes(opt.outcodes)

    if opt.test == '':
        voc, _ = loadPrepareData(None, "all", os.path.join(TEXT_DATA_DIR, "dialogues_text.txt"), 768)
        # tensorboardX writer
        writer = SummaryWriter(comment='**' + opt.remark)
        # Get the dataset
        traindir = os.path.join(DATA_DIR, 'train')
        texttraindir = os.path.join(TEXT_DATA_DIR, "train/dialogues_train.txt")
        valdir = os.path.join(DATA_DIR, 'val')
        textvaldir = os.path.join(TEXT_DATA_DIR, "validation/dialogues_validation.txt")
        train_dataset = MyImageFolder(
            traindir, # Preprocessing the data
            transforms.Compose([
                transforms.Resize([opt.imagesize, opt.imagesize]), # Randomly cut and resize the data to a given size
                transforms.ToTensor(),
                # Convert a numpy.ndarray with a value range of [0,255] or a shape of (H,W,C) to
                # a torch.FloatTensor with a shape of [C,H,W] and a value of [0, 1.0] torch.FloatTensor
            ]),
            True)
        _, text_train_dataset = loadPrepareData(None, "train", texttraindir, 768)
        val_dataset = MyImageFolder(
            valdir, # Preprocessing the data
            transforms.Compose([ # Combine several transforms together
                transforms.Resize([opt.imagesize, opt.imagesize]), # Randomly cut and resize the data to a given size
                transforms.ToTensor(),
                # Convert a numpy.ndarray with a value range of [0, 255] or a shpae of (H,W,C) to
                # a torch.FloatTensor with a shape of [C,H,W] and a value of [0, 1.0] torch.FloatTensor
            ]))
        _, text_val_dataset = loadPrepareData(None, "val", textvaldir, 768)
        assert train_dataset
        assert val_dataset
        assert text_train_dataset
        assert text_val_dataset
    else:
        testdir = opt.test
        texttestdir =  os.path.join(TEXT_DATA_DIR, "test/dialogues_test.txt")
        test_dataset = MyImageFolder(
            testdir, # Preprocessing the data
            transforms.Compose([ # Combine several transfroms together
                transforms.Resize([opt.imagesize, opt.imagesize]),
                transforms.ToTensor(),
            ]))
        _, text_test_dataset = loadPrepareData(None, "test", texttestdir, 768)
        assert test_dataset
        assert text_test_dataset
    # Create word embedding layer
    embedding = nn.Embedding(voc.num_words, 256)
    embedding.cuda()
    embedding.weight.data.uniform_(-1, 1)
    if opt.embedding != '':
        embedding.load_state_dict(torch.load(opt.embedding))
    if opt.ngpu > 1:
        embedding = torch.nn.DataParallel(embedding).cuda()

    # Create Hiding network objects
    Hnet = UnetGenerator(input_nc=6, output_nc=3, num_downs=7, output_function=nn.Sigmoid)
    Hnet.cuda()
    Hnet.apply(weights_init)
    # Determine whether to continue the previous training
    if opt.Hnet != "":
        Hnet.load_state_dict(torch.load(opt.Hnet))
    if opt.ngpu > 1:
        Hnet = torch.nn.DataParallel(Hnet).cuda()
    print_network(opt, Hnet, logpath)

    # Create Reveal network objects
    Rnet = RevealNet(output_function=nn.Sigmoid)
    Rnet.cuda()
    Rnet.apply(weights_init)
    if opt.Rnet != '':
        Rnet.load_state_dict(torch.load(opt.Rnet))
    if opt.ngpu > 1:
        Rnet = torch.nn.DataParallel(Rnet).cuda()
    print_network(opt, Rnet, logpath)

    # CosineSimilarity
    cosinesimilarity = nn.CosineSimilarity(dim=4).cuda()
    # Mean Square Error loss
    criterion = nn.MSELoss().cuda()
    # training mode
    if opt.test == '':
        # setup optimizer
        optimizerH = optim.Adam(Hnet.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        schedulerH = ReduceLROnPlateau(optimizerH, mode='min', factor=0.2, patience=5, verbose=True)

        optimizerR = optim.Adam(Rnet.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        schedulerR = ReduceLROnPlateau(optimizerR, mode='min', factor=0.2, patience=8, verbose=True)

        train_loader = DataLoader(train_dataset, batch_size=opt.batchsize,
                                  shuffle=True, num_workers=int(opt.workers))
        val_loader = DataLoader(val_dataset, batch_size=opt.batchsize,
                                shuffle=True, num_workers=int(opt.workers))
        smallestLoss = 10000
        print_log(opt, "-------------------Starts Training----------------------", logpath)

        for epoch in range(opt.epochs):
            # train
            train(opt, train_loader, epoch, voc, text_train_dataset, Hnet=Hnet, Rnet=Rnet,
                  criterion=criterion, cosinesimilarity=cosinesimilarity, logpath=logpath)

            # validation
            val_hloss, val_rloss, val_sumloss = validation(opt, val_loader, epoch, voc, text_val_dataset,
                                                           Hnet=Hnet, Rnet=Rnet, criterion=criterion, logpath=logpath)

            # adjust learning rate
            schedulerH.step(val_sumloss)
            schedulerR.step(val_rloss)

            # save the best model parameters
            if val_sumloss < globals()["smallestLoss"]:
                globals()["smallestLoss"] = val_sumloss
                # do check pointing
                torch.save(Hnet.state_dict(),
                           '%s/netH_epoch_%d,sumloss=%.f,Hloss=%.6f.pth' % (
                           opt.outckpts, epoch, val_sumloss, val_hloss))
                torch.save(Rnet.state_dict(),
                           '%s/netR_epoch_%d,sumloss=%.6f,Rloss=%.6f.pth' % (
                           opt.outckpts, epoch, val_sumloss, val_rloss))

        writer.close()

    # test mode
    else:
        test_loader = DataLoader(test_dataset, batch_size=opt.batchsize,
                                 shuffle=False, num_workers=int(opt.workers))
        test(opt, test_loader, 0, Hnet=Hnet, Rnet=Rnet, criterion=criterion, logpath=logpath)
        print("-------------------Test is completed-------------------")

def train(opt, train_loader, epoch, voc, embedding, text_train_dataset, Hnet, Rnet, criterion, cosinesimilarity, logpath):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    Hlosses = AverageMeter() # record the loss of each epoch Hnet
    Rlosses = AverageMeter() # record the loss of each epoch Rnet
    SumLosses = AverageMeter() # record the each epoch Hloss + β*Rloss

    # switch to train mode
    Hnet.train()
    Rnet.train()

    start_time = time.time()
    for i, data in enumerate(train_loader, 0):
        data_time.update(time.time() - start_time)

        Hnet.zero_grad()
        Rnet.zero_grad()

        all_pics = data # all pics contains coverImg and secretImg, no label needed
        this_batch_size = int(all_pics.size()[0])

        #--------------------------------------------------------------------------------------------------------------------------------
        # The first half of the picture is used as coverImg, and the second half of the picture is used as secretImg
        #cover_img = all_pics[0:this_batch_size, :, :, :] # batch_size, 3, ,256, 256
        cover_img = all_pics
        #--------------------------------------------------------------------------------------------------------------------------------
        # should change secret_img -> secret_text and secret_text has the same size with cover_img
        #secret_img = all_pics[this_batch_size:this_batch_size * 2, :, :, :]
        text_batches = batch2TrainData(voc, [random.choice(text_train_dataset) for _ in range(this_batch_size)])
        secret_text, text_lengths, target_text, mask, max_target_len = text_batches
        secret_text = embedding(secret_text)
        secret_text = secret_text.view(this_batch_size, 3, 256, 256)
        #--------------------------------------------------------------------------------------------------------------------------------


        # Concat the pictures together to get six-channel pictures as input to the Hnet
        concat_img_text = torch.cat([cover_img, secret_text], dim=1)

        # Data into gpu
        if opt.cuda:
            cover_img = cover_img.cuda()
            #secret_img = secret_img.cuda()
            #concat_img = concat_img.cuda()
            secret_text = secret_text.cuda()
            concat_img_text = concat_img_text.cuda()

        #concat_imgv = Variable(concat_img) # concat_img as input to the Hnet
        concat_img_textv = Variable(concat_img_text)
        cover_imgv = Variable(cover_img) # cover_img as label of Hnet

        #container_img = Hnet(concat_imgv) # Get container_img with secret_img
        container_img = Hnet(concat_img_textv)
        errH = criterion(container_img, cover_imgv) # Hnet reconstruction error
        Hlosses.update(errH, this_batch_size) # record H_loss value

        rev_secret_img = Rnet(container_img) # container_img is used as input to the Rnet to get rev_secret_img
        #secret_imgv = Variable(secret_img) # secret_img as the label of the Rnet
        secret_textv = Variable(secret_text)

        #errR = criterion(rev_secret_img, secret_imgv) # Rnet reconstruction error
        errR = criterion(rev_secret_img, secret_textv)
        #-----------------------------------------------------------------------------------------------------------------------------
        distance = cosinesimilarity(rev_secret_img, secret_textv)
        #-----------------------------------------------------------------------------------------------------------------------------
        Rlosses.update(errR, this_batch_size) # record R_loss value

        betaerrR_secret = opt.beta * errR
        err_sum = errH + betaerrR_secret
        SumLosses.update(err_sum, this_batch_size)

        # Calculate the gradient
        err_sum.backward()

        # Optimize the parameters of both networks
        optimizerH.step()
        optimizerR.step()

        # Update the time of a batch
        batch_time.update(time.time() - start_time)
        start_time = time.time()

        # log information
        log = '[%d/%d][%d/%d]\tLoss_H: %.4f Loss_R: %.4f Loss_sum: %.4f \tdatatime: %.4f \tbatchtime: %.4f' % (
                epoch, opt.epochs, i, len(train_loader),
                Hlosses.val, Rlosses.val, SumLosses.val, data_time.val, batch_time.val)

        # print log information
        if i % opt.logfrequency == 0:
            print_log(opt, log, logpath)
        else:
            print_log(opt, log, logpath, console=False)

        # Related operations such as storing records
        # Generate a picture in 100 steps
        if epoch % 1 == 0 and i % opt.resultpicfrequency == 0:
            save_result_pic(opt, this_batch_size, cover_img, container_img.data, secret_img, rev_secret_img.data, epoch, i,
                            opt.trainpics)

    # Time taken to output an epoch
    epoch_log = "one epoch time is %.4f==================================================" % (
        batch_time.sum) + "\n"
    epoch_log = epoch_log + "epoch learning rate: optimizerH_lr = %.8f    optimizerR_lr = %.8f" % (
        Hlosses.avg, Rlosses.avg, SumLosses.avg)
    print_log(opt, epoch_log, logpath)

    if not opt.debug:
        # record learning rate
        writer.add_scalar("lr/H_lr", optimizerH.param_groups[0]['lr'], epoch)
        writer.add_scalar("lr/R_lr", optimizerR.param_groups[0]['lr'], epoch)
        writer.add_scalar("lr/beta", opt.beta, epoch)
        # Every epoch records an average loss on tensorboard display
        writer.add_scalar("train/R_loss", Rlosses.avg, epoch)
        writer.add_scalar("train/H_loss", Hlosses.avg, epoch)
        writer.add_scalar("train/sum_loss", SumLosses.avg, epoch)

def validation(opt, val_loader, epoch, voc, text_val_dataset, Hnet, Rnet, criterion, logpath):
    print("--------------------------------------------------validation begin--------------------------------------------------")
    start_time = time.time()
    Hnet.eval()
    Rnet.eval()
    Hlosses = AverageMeter() # record the loss of each epoch Hnet
    Rlosses = AverageMeter() # record the loss of each epoch Rnet
    for i, data in enumerate(val_loader, 0):
        Hnet.zero_grad()
        Rnet.zero_grad()
        with torch.no_grad():
            all_pics = data # allpics contains coverImg and secretImg, no label needed
            this_batch_size = int(all_pics.size()[0] / 2) # Processing the last batch of each epoch may be insufficient for opt.batchsize

            # The first half of the picture is used as coverImg, and the second half of the picture is used as secretImg
            cover_img = all_pics[0:this_batch_size, :, :, :] # batchsize, 3, 256, 256
            secret_img = all_pics[this_batch_size:this_batch_size * 2, :, :, :]

            # Concat the pictures together to get six-channel pictures as input to the Hnet
            concat_img = torch.cat([cover_img, secret_img], dim=1)

            # Data into gpu
            if opt.cuda:
                cover_img = cover_img.cuda()
                secret_img = secret_img.cuda()
                concat_img = concat_img.cuda()

            concat_imgv = Variable(concat_img) # concat_img as input to the Hnet
            cover_imgv = Variable(cover_img) # cover_img as label of Hnet

            container_img = Hnet(concat_imgv) # Get container_img with secret_img
            errH = criterion(container_img, cover_imgv) # Hnet reconstruction error
            Hlosses.update(errH, this_batch_size) # record H_loss value

            rev_secret_img = Rnet(container_img) # container_img is used as input to the Rnet to get rev_secret_img
            secret_imgv = Variable(secret_img) # secret_img as the label of the Rnet
            errR = criterion(rev_secret_img, secret_imgv) # Rnet reconstruction error
            Rlosses.update(errR, this_batch_size) # record R_loss value

            if i % 50 == 0:
                save_result_pic(opt, this_batch_size, cover_img, container_img.data, secret_img, rev_secret_img.data, epoch, i,
                                opt.validationpics)

    val_hloss = Hlosses.avg
    val_rloss = Rlosses.avg
    val_sumloss = val_hloss + opt.beta * val_rloss

    val_time = time.time() - start_time
    val_log = "validation[%d] val_Hloss = %.6f\t val_Rloss = %.6f\t val_Sumloss = %.6f\t validation time=%.2f" % (
        epoch, val_hloss, val_rloss, val_sumloss, val_time)
    print_log(opt, val_log, logpath)

    if not opt.debug:
        writer.add_scalar('validation/H_loss_avg', Hlosses.avg, epoch)
        writer.add_scalar('validation/R_loss_avg', Rlosses.avg, epoch)
        writer.add_scalar('validation/sum_loss_avg', val_sumloss, epoch)

    print("--------------------------------------------------validation end--------------------------------------------------")
    return val_hloss, val_rloss, val_sumloss

def test(opt, test_loader, epoch, Hnet, Rnet, criterion, logpath):
    print("--------------------------------------------------test begin--------------------------------------------------")
    start_time = time.time()
    Hnet.eval()
    Rnet.eval()
    Hlosses = AverageMeter() # to record the Hloss in one epoch
    Rlosses = AverageMeter() # to record the Rloss in one epoch
    for i, data in enumerate(test_loader, 0):
        Hnet.zero_grad()
        Rnet.zero_grad()
        with torch.no_grad():
            all_pics = data # all_pics contain cover_img and secret_img, label is not needed
            this_batch_size = int(all_pics.size()[0] / 2) # in order to handle the final batch which may not have opt.size

            # half of the front is as cover_img, half of the end is as secret_img
            cover_img = all_pics[0:this_batch_size, :, :, :] # batchsize,3,256,256
            secret_img = all_pics[this_batch_size:this_batch_size * 2, :, :, :]

            # concat cover and original secret get the concat_img with  channels
            concat_img = torch.cat([cover_img, secret_img], dim=1)

            # data into gpu
            if opt.cuda:
                cover_img = cover_img.cuda()
                secret_img = secret_img.cuda()
                concat_img = concat_img.cuda()

            concat_imgv = Variable(concat_img) # concat_img is the input of Hnet
            cover_imgv = Variable(cover_img) # Hnet reconstruction error

            container_img = Hnet(concat_imgv) #  concat_img as the input of Hnet and get the container_img
            errH = criterion(container_img, cover_imgv) # Hnet reconstruction error
            Hlosses.update(errH, this_batch_size) # record the H loss value

            rev_secret_img = Rnet(container_img) # container_img is the input of the Rnet and get the output "rev_secret_img"
            secret_imgv = Variable(secret_img) # secret_imgv is the label of Rnet
            errR = criterion(rev_secret_img, secret_imgv) # Rnet reconstructed error
            Rlosses.update(errR, this_batch_size) # record the R_loss value
            save_result_pic(opt, this_batch_size, cover_img, container_img.data, secret_img, rev_secret_img.data, epoch, i,
                            opt.testpics)

    val_hloss = Hlosses.avg
    val_rloss = Rlosses.avg
    val_sumloss = val_hloss + opt.beta * val_rloss

    val_time = time.time() - start_time
    val_log = "validation[%d] val_Hloss = %.6f\t val_Rloss = %.6f\t val_Sumloss = %.6f\t validation time=%.2f" % (
        epoch, val_hloss, val_rloss, val_sumloss, val_time)
    print_log(opt, val_log, logpath)

    print("--------------------------------------------------test end--------------------------------------------------")
    return val_hloss, val_rloss, val_sumloss

if __name__ == '__main__':
    main()
