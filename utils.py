import torch
import math
import random
import time
import os
import shutil

from PIL import Image, ImageOps, ImageEnhance

try:
    import accimage
except ImportError:
    accimage = None
import numpy as np
import numbers
import types
import collections
import warnings
import torchvision.utils as vutils

# custom weight initialization called on netG and netD
def weights_init(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        model.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        model.weight.data.normal_(1.0, 0.02)
        model.bias.data.fill_(0)

# print the training log and save info logfiles
def print_log(opt, log_info, log_path, console=True):
    #print the info into the console
    if console:
        print(log_info)
    # debug mode don't write the log info files
    if not opt.debug:
        #write the log into log file
        if not os.path.exists(log_path):
            fp = open(log_path, "w")
            fp.writelines(log_info + "\n")
        else:
            with open(log_path, 'a+') as f:
                f.writelines(log_info + '\n')

# print the structure and parameters number of the net
def print_network(opt, net, logpath):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print_log(opt, str(net), logpath)
    print_log(opt, 'Total number of parameters: %d' % num_params, logpath)

def save_current_codes(dest_path):
    main_file_path = os.path.realpath(__file__) # eg: /home/changmin/research/steganography/project/main.py
    cur_work_dir, mainfile = os.path.split(main_file_path) # eg: /home/changmin/research/steganography/project, main.py

    new_main_path = os.path.join(dest_path, mainfile)
    shutil.copyfile(main_file_path, new_main_path)

    data_dir = cur_work_dir + "/data/"
    new_data_dir_path = dest_path + "/data/"
    shutil.copytree(data_dir, new_data_dir_path)

    model_dir = cur_work_dir + "/models/"
    new_model_dir_path = dest_path + "/models/"
    shutil.copytree(model_dir, new_model_dir_path)

    utils_dir = cur_work_dir + "/utils/"
    new_utils_dir_path = dest_path + "/utils/"
    shutil.copytree(utils_dir, new_utils_dir_path)

def save_result_pic(opt, this_batch_size, originalLabelv, ContainerImg, secretLabelv, RevSecImg, epoch, i, save_path):
    if not opt.debug:
        originalFrames = originalLabelv.resize_(this_batch_size, 3, opt.imagesize, opt.imagesize)
        containerFrames = ContainerImg.resize_(this_batch_size, 3, opt.imagesize, opt.imagesize)
        secretFrames = secretLabelv.resize_(this_batch_size, 3, opt.imagesize, opt.imagesize)
        revSecFrames = RevSecImg.resize_(this_batch_size, 3, opt.imagesize, opt.imagesize)

        showContainer = torch.cat([originalFrames, containerFrames], 0)
        showReveal = torch.cat([secretFrames, revSecFrames], 0)
        # resultImg contains four rows, each row is coverImg containerImg secretImg RevSecImg, total this_batch_size columns
        resultImg = torch.cat([showContainer, showReveal], 0)
        resultImgName = '%s/ResultPics_epoch%03d_batch%04d.png' % (save_path, epoch, i)
        vutils.save_image(resultImg, resultImgName, nrow=this_batch_size, padding=1, normalize=True)

def create_dir_to_save_result(opt):
    if not opt.debug:
        try:
            cur_time = time.strftime('%Y-%m-%d-%H_%M_%S', time.localtime())
            experiment_dir = opt.hostname + "_" + cur_time + opt.remark
            opt.outckpts += experiment_dir + "/checkPoints"
            opt.trainpics += experiment_dir + "/trainPics"
            opt.validationpics += experiment_dir + "/validationPics"
            opt.outlogs += experiment_dir + "/trainingLogs"
            opt.outcodes += experiment_dir + "/codes"
            opt.testpics += experiment_dir + "/testPics"
            if not os.path.exists(opt.outckpts):
                os.makedirs(opt.outckpts)
            if not os.path.exists(opt.trainpics):
                os.makedirs(opt.trainpics)
            if not os.path.exists(opt.validationpics):
                os.makedirs(opt.validationpics)
            if not os.path.exists(opt.outlogs):
                os.makedirs(opt.outlogs)
            if not os.path.exists(opt.outcodes):
                os.makedirs(opt.outcodes)
            if (not os.path.exists(opt.testpics)) and opt.test != '':
                os.makedirs(opt.testpics)

        except OSError:
            print("mkdir failed!")

class AverageMeter(object):
    """
    Computes and stores the average and current value.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
