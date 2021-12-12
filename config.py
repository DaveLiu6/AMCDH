# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Time       : 2021/4/1 18:41
# @Site        : xxx#2L4
# @File         : config
# @Software : PyCharm
# @Author   : Dave Liu
# @Email    :
# @Version  : V1.1.0
------------------------------------------------- 
"""
import warnings


class DefaultConfig(object):
    # visualization
    vis_env = None  # visdom env
    vis_port = 8097  # visdom port

    # for flickr25k dataset
    dataset = 'flickr25k'

    data_path = '/ssd2t/linqiubin/Datasets/DCMH/mirflickr-25k/'

    pretrain_model_path = ''

    db_size = 18015
    num_label = 24
    tag_dim = 1386

    num_workers = 8

    # for nus-wide dataset
    # dataset = 'nus-wide'
    # data_path = './data/dcmh-nus-wide'
    # db_size = 193734
    # num_label = 21
    # tag_dim = 1000
    # query_size = 2100

    # added by Qiubin
    load_model_path = None  # load model path

    # data parameters
    training_size = 10000
    query_size = 2000
    database_size = 18015

    batch_size = 32

    emb_dim = 512

    valid = True  # whether to use validation
    valid_freq = 10
    max_epoch = 300

    bit = 16  # final binary code length
    lr = 0.0001  # initial learning rate

    #  =============================== hash_center parameter =============================

    hash_center_path = 'generate_hash_centers/' + str(bit) + '_mir_wide_24_class.pkl'

    two_loss_epoch = -1
    data_imbalance = 5

    # ====================================================================================

    device = 'cuda:0'  # if `device` is not None then use cpu for default

    # hyper-parameters

    lambda0 = 1
    lambda1 = 0.2
    lambda2 = 0.05

    alpha = 1
    beta = 0
    gamma = 0.001
    eta = 1
    mu = 1
    delta = 0.5

    lambd = 0.8
    margin = 0.3

    # for program debug
    debug = False
    data_enhance = False

    def parse(self, kwargs):
        """
        update configuration by kwargs.
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Waning: opt has no attribute %s" % k)
            setattr(self, k, v)

        print('Configuration:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__') and str(k) != 'parse':
                print('\t{0}: {1}'.format(k, getattr(self, k)))


opt = DefaultConfig()
