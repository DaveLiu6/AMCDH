# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Time       : 2021/4/2 14:53
# @Site        : xxx#2L4
# @File         : my_demo_center_similarity
# @Software : PyCharm
# @Author   : Dave Liu
# @Email    :
# @Version  : V1.1.0
-------------------------------------------------
"""

import os
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm

import numpy as np
from Data_Processing import *
from functions import *
from models import *
from parameter import *
import datetime

global run_number


def train(**kwargs):
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    opt.parse(kwargs)

    # opt.data_path = opt.data_path
    # opt.pretrain_model_path = opt.pretrain_model_path_3

    if opt.device == 'cpu':
        print('Using CPU ...')
        opt.device = torch.device('cpu')
    else:
        print('Using GPU ...')
        opt.device = torch.device(opt.device)

    print("... loading module ...")
    img_model = MAWDH(opt.bit, opt.tag_dim).cuda()
    txt_model = Text_net(opt.bit, opt.tag_dim).cuda()

    # images: (20015, 3, 224, 224)
    # text/ tags: (20015, 1386)
    # labels: (20015, 24)
    print("... loading data ...")
    images, tags, labels = loading_data(opt.data_path)

    train_data = Dataset(opt, images, tags, labels)
    train_dataloader = DataLoader(train_data, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=True, drop_last=True)

    train_L = train_data.get_labels().cuda()

    # valid or test data
    x_query_data = Dataset(opt, images, tags, labels, test='image.query')
    x_db_data = Dataset(opt, images, tags, labels, test='image.db')
    y_query_data = Dataset(opt, images, tags, labels, test='text.query')
    y_db_data = Dataset(opt, images, tags, labels, test='text.db')

    x_query_dataloader = DataLoader(x_query_data, opt.batch_size, num_workers=opt.num_workers, shuffle=False)
    x_db_dataloader = DataLoader(x_db_data, opt.batch_size, num_workers=opt.num_workers, shuffle=False)
    y_query_dataloader = DataLoader(y_query_data, opt.batch_size, num_workers=opt.num_workers, shuffle=False)
    y_db_dataloader = DataLoader(y_db_data, opt.batch_size, num_workers=opt.num_workers, shuffle=False)

    query_labels, db_labels = x_query_data.get_labels()
    query_labels = query_labels.cuda()
    db_labels = db_labels.cuda()

    img_optimizer = Adam(img_model.parameters(), lr=0.00001, weight_decay=5 * 10 ** -4)
    txt_optimizer = Adam(txt_model.parameters(), lr=0.00001)

    # txt_optimizer = Adam([
    #     {'params': txt_model.txt_module.parameters(), 'lr': 0.0001, 'weight_decay': 0},
    #     {'params': txt_model.txt_hash_module.parameters(), 'lr': 0.0001},
    #     {'params': txt_model.txt_classifier_module.parameters(), 'lr': 0.0001},
    # ])

    criterion_tri_cos = TripletAllLoss(dis_metric='cos', reduction='mean')
    criterion_bce = nn.BCEWithLogitsLoss(reduction='mean')
    criterion = nn.BCELoss().cuda()

    loss = []
    x_loss = {}
    y_loss = {}

    x_loss['precision'] = []  # 精确率
    x_loss['recall'] = []  # 召回率
    x_loss['accuracy'] = []  # 准确率

    y_loss['precision'] = []  # 精确率
    y_loss['recall'] = []  # 召回率
    y_loss['accuracy'] = []  # 准确率

    max_mapi2t = 0.
    max_mapt2i = 0.
    max_mapi2i = 0.
    max_mapt2t = 0.
    max_epoch = 0

    U = torch.randn(opt.training_size, opt.bit).cuda()
    B = torch.sign(U)

    FEATURE_MAP = torch.randn(opt.num_label, opt.emb_dim).cuda()
    # CODE_MAP = torch.sign(torch.randn(opt.num_label, opt.bit)).cuda()

    # Hash_center = torch.load(opt.hash_center_path)
    #
    # global random_center
    # random_center = torch.randint_like(Hash_center[0], 2).cuda()
    # random_center = random_center.detach().cpu().numpy()

    mapt2i_list = []
    mapi2t_list = []
    X_buffer = torch.randn(opt.training_size, opt.bit).cuda()
    Y_buffer = torch.randn(opt.training_size, opt.bit).cuda()
    B = torch.sign(X_buffer + Y_buffer)

    # train

    # add by xh
    # ================ write run result in txt =========================

    run_result = open("{num1}_run_result_{num2}bit_{num3}.txt".format(num1=opt.dataset, num2=opt.bit, num3=run_number),
                      "a")

    run_result.write("Data_set: {num1}, Hash_length: {num2} bit".format(num1=opt.dataset, num2=opt.bit) + '\n' * 2)
    run_result.close()

    # ==================================================================

    # train

    LossI2T_list = []
    LossT2I_list = []
    for epoch in range(50):
        print('\n==============================================================')
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("Nowtime:" + "%s" % nowtime)

        x_loss['precision'] = []  # 精确率
        x_loss['recall'] = []  # 召回率
        x_loss['accuracy'] = []  # 准确率

        y_loss['precision'] = []
        y_loss['recall'] = []
        y_loss['accuracy'] = []

        for i, (ind, x, y, l) in enumerate(train_dataloader, 0):

            # imgs = x.numpy()
            # imgs = imgs.transpose(0, 3, 2, 1)
            # imgs = torch.from_numpy(imgs)
            imgs = x.cuda()
            tags = y.cuda()
            labels = l.cuda()
            # ind = ind - 1


            img_optimizer.zero_grad()
            txt_optimizer.zero_grad()

            h_x = img_model(imgs)
            h_y = txt_model(tags)

            S = calc_neighbor(labels, train_L)
            S1 = jaccard_similarity(labels, train_L)

            X_buffer[ind, :] = h_x.data
            Y_buffer[ind, :] = h_y.data

            X = Variable(X_buffer)
            Y = Variable(Y_buffer)
            # ===================  hash_center_similarity ==========================================
            my_h_center = my_learn_hash_center(h_x)
            # my_h_center = torch.from_numpy(my_h_center).cuda()
            hash_center = my_multi_hash_cneter(labels, my_h_center)

            # hash_center = Variable(hash_center).cuda()

            center_loss_x = criterion(0.5 * (h_x + 1), 0.5 * (hash_center.detach() + 1))
            center_loss_y = criterion(0.5 * (h_y + 1), 0.5 * (hash_center.detach() + 1))


            # ======================================================================================

            # ============================= log_loss ===============================================

            theta_xy = calc_inner(h_x, Y)
            logloss_xy = torch.sum(torch.mul(torch.pow(S - theta_xy, 2), torch.exp(S1))) / (opt.training_size * opt.batch_size)
            theta_xx = calc_inner(h_x, X)
            logloss_xx = torch.sum(torch.pow(S - theta_xx, 2)) / (opt.training_size * opt.batch_size)
            loss_x = logloss_xy + logloss_xx

            # theta_xx = 1.0 / 2 * torch.matmul(h_x, Y)


            # loss_x = 10 * loss_x

            theta_yx = calc_inner(h_y, X)
            logloss_yx = torch.sum(torch.mul(torch.pow(S - theta_yx, 2), torch.exp(S1))) / (opt.training_size * opt.batch_size)
            theta_yy = calc_inner(h_y, Y)
            logloss_yy = torch.sum(torch.pow(S - theta_yy, 2)) / (opt.training_size * opt.batch_size)
            loss_y = logloss_yx + logloss_yy
            # loss_y = 10 * loss_y

            # loss_quant_and_log = loss_x + loss_y

            # ======================================================================================

            loss_triplet_xy = criterion_tri_cos(h_x, labels, target=h_y.detach(), t_labels=labels, margin=opt.margin)
            loss_triplet_xx = criterion_tri_cos(h_x, labels, target=h_x, t_labels=labels, margin=opt.margin)
            quantization_x = torch.mean(torch.pow(1 / 2. * (B[ind, :] - h_x), 2))
            loss_triplet_x = loss_triplet_xx + loss_triplet_xy
            # loss_classifier_x = criterion_bce(x_class, labels)

            loss_triplet_yx = criterion_tri_cos(h_y, labels, target=h_x.detach(), t_labels=labels, margin=opt.margin)
            loss_triplet_yy = criterion_tri_cos(h_y, labels, target=h_y, t_labels=labels, margin=opt.margin)
            quantization_y = torch.mean(torch.pow(1 / 2. * (B[ind, :] - h_y), 2))
            loss_triplet_y = loss_triplet_yx + loss_triplet_yy
            # loss_classifier_y = criterion_bce(y_class, labels)
            # loss_difference_y = torch.mean(torch.pow(f_x.detach() - f_y, 2))

            # img_loss = loss_triplet_xy + loss_triplet_xx + quantization_x + center_loss_x + loss_x
            # txt_loss = loss_triplet_yx + loss_triplet_yy + quantization_y + center_loss_y + loss_y
            # img_loss =  quantization_x + center_loss_x + loss_x
            # txt_loss =  quantization_y + center_loss_y + loss_y
            # img_loss = center_similarity_x
            # txt_loss = center_similarity_y
            img_loss = loss_triplet_x + quantization_x + center_loss_x + loss_x
            txt_loss = loss_triplet_y + quantization_y + center_loss_y + loss_y

            img_loss.backward()
            txt_loss.backward()
            img_optimizer.step()
            txt_optimizer.step()

            # U[ind] = h_x.detach()

        B = torch.sign(X_buffer + Y_buffer)

        # print('...epoch: %3d' % (epoch + 1))
        # validate
        # if (epoch + 1) % 2 == 0 or epoch == 0:
        # print(img_model.weight)
        mapi2t, mapt2i, mapi2i, mapt2t, qBX, qBY, rBX, rBY = valid_retrieval(img_model, txt_model, x_query_dataloader,
                                                         x_db_dataloader, y_query_dataloader,
                                                         y_db_dataloader,
                                                         query_labels, db_labels, FEATURE_MAP)
        mapi2t_list.append(mapi2t)
        mapt2i_list.append(mapt2i)

        LossI2T_list.append(img_loss.detach().cpu().numpy())
        LossT2I_list.append(txt_loss.detach().cpu().numpy())

        if mapt2i >= max_mapt2i and mapi2t >= max_mapi2t:
            if mapt2i >= 0.80 and mapi2t >= 0.80:

                dct = {'qBX': qBX.cpu().numpy(), 'qBY': qBY.cpu().numpy(), 'rBX': rBX.cpu().numpy(),
                       'rBY': rBY.cpu().numpy(), 'query_L': query_labels.cpu().numpy(),
                       'retrieval_L': db_labels.cpu().numpy(),
                       'mapi2t': mapi2t, 'mapt2i': mapt2i, 'lossi2t':img_loss.detach().cpu().numpy(),'losst2i':txt_loss.detach().cpu().numpy()}
                outfile_1 = 'result_{num1}_{num2}bit_{num3}.npz'.format(num1=opt.dataset, num2=opt.bit, num3=run_number)
                np.savez(outfile_1, **dct)
            max_mapi2t = mapi2t
            max_mapt2i = mapt2i
            max_mapi2i = mapi2i
            max_mapt2t = mapt2t
            max_epoch = epoch + 1

        dect = {'mapi2t_list': mapi2t_list, 'mapt2i_list': mapt2i_list, 'LossI2T_list': LossI2T_list,
                'LossT2I_list': LossT2I_list}
        outfile_2 = 'AMCDH_MIR16bit.npz'
        np.savez(outfile_2, **dect)

        print('...epoch: %3d, valid MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (epoch + 1, mapi2t, mapt2i))
        print('...epoch: %3d, valid MAP: MAP(i->i): %3.4f, MAP(t->t): %3.4f' % (epoch + 1, mapi2i, mapt2t))

        print('   max epoch: %3d' % max_epoch)
        print('   max MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (max_mapi2t, max_mapt2i))
        print('   max MAP: MAP(i->i): %3.4f, MAP(t->t): %3.4f' % (max_mapi2i, max_mapt2t))

        # add by xh
        # =========================== weite result in .txt =============================================

        run_result = open(
            "{num1}_run_result_{num2}bit_{num3}.txt".format(num1=opt.dataset, num2=opt.bit, num3=run_number), "a")

        # run_result.write("Data_set: Fir_25K, Hash_length: 16 bit" + '\n' * 2)

        run_result.write("...epoch: {epoch}, valid MAP: MAP(i->t): {mapit}, MAP(t->i): {mapti}".format(epoch=epoch + 1,
                                                                                                       mapit='%.4f' % mapi2t,
                                                                                                       mapti='%.4f' % mapt2i) + '\n')
        run_result.write("...epoch: {epoch}, valid MAP: MAP(i->i): {mapit}, MAP(t->t): {mapti}".format(epoch=epoch + 1,
                                                                                                       mapit='%.4f' % mapi2i,
                                                                                                       mapti='%.4f' % mapt2t) + '\n' * 2)
        run_result.write('Max epoch: {epoch}'.format(epoch=max_epoch) + '\n')

        run_result.write('Max MAP: MAP(i->t): {max_i2t}, MAP(t->i): {max_t2i}'.format(max_i2t='%.4f' % max_mapi2t,
                                                                                      max_t2i='%.4f' % max_mapt2i) + '\n')
        run_result.write('Max MAP: MAP(i->i): {max_i2t}, MAP(t->t): {max_t2i}'.format(max_i2t='%.4f' % max_mapi2i,
                                                                                      max_t2i='%.4f' % max_mapt2t) + '\n')
        run_result.write('\n' * 2)

        run_result.close()

        # ====================================================

        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("Nowtime:" + "%s" % nowtime)
        print('==============================================================\n')

    if (epoch + 1) % 5 == 0:
            for params in img_optimizer.param_groups:
                params['lr'] *= 0.9
            for params in txt_optimizer.param_groups:
                params['lr'] *= 0.9

    print('...training procedure finish')
    if opt.valid:
        print('   max epoch: %3d' % max_epoch)
        print('   max MAP: MAP(i->t): %3.4f, MAP(t->i): %3.4f' % (max_mapi2t, max_mapt2i))
        print('   max MAP: MAP(i->i): %3.4f, MAP(t->t): %3.4f' % (max_mapi2i, max_mapt2t))


def help():
    """
    打印帮助的信息： python file.py help
    """
    print('''========================::HELP::=========================
        usage : python file.py <function> [--args=value]
        <function> := train | test | help
        example:
                python {0} train --lr=0.01
                python {0} help
        avaiable args (default value):'''.format(__file__))
    for k, v in opt.__class__.__dict__.items():
        if not k.startswith('__') and str(k) != 'parse':
            print('            {0}: {1}'.format(k, v))
    print('========================::HELP::=========================')

def my_multi_hash_cneter(labels, Hash_center):
    is_start = True
    for label in labels:
        one_labels = torch.nonzero(label == 1).squeeze(1)
        Center_mean = torch.mean(Hash_center[one_labels], dim=0)
        Center_mean = torch.sign(Center_mean)
        Center_mean[Center_mean == 0] = 1
        Center_mean = Center_mean.view(1, -1)  # shape:[1,hash_bit]

        if is_start:  # the first time
            hash_center = Center_mean
            is_start = False
        else:
            hash_center = torch.cat((hash_center, Center_mean), 0)
            # hash_center = torch.stack((hash_center, Center_mean), dim=0)

    return hash_center



def pairwise_loss(outputs1, outputs2, label1, label2, sigmoid_param=1, data_imbalance=1):
    similarity = Variable(torch.mm(label1.data.float(), label2.data.float().t()) > 0).float()
    dot_product = sigmoid_param * torch.mm(outputs1, outputs2.t())
    exp_product = torch.exp(dot_product)

    exp_loss = (torch.log(1 + exp_product) - similarity * dot_product)
    loss = torch.mean(exp_loss)

    return loss



def my_learn_hash_center(hash_input):
    # hash_input = hash_input.detach().cpu().numpy()
    # row = len(hash_input)
    # column = len(hash_input[0])
    # for i in range(row):
    #     for j in range(column):
    #         if hash_input[i][j] > 0:
    #             hash_input[i][j] = 1
    #         if hash_input[i][j] < 0:
    #             hash_input[i][j] = -1
    # learn_center = hash_input
    learn_center = torch.sign(hash_input)

    return learn_center[0: opt.num_label]



if __name__ == '__main__':
    run_number = 1208
    train()

