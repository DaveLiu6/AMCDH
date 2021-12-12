# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Time       : 2021/4/1 18:41
# @Site        : xxx#2L4
# @File         : functions
# @Software : PyCharm
# @Author   : Dave Liu
# @Email    :
# @Version  : V1.1.0
-------------------------------------------------
"""
# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Time       : 2021/2/27 20:55
# @Site        : xxx#2L4
# @File         : Functions
# @Software : PyCharm
# @Author   : Dave Liu
# @Email    :
# @Version  : V1.1.0
------------------------------------------------- 
"""
import torch
from PIL import Image
import numpy as np
#import visdom
import time
import scipy.io as sio
import os
from parameter import *
from tqdm import tqdm


def calc_similarity(label1, label2):
    # calculate the similarity matrix
    if opt.device == 'cpu':
        Sim = (label1.matmul(label2.transpose(0, 1)) > 0).type(torch.FloatTensor)
    else:
        Sim = (label1.matmul(label2.transpose(0, 1)) > 0).type(torch.cuda.FloatTensor)
    return Sim


# 计算精确率，召回率
# 设定一个阈值，当预测的概率值大于这个阈值，则认为这幅图像中含有这类标签
def calculate_precision_recall(model_pred, labels):
    # 注意这里的model_pred是经过sigmoid处理的，sigmoid处理后可以视为预测是这一类的概率
    # 预测结果，大于这个阈值则视为预测正确
    accuracy_th = 0.5
    pred_result = model_pred > accuracy_th
    pred_result = pred_result.float()
    pred_one_num = torch.sum(pred_result)
    if pred_one_num == 0:
        return 0, 0
    target_one_num = torch.sum(labels)  # 真实的标签的数目
    true_predict_num = torch.sum(pred_result * labels)  # 预测中真实的标签
    # 模型预测的结果中有多少个是正确的
    precision = true_predict_num / pred_one_num
    # 模型预测正确的结果中，占所有真实标签的数量
    recall = true_predict_num / target_one_num

    return precision.item(), recall.item()


# 计算准确率
def calculate_accuracy(model_pred, labels):
    # 注意这里的model_pred是经过sigmoid处理的，sigmoid处理后可以视为预测是这一类的概率
    # 预测结果，大于这个阈值则视为预测正确
    accuracy = 0.
    accuracy_th = 0.5
    pred_result = model_pred > accuracy_th
    pred_result = pred_result.float()
    accuracy += (pred_result == labels).sum().float()

    return (accuracy / (labels.shape[0] * labels.shape[1])).item()


def update_code_map(U, V, M, L):
    CODE_MAP = M
    U = torch.sign(U)
    V = torch.sign(V)
    S = torch.eye(opt.num_label).cuda() * 2 - 1

    Q = 2 * opt.bit * (L.t().mm(U + V) + S.mm(M))

    for k in range(opt.bit):
        ind = np.setdiff1d(np.arange(0, opt.bit), k)
        term1 = CODE_MAP[:, ind].mm(U[:, ind].t()).mm(U[:, k].unsqueeze(-1)).squeeze()
        term2 = CODE_MAP[:, ind].mm(V[:, ind].t()).mm(V[:, k].unsqueeze(-1)).squeeze()
        term3 = CODE_MAP[:, ind].mm(M[:, ind].t()).mm(M[:, k].unsqueeze(-1)).squeeze()
        CODE_MAP[:, k] = torch.sign(Q[:, k] - 2 * (term1 + term2 + term3))

    return CODE_MAP


def update_feature_map(FEAT_I, FEAT_T, L, mode='average'):
    if mode is 'average':
        feature_map_I = L.t().mm(FEAT_I) / L.sum(dim=0).unsqueeze(-1)
        feature_map_T = L.t().mm(FEAT_T) / L.sum(dim=0).unsqueeze(-1)
    else:
        assert mode is 'max'
        feature_map_I = (L.t().unsqueeze(-1) * FEAT_I).max(dim=1)[0]
        feature_map_T = (L.t().unsqueeze(-1) * FEAT_T).max(dim=1)[0]

    FEATURE_MAP = (feature_map_T + feature_map_I) / 2
    # normalization
    FEATURE_MAP = FEATURE_MAP / torch.sqrt(torch.sum(FEATURE_MAP ** 2, dim=-1, keepdim=True))
    return FEATURE_MAP


def valid_retrieval(img_model, txt_model, x_query_dataloader, x_db_dataloader, y_query_dataloader, y_db_dataloader,
                    query_labels, db_labels, FEATURE_MAP):
    img_model.eval()
    txt_model.eval()
    qBX = generate_img_code(img_model, x_query_dataloader, opt.query_size, FEATURE_MAP)
    qBY = generate_txt_code(txt_model, y_query_dataloader, opt.query_size, FEATURE_MAP)
    rBX = generate_img_code(img_model, x_db_dataloader, opt.db_size, FEATURE_MAP)
    rBY = generate_txt_code(txt_model, y_db_dataloader, opt.db_size, FEATURE_MAP)

    mapi2t = calc_map_k(qBX, rBY, query_labels, db_labels)
    mapt2i = calc_map_k(qBY, rBX, query_labels, db_labels)
    mapi2i = calc_map_k(qBX, rBX, query_labels, db_labels)
    mapt2t = calc_map_k(qBY, rBY, query_labels, db_labels)

    # K = [1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    # pk_i2t = p_topK(qBX, rBY, query_labels, db_labels, K)
    # pk_t2i = p_topK(qBY, rBX, query_labels, db_labels, K)

    img_model.train()
    txt_model.train()
    # return mapi2t.item(), mapt2i.item(), mapi2i.item(), mapt2t.item(), pk_i2t.numpy(), pk_t2i.numpy()
    return mapi2t.item(), mapt2i.item(), mapi2i.item(), mapt2t.item(), qBX, qBY, rBX, rBY


def valid_classification(model, x_query_dataloader, x_db_dataloader, y_query_dataloader, y_db_dataloader,
                         query_labels, db_labels):
    model.eval()
    qCX = predict_img_class(model, x_query_dataloader, opt.query_size)
    qCY = predict_txt_class(model, y_query_dataloader, opt.query_size)
    rCX = predict_img_class(model, x_db_dataloader, opt.db_size)
    rCY = predict_txt_class(model, y_db_dataloader, opt.db_size)

    qCX_precision, qCX_recall = calculate_precision_recall(qCX, query_labels)
    qCY_precision, qCY_recall = calculate_precision_recall(qCY, query_labels)
    rCX_precision, rCX_recall = calculate_precision_recall(rCX, db_labels)
    rCY_precision, rCY_recall = calculate_precision_recall(rCY, db_labels)

    qCX_accuracy = calculate_accuracy(qCX, query_labels)
    qCY_accuracy = calculate_accuracy(qCY, query_labels)
    rCX_accuracy = calculate_accuracy(rCX, db_labels)
    rCY_accuracy = calculate_accuracy(rCY, db_labels)

    print('Query dataset:')
    print(
        "image precision: {0:.4f}, recall: {1:.4f}, accuracy: {2:.4f}".format(qCX_precision, qCX_recall, qCX_accuracy))
    print("text precision: {0:.4f}, recall: {1:.4f}, accuracy: {2:.4f}".format(qCY_precision, qCY_recall, qCY_accuracy))

    print('Retrieval dataset:')
    print(
        "image precision: {0:.4f}, recall: {1:.4f}, accuracy: {2:.4f}".format(rCX_precision, rCX_recall, rCX_accuracy))
    print("text precision: {0:.4f}, recall: {1:.4f}, accuracy: {2:.4f}".format(rCY_precision, rCY_recall, rCY_accuracy))

    model.train()





def generate_img_code(model, test_dataloader, num, FEATURE_MAP):
    B = torch.zeros(num, opt.bit).cuda()

    for i, input_data in enumerate(test_dataloader):
        input_data = input_data.cuda()
        b = model(input_data)
        idx_end = min(num, (i + 1) * opt.batch_size)
        B[i * opt.batch_size: idx_end, :] = b.data

    B = torch.sign(B)
    return B


def generate_txt_code(model, test_dataloader, num, FEATURE_MAP):
    B = torch.zeros(num, opt.bit).cuda()

    for i, input_data in enumerate(test_dataloader):
        input_data = input_data.cuda()
        b = model(input_data)
        idx_end = min(num, (i + 1) * opt.batch_size)
        B[i * opt.batch_size: idx_end, :] = b.data

    B = torch.sign(B)
    return B


def predict_img_class(model, test_dataloader, num):
    C = torch.zeros(num, opt.num_label).cuda()

    for i, input_data in enumerate(test_dataloader):
        input_data = input_data.cuda()
        c = model.predict_img_class(input_data)
        idx_end = min(num, (i + 1) * opt.batch_size)
        C[i * opt.batch_size: idx_end, :] = c.data

    return C


def predict_txt_class(model, test_dataloader, num):
    C = torch.zeros(num, opt.num_label).cuda()

    for i, input_data in enumerate(test_dataloader):
        input_data = input_data.cuda()
        c = model.predict_txt_class(input_data)
        idx_end = min(num, (i + 1) * opt.batch_size)
        C[i * opt.batch_size: idx_end, :] = c.data

    return C


def calc_loss(loss):
    l = 0.
    for v in loss.values():
        l += v[-1]
    return l


def avoid_inf(x):
    return torch.log(1.0 + torch.exp(-torch.abs(x))) + torch.max(torch.zeros_like(x), x)


def load_model(model, path):
    if path is not None:
        model.load(os.path.join(path, model.module_name + '.pth'))


def save_model(model):
    path = 'checkpoints/' + opt.dataset + '_' + str(opt.bit)
    model.save(model.module_name + '.pth', path, cuda_device=opt.device)


def calc_hamming_dist(B1, B2):
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (q - B1.mm(B2.t()))
    return distH


def calc_map_k(qB, rB, query_label, retrieval_label, k=None):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # sim: {0, 1}^{mxn}
    num_query = query_label.shape[0]
    map = 0.
    if k is None:
        k = retrieval_label.shape[0]
    for iter in range(num_query):
        gnd = (query_label[iter].unsqueeze(0).mm(retrieval_label.t()) > 0).type(torch.float).squeeze()
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hamming_dist(qB[iter, :], rB)
        _, ind = torch.sort(hamm)
        ind.squeeze_()
        gnd = gnd[ind]
        total = min(k, int(tsum))
        count = torch.arange(1, total + 1).type(torch.float).to(gnd.device)
        tindex = torch.nonzero(gnd)[:total].squeeze().type(torch.float) + 1.0
        map += torch.mean(count / tindex)
    map = map / num_query
    return map


def image_from_numpy(x):
    if x.max() > 1.0:
        x = x / 255
    if type(x) != np.ndarray:
        x = x.numpy()
    im = Image.fromarray(np.uint8(x * 255))
    im.show()


def pr_curve(qB, rB, query_label, retrieval_label):
    num_query = qB.shape[0]
    num_bit = qB.shape[1]
    P = torch.zeros(num_query, num_bit + 1)
    R = torch.zeros(num_query, num_bit + 1)
    for i in range(num_query):
        gnd = (query_label[i].unsqueeze(0).mm(retrieval_label.t()) > 0).float().squeeze()
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hamming_dist(qB[i, :], rB)
        tmp = (hamm <= torch.arange(0, num_bit + 1).reshape(-1, 1).float().to(hamm.device)).float()
        total = tmp.sum(dim=-1)
        total = total + (total == 0).float() * 0.1
        t = gnd * tmp
        count = t.sum(dim=-1)
        p = count / total
        r = count / tsum
        P[i] = p
        R[i] = r
    mask = (P > 0).float().sum(dim=0)
    mask = mask + (mask == 0).float() * 0.1
    P = P.sum(dim=0) / mask
    R = R.sum(dim=0) / mask
    return P, R


def p_topK(qB, rB, query_label, retrieval_label, K):
    num_query = query_label.shape[0]
    p = [0] * len(K)
    for iter in range(num_query):
        gnd = (query_label[iter].unsqueeze(0).mm(retrieval_label.t()) > 0).float().squeeze()
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hamming_dist(qB[iter, :], rB).squeeze()
        for i in range(len(K)):
            total = min(K[i], retrieval_label.shape[0])
            ind = torch.sort(hamm)[1][:total]
            gnd_ = gnd[ind]
            p[i] += gnd_.sum() / total
    p = torch.Tensor(p) / num_query
    return p


class Visualizer(object):
    """
    封装了visdom的基本操作，但是你仍然可以通过`self.vis.function`
    调用原生的visdom接口
    """

    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, use_incoming_socket=False, **kwargs)

        # 画的第几个数，相当于横座标
        # 保存（’loss',23） 即loss的第23个点
        self.index = {}
        self.log_text = ''

    def reinit(self, env='default', **kwargs):
        """
        修改visdom的配置
        """
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self

    def plot_many(self, d):
        """
        一次plot多个
        @params d: dict (name,value) i.e. ('loss',0.11)
        """
        for k, v in d.items():
            self.plot(k, v)

    def img_many(self, d):
        for k, v in d.items():
            self.img(k, v)

    def plot(self, name, y, **kwargs):
        """
        self.plot('loss',1.00)
        """
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=name, opts=dict(title=name),
                      update=None if x == 0 else 'append',
                      **kwargs
                      )
        self.index[name] = x + 1

    def img(self, name, img_, **kwargs):
        """
        self.img('input_img',t.Tensor(64,64))
        self.img('input_imgs',t.Tensor(3,64,64))
        self.img('input_imgs',t.Tensor(100,1,64,64))
        self.img('input_imgs',t.Tensor(100,3,64,64),nrows=10)
        ！！！don‘t ~~self.img('input_imgs',t.Tensor(100,64,64),nrows=10)~~！！！
        """
        self.vis.images(img_.cpu().numpy(),
                        win=name,
                        opts=dict(title=name),
                        **kwargs
                        )

    def log(self, info, win='log_text'):
        """
        self.log({'loss':1,'lr':0.0001})
        """

        self.log_text += ('[{time}] {info} <br>'.format(
            time=time.strftime('%m%d_%H%M%S'),
            info=info))
        self.vis.text(self.log_text, win)

    def __getattr__(self, name):
        return getattr(self.vis, name)

def myNormalization(X):
    # a = torch.sum(torch.pow(X, 2), 1)
    # print(a)
    # b = torch.sqrt(a)
    # print('b', b)
    x1 = torch.sqrt(torch.sum(torch.pow(X, 2), 1)).unsqueeze(1)
    return X / x1


def calc_inner(X1, X2):
    X1 = myNormalization(X1)
    X2 = myNormalization(X2)
    X = torch.matmul(X1, X2.t())  # [-1,1]

    return X
def calc_neighbor(label1, label2):
    # calculate the similar matrix

    label1 = label1.float()
    label2 = label2.float()
    Sim = label1.matmul(label2.transpose(0, 1)).type(torch.cuda.FloatTensor)

    numLabel_label1 = torch.sum(label1, 1)
    numLabel_label2 = torch.sum(label2, 1)

    x = numLabel_label1.unsqueeze(1) + numLabel_label2.unsqueeze(0) - Sim
    Sim = 2 * Sim / x  # [0,2]

    # cosine similarity
    # label1 = myNormalization(label1)
    # label2 = myNormalization(label2)
    # Sim = (label1.unsqueeze(1) * label2.unsqueeze(0)).sum(dim=2)

    # print(torch.max(Sim))
    # print(torch.min(Sim))
    return Sim

def jaccard_similarity(a, b):

    # a = (a.detach().cpu().numpy()).astype(np.float32)
    # b = (b.detach().cpu().numpy()).astype(np.float32)

    x = torch.sum(a, dim=-1)
    x = torch.unsqueeze(x, dim=0)
    num1 = b.shape[0]
    x = x.repeat((num1, 1))
    x = torch.t(x)
    #print(x)
    y = torch.sum(b, dim=-1)
    y = torch.unsqueeze(y, dim=0)
    num2 = a.shape[0]
    y = y.repeat(num2, 1)

    #print(y)

    # z = (np.dot(a, b.t())).astype(np.float32)
    z =(np.dot(a.cpu().numpy(), b.cpu().numpy().transpose()))
    z = torch.from_numpy(z).cuda()
    #print(z)

    #jacc = tf.convert_to_tensor(z / (x + y - z), dtype=tf.float32)
    #jacc = tf.convert_to_tensor(tf.divide(z, tf.subtract(tf.add(x, y), z)), dtype=tf.float32)
    # jacc = (np.divide(z, np.subtract(np.add(x, y), z))).astype(np.float32)
    jacc = torch.div(z,torch.sub(torch.add(x, y), z))
    #print(jacc)

    return jacc

#  theta_xy = 1.0 / 2 * torch.matmul(h_x, Y.t())
# # logloss_xy = torch.sum(torch.mul(torch.pow(S - theta_xy, 2), torch.exp(S1))) / (opt.training_size * opt.batch_size)
# logloss_xy = torch.sum(torch.mul((-torch.mul(S, theta_xy) + torch.log(1.0 + torch.exp(theta_xy))), torch.exp(S1))) / (opt.training_size * opt.batch_size)
# theta_xx = 1.0 / 2 * torch.matmul(h_x, X.t())
# # logloss_xx = torch.sum(torch.pow(S - theta_xx, 2)) / (opt.training_size * opt.batch_size)
# logloss_xx = torch.sum(
#     torch.mul((-torch.mul(S, theta_xx) + torch.log(1.0 + torch.exp(theta_xx))), torch.exp(S1))) / (opt.training_size * opt.batch_size)
# loss_x = logloss_xy + logloss_xx
#
# # theta_xx = 1.0 / 2 * torch.matmul(h_x, Y)
#
#
# # loss_x = 10 * loss_x
#
# theta_yx = 1.0 / 2 * torch.matmul(h_y, X.t())
# # logloss_yx = torch.sum(torch.mul(torch.pow(S - theta_yx, 2), torch.exp(S1))) / (opt.training_size * opt.batch_size)
# logloss_yx = torch.sum(
#     torch.mul((-torch.mul(S, theta_yx) + torch.log(1.0 + torch.exp(theta_yx))), torch.exp(S1))) / (opt.training_size * opt.batch_size)
#
# theta_yy = 1.0 / 2 * torch.matmul(h_y, Y.t())
# # logloss_yy = torch.sum(torch.pow(S - theta_yy, 2)) / (opt.training_size * opt.batch_size)
# logloss_yy = torch.sum(
#     torch.mul((-torch.mul(S, theta_yy) + torch.log(1.0 + torch.exp(theta_yy))), torch.exp(S1))) / (opt.training_size * opt.batch_size)
#
# loss_y = logloss_yx + logloss_yy


if __name__ == '__main__':
    qB = torch.Tensor([[1, -1, 1, 1],
                       [-1, -1, -1, 1],
                       [1, 1, -1, 1],
                       [1, 1, 1, -1]])
    rB = torch.Tensor([[1, -1, 1, -1],
                       [-1, -1, 1, -1],
                       [-1, -1, 1, -1],
                       [1, 1, -1, -1],
                       [-1, 1, -1, -1],
                       [1, 1, -1, 1]])
    query_labels = torch.Tensor([[0, 1, 0, 0],
                            [1, 1, 0, 0],
                            [1, 0, 0, 1],
                            [0, 1, 0, 1]])
    retrieval_labels = torch.Tensor([[1, 0, 0, 1],
                                [1, 1, 0, 0],
                                [0, 1, 1, 0],
                                [0, 0, 1, 0],
                                [1, 0, 0, 0],
                                [0, 0, 1, 0]])

    # query_labels = torch.Tensor([[0, 1, 0, 0],
    #                              [1, 0, 0, 0],
    #                              [1, 0, 0, 0],
    #                              [0, 1, 0, 0]])
    # retrieval_labels = torch.Tensor([[1, 0, 0, 0],
    #                                  [0, 1, 0, 0],
    #                                  [0, 0, 1, 0],
    #                                  [0, 0, 1, 0],
    #                                  [1, 0, 0, 0],
    #                                  [0, 0, 1, 0]])

    trn_bainary = torch.Tensor(
        [
            [1, -1, 1, 1, -1],
            [-1, -1, -1, -1, -1],
            [1, 1, 1, 1, 1],
            [-1, 1, 1, -1, 1]
        ]
    )
    tst_binary = torch.Tensor(
        [
            [1, 1, 1, -1, -1],
            [1, 1, 1, 1, 1]
        ]
    )
    trn_label = torch.Tensor(
        [[0, 1], [0, 1], [1, 0], [1, 0]]
    )
    tst_label = torch.Tensor(
        [[1, 0], [0, 0]]
    )

    map = calc_map_k(qB, rB, query_labels, retrieval_labels)
    # map = calc_map(tst_binary, trn_bainary, tst_label, trn_label)
    print(map)
    # a = torch.randint(0, 256, (224, 224, 3))
    # image_from_numpy(a)

def load_pretrain_model(path):
    return sio.loadmat(path)