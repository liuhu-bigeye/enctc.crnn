#coding=utf-8
import os
import torch as T
from torch.autograd import Variable
import numpy as np
import pdb
import glog
import pickle

import copy
import time

cuda = True
if cuda:
    floatX = T.cuda.FloatTensor
    intX = T.cuda.IntTensor
    byteX = T.cuda.ByteTensor
    longX = T.cuda.LongTensor
else:
    floatX = T.FloatTensor
    intX = T.IntTensor
    byteX = T.ByteTensor
    longX = T.LongTensor

from m_ctc import m_eye, log_batch_dot, log_sum_exp

def ctc_ent_loss(pred, pred_len, token, token_len, blank=0):
    '''
    :param pred: (Time, batch, voca_size+1)
    :param pred_len: (batch)
    :param token: (batch, U)
    :param token_len: (batch)

    :out alpha: (Time, batch, 2U+1) ∑p(π|x)
    :out beta: (Time, batch, 2U+1)  ∑p(π|x)logp(π|x)
    :out H: -beta/alpha+log(alpha)
    '''
    Time, batch = pred.size(0), pred.size(1)
    U = token.size(1)
    eps = 0

    # token_with_blank
    token_with_blank = T.cat((T.zeros(batch, U, 1).type(longX), token[:, :, None]), dim=2).view(batch, -1)    # (batch, 2U)
    token_with_blank = T.cat((token_with_blank, T.zeros(batch, 1).type(longX)), dim=1)  # (batch, 2U+1)
    length = token_with_blank.size(1)

    pred = pred[T.arange(0, Time).type(longX)[:, None, None], T.arange(0, batch).type(longX)[None, :, None], token_with_blank[None, :]]  # (T, batch, 2U+1)

    # recurrence relation
    sec_diag = T.cat((T.zeros((batch, 2)).type(floatX), T.ne(token_with_blank[:, :-2], token_with_blank[:, 2:]).type(floatX)), dim=1) * T.ne(token_with_blank, blank).type(floatX)	# (batch, 2U+1)
    recurrence_relation = (m_eye(length) + m_eye(length, k=1)).repeat(batch, 1, 1) + m_eye(length, k=2).repeat(batch, 1, 1) * sec_diag[:, None, :]	# (batch, 2U+1, 2U+1)

    # alpha
    alpha_t = T.cat((pred[0, :, :2], T.zeros(batch, 2*U-1).type(floatX)), dim=1) # (batch, 2U+1)
    beta_t = T.cat((pred[0, :, :2] * T.log(pred[0, :, :2]), T.zeros(batch, 2*U-1).type(floatX)), dim=1) # (batch, 2U+1)

    alphas = alpha_t[None] # (1, batch, 2U+1)
    betas = beta_t[None] # (1, batch, 2U+1)

    # dynamic programming
    # (T, batch, 2U+1)
    for t in T.arange(1, Time).type(longX):
        alpha_t = T.bmm(alpha_t[:, None], recurrence_relation)[:, 0] * pred[t]
        beta_t = T.bmm(beta_t[:, None], recurrence_relation)[:, 0] * pred[t] + T.log(pred[t]) * alpha_t

        alphas = T.cat((alphas, alpha_t[None]), dim=0)
        betas = T.cat((betas, beta_t[None]), dim=0)

    def collect_label(probability):
        labels_2 = probability[pred_len-1, T.arange(batch).type(longX), 2*token_len-1]
        labels_1 = probability[pred_len-1, T.arange(batch).type(longX), 2*token_len]
        labels_prob = labels_2 + labels_1
        return labels_prob

    alpha = collect_label(alphas)
    beta = collect_label(betas)

    H = -beta/alpha + T.log(alpha+eps)
    costs = -T.log(alpha+eps)
    return H, costs

def ctc_ent_loss_log(pred, pred_len, token, token_len, blank=0):
    '''
    :param pred: (Time, batch, voca_size+1)
    :param pred_len: (batch)
    :param token: (batch, U)
    :param token_len: (batch)

    :out alpha: (Time, batch, 2U+1) ∑p(π|x)
    :out beta: (Time, batch, 2U+1)  ∑p(π|x)logp(π|x)
    :out H: -beta/alpha+log(alpha)
    '''
    Time, batch = pred.size(0), pred.size(1)
    U = token.size(1)
    eps_nan = -1e8
    eps = 1e-8

    # token_with_blank
    token_with_blank = T.cat((T.zeros(batch, U, 1).type(longX), token[:, :, None]), dim=2).view(batch, -1)    # (batch, 2U)
    token_with_blank = T.cat((token_with_blank, T.zeros(batch, 1).type(longX)), dim=1)  # (batch, 2U+1)
    length = token_with_blank.size(1)

    pred = pred[T.arange(0, Time).type(longX)[:, None, None], T.arange(0, batch).type(longX)[None, :, None], token_with_blank[None, :]]  # (T, batch, 2U+1)

    # recurrence relation
    sec_diag = T.cat((T.zeros((batch, 2)).type(floatX), T.ne(token_with_blank[:, :-2], token_with_blank[:, 2:]).type(floatX)), dim=1) * T.ne(token_with_blank, blank).type(floatX)	# (batch, 2U+1)
    recurrence_relation = (m_eye(length) + m_eye(length, k=1)).repeat(batch, 1, 1) + m_eye(length, k=2).repeat(batch, 1, 1) * sec_diag[:, None, :]	# (batch, 2U+1, 2U+1)
    recurrence_relation = eps_nan * (T.ones_like(recurrence_relation) - recurrence_relation)

    # alpha
    alpha_t = T.cat((pred[0, :, :2], T.ones(batch, 2*U-1).type(floatX)*eps_nan), dim=1) # (batch, 2U+1)
    beta_t = T.cat((pred[0, :, :2] + T.log(-pred[0, :, :2]+eps),
                    T.ones(batch, 2*U-1).type(floatX)*eps_nan), dim=1) # (batch, 2U+1)

    alphas = alpha_t[None] # (1, batch, 2U+1)
    betas = beta_t[None] # (1, batch, 2U+1)

    # dynamic programming
    # (T, batch, 2U+1)
    for t in T.arange(1, Time).type(longX):
        alpha_t = log_batch_dot(alpha_t, recurrence_relation) + pred[t]
        beta_t = log_sum_exp(log_batch_dot(beta_t, recurrence_relation) + pred[t], T.log(-pred[t]+eps) + alpha_t)

        alphas = T.cat((alphas, alpha_t[None]), dim=0)
        betas = T.cat((betas, beta_t[None]), dim=0)

    def collect_label(probability):
        labels_2 = probability[pred_len-1, T.arange(batch).type(longX), 2*token_len-1]
        labels_1 = probability[pred_len-1, T.arange(batch).type(longX), 2*token_len]
        labels_prob = log_sum_exp(labels_2, labels_1)
        return labels_prob

    alpha = collect_label(alphas)
    beta = collect_label(betas)

    H = T.exp(beta-alpha) + alpha
    costs = -alpha
    return H, costs

def ctc_ent_cost(out, targets, sizes, target_sizes, use_softmax=True, use_log=True, sumed=True):
#    A batched version for uni_alpha_cost
#    param out: (Time, batch, voca_size+1)
#    param targets: targets without splited
#    param sizes: size for out (N)
#    param target_sizes: size for targets (N)

    Time = out.size(0)
    if use_log:
        if use_softmax:
            pred = T.nn.functional.log_softmax(out, dim=-1)
        else:
            pred = out
        loss_func = ctc_ent_loss_log
    else:
        if use_softmax:
            pred = T.nn.functional.softmax(out, dim=-1)
        else:
            pred = out
        loss_func = ctc_ent_loss

    offset = 0
    batch = target_sizes.size(0)
    target_max = target_sizes.max().item()
    target = T.zeros(batch, target_max).type(longX)

    for index, (target_size, size) in enumerate(zip(target_sizes, sizes)):
        target[index, :target_size.item()] = targets[offset: offset+target_size.item()].data
        offset += target_size.item()

    if not cuda:
        H, costs = loss_func(pred.cpu(), sizes.data.type(longX), target, target_sizes.data.type(longX))
    else:
        H, costs = loss_func(pred, sizes.data.type(longX), target, target_sizes.data.type(longX))

    if sumed:
        return H.sum(), costs.sum()
    else:
        return H, costs

def test_seg_ctc(use_mine=True, use_log=False):
    size = 43
    voca_size = 37
    n = 20

    np.random.seed(1234)
    pred_len_np = np.ones([n])*size
    pred_np = np.random.random([size, n, voca_size+1])
    pred_np = np.log(pred_np)

    token_len_np = np.random.randint(low=2, high=10, size=n)
    token_np = np.random.randint(voca_size, size=token_len_np.sum())+1

    pred = Variable(floatX(pred_np), requires_grad=True)
    token = Variable(T.IntTensor(token_np))
    sizes = Variable(T.IntTensor(pred_len_np))
    target_sizes = Variable(T.IntTensor(token_len_np))

    for i in range(50):
        if use_mine:
            H, cost = ctc_ent_cost(pred, token, sizes, target_sizes, use_log=use_log)
            glog.info('%d, cost: %s, entropy: %s'% (i, cost.data.item(), H.data.item()))
#            cost = 0.9*cost - 0.1*H
        else:
            from warpctc_pytorch import CTCLoss
            criterion = CTCLoss().cuda()
            cost = criterion(pred, token, sizes, target_sizes)
            glog.info('%d, cost: %s'% (i, cost.data.item()))

        optimizer = T.optim.Adam([pred], lr=3e-1)#, nesterov=True)
        optimizer.zero_grad()
        (cost).backward()
        optimizer.step()

if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]="3"
    print('_________')
    test_seg_ctc(use_mine=True, use_log=True)
