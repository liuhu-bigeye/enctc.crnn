from __future__ import print_function
import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import sys
sys.path.insert(0, 'pytorch_ctc')
from ctc_ent import ctc_ent_cost
import os
import utils
import dataset

import models.crnn as crnn
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--trainroot', required=True, help='path to dataset')
parser.add_argument('--valroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
parser.add_argument('--imgW', type=int, default=100, help='the width of the input image to network')
parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state')
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate for Critic, default=0.00005')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--crnn', default='', help="path to crnn (to continue training)")
parser.add_argument('--alphabet', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz')
parser.add_argument('--experiment', default=None, help='Where to store samples and models')
parser.add_argument('--displayInterval', type=int, default=500, help='Interval to be displayed')
parser.add_argument('--n_test_disp', type=int, default=10, help='Number of samples to display when test')
parser.add_argument('--valInterval', type=int, default=500, help='Interval to be displayed')
parser.add_argument('--saveInterval', type=int, default=500, help='Interval to be displayed')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('--adadelta', action='store_true', help='Whether to use adadelta (default is rmsprop)')
parser.add_argument('--keep_ratio', action='store_true', help='whether to keep ratio for image resize')
parser.add_argument('--random_sample', action='store_true', help='whether to sample the dataset with random sampler')
parser.add_argument('--eval_all', action='store_true', help='whether evaluate on the whole dataset')
parser.add_argument('--max_norm', default=400, type=int, help='Norm cutoff to prevent explosion of gradients')
parser.add_argument('--h_rate', type=float, default=0.1, help='h_rate for max ent reg. default=0.1')
opt = parser.parse_args()
print(opt)

if opt.experiment is None:
    opt.experiment = 'expr'

os.system('mkdir {0}'.format(opt.experiment))

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

train_dataset = dataset.lmdbDataset(root=opt.trainroot)
assert train_dataset
if False:#opt.random_sample: use shuffle
    sampler = dataset.randomSequentialSampler(train_dataset, opt.batchSize)
else:
    sampler = None

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=opt.batchSize,
    shuffle=True, sampler=sampler,
    num_workers=int(opt.workers),
    collate_fn=dataset.alignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio=opt.keep_ratio))
test_dataset = dataset.lmdbDataset(
    root=opt.valroot, transform=dataset.resizeNormalize((100, 32)))

nclass = len(opt.alphabet) + 1
nc = 1

converter = utils.strLabelConverter(opt.alphabet)

# custom weights initialization called on crnn
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


crnn = crnn.CRNN(opt.imgH, nc, nclass, opt.nh)
crnn.apply(weights_init)
if opt.crnn != '':
    print('loading pretrained model from %s' % opt.crnn)
    while True:
        try:
            crnn.load_state_dict(torch.load(opt.crnn))
            break
        except:
            if opt.cuda:
                crnn.cuda()
                crnn = torch.nn.DataParallel(crnn, device_ids=range(opt.ngpu))

print(crnn)

image = torch.FloatTensor(opt.batchSize, 3, opt.imgH, opt.imgH)
text = torch.IntTensor(opt.batchSize * 5)
length = torch.IntTensor(opt.batchSize)

if opt.cuda:
    if opt.crnn == '':
        crnn.cuda()
        crnn = torch.nn.DataParallel(crnn, device_ids=range(opt.ngpu))
    image = image.cuda()

image = Variable(image)
text = Variable(text)
length = Variable(length)

# loss averager
h_avg = utils.averager()
loss_avg = utils.averager()

# setup optimizer
if opt.adam:
    optimizer = optim.Adam(crnn.parameters(), lr=opt.lr,
                           betas=(opt.beta1, 0.999))
elif opt.adadelta:
    #do not use lr, according to orig paper...
    #always meet inf, set lr to 1e-3
    optimizer = optim.Adadelta(crnn.parameters(), lr=opt.lr)
else:
    optimizer = optim.RMSprop(crnn.parameters(), lr=opt.lr)


def val(net, dataset, max_iter=100):
    print('Start val')

    for p in crnn.parameters():
        p.requires_grad = False

    net.eval()
    data_loader = torch.utils.data.DataLoader(
        dataset, shuffle=False, batch_size=opt.batchSize, num_workers=int(opt.workers))
    val_iter = iter(data_loader)

    i = 0
    n_correct_greed = 0
    h_avg = utils.averager()
    loss_avg = utils.averager()

    if opt.eval_all:
        max_iter = len(data_loader)
    else:
        max_iter = min(max_iter, len(data_loader))

    for i in range(max_iter):
        data = val_iter.next()
        i += 1
        cpu_images, cpu_texts = data
        batch_size = cpu_images.size(0)
        utils.loadData(image, cpu_images)
        t, l = converter.encode(cpu_texts)
        utils.loadData(text, t)
        utils.loadData(length, l)

        preds = crnn(image)
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
        H, cost = ctc_ent_cost(preds, text, preds_size, length, sumed=False)
        h_avg.add(H.sum()/batch_size)
        loss_avg.add(cost.sum()/batch_size)

        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        preds_greed = converter.decode(preds.data, preds_size.data, raw=False)
        for idx, (pred_greed, target) in enumerate(zip(preds_greed, cpu_texts)):
            if pred_greed == target.lower():
                n_correct_greed += 1

    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:opt.n_test_disp]
    for raw_pred, pred, gt in zip(raw_preds, preds_greed, cpu_texts):
        print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    accuracy_greed = n_correct_greed / float(max_iter * opt.batchSize)
    print('Test H: %f, loss: %f, accuray_greed: %f' % (h_avg.val(), loss_avg.val(), accuracy_greed))


def trainBatch(net, optimizer):
    data = train_iter.next()
    cpu_images, cpu_texts = data
    batch_size = cpu_images.size(0)
    utils.loadData(image, cpu_images)
    t, l = converter.encode(cpu_texts)
    utils.loadData(text, t)
    utils.loadData(length, l)

    preds = crnn(image)
    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
    # length (batch)
    H, cost = ctc_ent_cost(preds, text, preds_size, length)
    cost_sum = cost.data.sum()
    inf = float("inf")
    if cost_sum == inf or cost_sum == -inf or cost_sum<=-1e5 or torch.isnan(cost) or torch.isnan(H):
        print("Warning: received an inf loss, setting loss value to 0")
        return torch.zeros(H.size()), torch.zeros(cost.size())

    crnn.zero_grad()
    (-opt.h_rate*H + (1-opt.h_rate)*cost).backward()

    torch.nn.utils.clip_grad_norm(crnn.parameters(), opt.max_norm)
    optimizer.step()
    return H / batch_size, cost / batch_size

## valid at the beginning

for epoch in range(opt.niter):
    train_iter = iter(train_loader)
    i = 0

    while i < len(train_loader):
        for p in crnn.parameters():
            p.requires_grad = True

        crnn.train()
        H, cost = trainBatch(crnn, optimizer)
        h_avg.add(H)
        loss_avg.add(cost)
        i += 1

        if i % opt.displayInterval == 0:
            print('[%d/%d][%d/%d] H: %f, Loss: %f' %
                  (epoch, opt.niter, i, len(train_loader), h_avg.val(), loss_avg.val()))
            h_avg.reset()
            loss_avg.reset()

        if i % opt.valInterval == 0:
            val(crnn, test_dataset)

        # save checkpoint
        if i % opt.saveInterval == 0 or (opt.saveInterval >= len(train_loader) and i == len(train_loader)-1):
            torch.save(
                crnn.state_dict(), '{0}/netCRNN_{1}_{2}.pth'.format(opt.experiment, epoch, i))
