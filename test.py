import glob
import torch
import argparse
from torch.autograd import Variable
import utils
import dataset
from PIL import Image

import models.crnn as crnn
from warpctc_pytorch import CTCLoss
import dataset

import numpy as np
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--valroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--crnn_path', required=True, help="path to crnn (to continue training)")
parser.add_argument('--batchSize', type=int, default=20, help='input batch size')
parser.add_argument('--n_test_disp', type=int, default=10, help='Number of samples to display when test')

opt = parser.parse_args()

alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'
model = crnn.CRNN(32, 1, 37, 256)

converter = utils.strLabelConverter(alphabet)
#beam_converter = BeamCTCDecoder('_'+alphabet, lm_path=None, alpha=0.8, beta=1,
#        cutoff_top_n=40, cutoff_prob=1.0, beam_width=10, num_processes=1)
criterion = CTCLoss()

test_dataset = dataset.lmdbDataset(
    root=opt.valroot, transform=dataset.resizeNormalize((100, 32)))

image = torch.FloatTensor(opt.batchSize, 3, 100, 32)
text = torch.IntTensor(opt.batchSize * 5)
length = torch.IntTensor(opt.batchSize)

image = image.cuda()
criterion = criterion.cuda()

image = Variable(image)
text = Variable(text)
length = Variable(length)

def val(net, dataset, criterion, model_path, max_iter=np.inf):
    for p in net.parameters():
        p.requires_grad = False

    net.eval()
    data_loader = torch.utils.data.DataLoader(
        dataset, shuffle=True, batch_size=opt.batchSize, num_workers=int(opt.workers))
    val_iter = iter(data_loader)

    i = 0
    n_correct = 0
    n_correct_greed = 0
    # loss averager
    loss_avg = utils.averager()
    loss_avg.reset()

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

        preds = net(image)
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
        cost = criterion(preds, text, preds_size, length) / batch_size
        loss_avg.add(cost)

        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds_greed = converter.decode(preds.data, preds_size.data, raw=False)
        for idx, (pred_greed, target) in enumerate(zip(sim_preds_greed, cpu_texts)):
            if pred_greed == target.lower():
                n_correct_greed += 1

    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:opt.n_test_disp]
    accuracy_greed = n_correct_greed / float(max_iter * opt.batchSize)

    print('test loss: %f, accuray_greed: %f, model: %s' % (loss_avg.val(), accuracy_greed, model_path))

model_path = sorted(glob.glob(opt.crnn_path+'/netCRNN_*00.pth'), key=lambda x:int(x.split('_')[-1][:-4]) + 3e5*int(x.split('_')[-2]), reverse=True)[0]

while True:
    try:
        model.load_state_dict(torch.load(model_path))
        break
    except:
        if torch.cuda.is_available():
            model.cuda()
            model = torch.nn.DataParallel(model, device_ids=range(1))

#print('loading pretrained model from %s' % model_path)
val(model, test_dataset, criterion, '_'.join(model_path.split('_')[-2:]))
