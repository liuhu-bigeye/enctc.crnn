Project for Connectionist Temporal Classification with Maximum Entropy Regularization(to be released after NIPS2018).

# Usage

1. Train

zsh shs/seg_ent_fb/seg_5k.sh

2. Test

CUDA_VISIBLE_DEVICES=0 python test.py --crnn_path model_dir --valroot data/svt1/testset.lmdb

