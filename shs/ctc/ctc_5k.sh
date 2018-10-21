set -x

device=$1
rate=$2

CUDA_VISIBLE_DEVICES=$device python crnn_main.py --trainroot data/max90k_train_5k.lmdb --valroot data/max90k_val_5k.lmdb --cuda --experiment expr/5k/ctc_rms_b200_1e-3 --workers 4 --displayInterval 50 --batchSize 20 --lr 1e-3 --eval_all --valInterval 250 --saveInterval 5000 --niter 250 --adam --h_rate $rate
#2>&1 | tee logs/log_max_ent_$rate.log

