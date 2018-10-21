set -x

for i in 2,0.2,1.5;do
    IFS=',' read gpu_id rate uni_rate<<< "${i}"
    echo $gpu_id:$rate:$uni_rate
    dat=`date +'%Y-%m-%d_%H-%M-%S'`
    CUDA_VISIBLE_DEVICES=$gpu_id python crnn_main_seg.py --trainroot data/max90k_train_5k.lmdb --valroot data/max90k_val_5k.lmdb --cuda \
        --experiment expr/segctc_ent/5k/segent_1e-3_$rate:$uni_rate:$dat \
        --workers 4 --displayInterval 50 --batchSize 20 --lr 1e-3 \
        --valInterval 250 --saveInterval 1250 --niter 250 \
        --eval_all \
        --h_rate $rate --uni_rate $uni_rate 2>&1 | tee logs/segctc_ent/log_segent5k_1e-3_$rate:$uni_rate:$dat.txt
done

# need sleep, because init may cause huge gpu usage
