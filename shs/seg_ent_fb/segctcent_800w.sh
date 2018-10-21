set -x

for i in 3,0.2,1.5;do
    IFS=',' read gpu_id rate uni_rate<<< "${i}"
    echo $gpu_id:$rate:$uni_rate
    dat=`date +'%Y-%m-%d_%H-%M-%S'`
    CUDA_VISIBLE_DEVICES=$gpu_id nohup python crnn_main_seg.py --trainroot data/max90k_train_val.lmdb --valroot data/max90k_test.lmdb --cuda \
        --experiment expr/segctc_ent/800w/float/segent_adam_$rate:$uni_rate:$dat \
        --workers 4 --displayInterval 50 --batchSize 256 --lr 1e-3 \
        --valInterval 500 --saveInterval 500 --niter 50 \
        --adam --h_rate $rate --uni_rate $uni_rate 2>&1 > logs/segctc_ent/800w/log_segent800w_adam_$rate:$uni_rate:$dat.txt &
    sleep 10
done

# need sleep, because init may cause huge gpu usage
