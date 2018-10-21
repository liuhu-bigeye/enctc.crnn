set -x

for i in 3,0 2,0.1 3,0.2;do
#for i in 2,0.1 2,0.2 3,0.5 3,1;do
    IFS=',' read gpu_id rate <<< "${i}"
    echo $gpu_id:$rate
    dat=`date +'%Y-%m-%d_%H-%M-%S'`
    CUDA_VISIBLE_DEVICES=$gpu_id nohup python crnn_main.py --trainroot data/max90k_train_val.lmdb --valroot data/max90k_test.lmdb --cuda \
        --experiment expr/ctc_env/800w/ctc_adam_$rate:$dat --workers 4 --displayInterval 50 --batchSize 256 --lr 1e-3 --valInterval 500 \
        --saveInterval 500 --niter 50 --h_rate $rate 2>&1 > logs/ctc_env/log_ctc800w_rms_$rate:$dat.txt &
    sleep 10
done

# need sleep, because init may cause huge gpu usage
