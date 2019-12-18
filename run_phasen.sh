#!/bin/bash 

lr=0.001

win_len=400
win_inc=100
fft_len=512

sample_rate=16k
win_type=hanning

batch_size=4
max_epoch=50
rnn_nums=300

tt_list='data/test_wsj0_0.lst'
cv_list='data/cv_wsj0_-5~20.lst'
tr_list='data/tr_wsj0_-5~20.lst'
tt_list='data/t'

retrain=1

num_gpu=2
batch_size=$[num_gpu*batch_size]

save_name=Phasen_${lr}_${sample_rate}_${win_len}_${win_inc}

exp_dir=exp/${save_name}

if [ ! -d ${exp_dir} ] ; then
    mkdir -p ${exp_dir}
fi

stage=2

if [ $stage -le 1 ] ; then
    CUDA_VISIBLE_DEVICES='0,1' nohup python -u ./steps/run_phasen.py \
    --decode=0 \
    --fft-len=${fft_len} \
    --window-len=${win_len} \
    --window-inc=${win_inc} \
    --exp-dir=${exp_dir} \
    --tr-list=${tr_list} \
    --cv-list=${cv_list} \
    --tt-list=${tt_list} \
    --retrain=${retrain} \
    --rnn-nums=${rnn_nums} \
    --learn-rate=${lr} \
    --max-epoch=${max_epoch} \
    --batch-size=${batch_size} \
    --sample-rate=${sample_rate} \
    --window-type=${win_type} > ${exp_dir}/train.log &
    exit 0
fi

if [ $stage -le 2 ] ; then 
    CUDA_VISIBLE_DEVICES='' python -u ./steps/run_phasen.py \
    --decode=1 \
    --fft-len=${fft_len} \
    --window-len=${win_len} \
    --window-inc=${win_inc} \
    --exp-dir=${exp_dir} \
    --tr-list=${tr_list} \
    --cv-list=${cv_list} \
    --tt-list=${tt_list} \
    --retrain=${retrain} \
    --rnn-nums=${rnn_nums} \
    --learn-rate=${lr} \
    --max-epoch=${max_epoch} \
    --batch-size=${batch_size} \
    --sample-rate=${sample_rate} \
    --window-type=${win_type}
    exit 0
fi

if [ $stage -le 3 ] ; then

for snr in -5 0 5 10 15 20 ; do 
    dataset_name=wsj0
    tgt=Phasen_${target_mode}_${dataset_name}_${snr}db.csv
    clean_wav_path="data/wavs/test_${dataset_name}_clean_${snr}/"
    noisy_wav_path="data/wavs/test_${dataset_name}_noisy_${snr}/"
    tgt=${exp_dir}/${tgt}
    enh_wav_path=${exp_dir}/test_${dataset_name}_noisy_${snr}/
    tt_list="data/test_${dataset_name}_${snr}.lst" 
    CUDA_VISIBLE_DEVICES='1' python -u ./steps/run_phasen.py \
    --decode=1 \
    --fft-len=${fft_len} \
    --window-len=${win_len} \
    --window-inc=${win_inc} \
    --exp-dir=${exp_dir} \
    --tt-list=${tt_list} \
    --rnn-nums=${rnn_nums} \
    --retrain=${retrain} \
    --learn-rate=${lr} \
    --max-epoch=${max_epoch} \
    --batch-size=${batch_size} \
    --sample-rate=${sample_rate} \
    --window-type=${win_type} || exit 1
    
    mv ${exp_dir}/rec_wav ${enh_wav_path}
    
    ls $noisy_wav_path > /tmp/t
    python ./tools/eval_objective.py --wav_list=/tmp/t --result_list=${tgt} --pathe=${enh_wav_path}\
    --pathc=${clean_wav_path} --pathn=${noisy_wav_path} ||exit 1
done

python ./tools/get_results.py ${exp_dir}
fi
