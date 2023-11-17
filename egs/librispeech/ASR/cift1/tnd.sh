#!/bin/bash

# set -eou pipefail

# for file in cift1/configs/*.txt ; do
#     setup="${file%.txt}"
#     setup="${setup#cift1/configs/}"
#     args=$(sed '/^##/,$d' $file)
#     exp_dir=exp0_"$setup"_1

#     echo "$args" | xargs -a - python cift1/train.py \
#         --world-size 1 \
#         --exp-dir cift1/$exp_dir \
#         --num-epochs 2 \
#         --start-epoch 1 \
#         --use-fp16 1 \
#         --max-duration 240 \
#         --lang data/lang_bpe_500 \
#         --manifest-dir data/fbank \
#         --full-libri 0 \
#         --prune-range 16 \
#         --pad-feature 30 \
#         --musan-dir /mnt/host/corpus/musan/musan/fbank \
#         --telegram-cred misc.ini || { python notify_tg.py "$exp_dir : Something wrong during training." ; exit 1; }

#     python notify_tg.py "cift1/$exp_dir Training done."
    
#     mv $file cift1/$exp_dir/

#     ./cift1/decode_main.sh cift1/$exp_dir 40

# done
exp_dir=exp0_meanabs
python -m pdb cift1/train.py \
    --world-size 1 \
    --exp-dir cift1/$exp_dir \
    --num-epochs 2 \
    --start-epoch 1 \
    --use-fp16 1 \
    --max-duration 240 \
    --lang data/lang_bpe_500 \
    --manifest-dir data/fbank \
    --full-libri 0 \
    --prune-range 16 \
    --pad-feature 30 \
    --musan-dir /mnt/host/corpus/musan/musan/fbank \
    --context-size 4 \
    --phi-arch vanilla \
    --phi-type "att;8,0" \
    --phi-norm-type biasnorm \
    --alpha-actv abs \
    --omega-type Mean 