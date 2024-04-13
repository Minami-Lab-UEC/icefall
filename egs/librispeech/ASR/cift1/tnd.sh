#!/bin/bash

set -eou pipefail

for file in cift1/configs/*.txt ; do
    setup="${file%.txt}"
    setup="${setup#cift1/configs/}"
    args=$(sed '/^##/,$d' $file)
    exp_dir=exp1_"$setup"_1

    echo "$args" | xargs -a - python cift1/train.py \
        --world-size 4 \
        --exp-dir cift1/$exp_dir \
        --num-epochs 30 \
        --start-epoch 1 \
        --use-fp16 1 \
        --max-duration 850 \
        --lang data/lang_bpe_500 \
        --manifest-dir data/fbank \
        --musan-dir /mnt/host/corpus/musan/fbank \
        --causal 1 \
        --full-libri 2 \
        --telegram-cred misc.ini || { python notify_tg.py "$exp_dir : Something wrong during training." ; exit 1; }

    python notify_tg.py "cift1/$exp_dir Training done."

    mv $file cift1/$exp_dir/

    ./cift1/decode_main.sh cift1/$exp_dir "7 8 9"

done

