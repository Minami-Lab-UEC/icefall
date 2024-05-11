#!/bin/bash

set -eou pipefail

libri=3
epoch=40

for file in cift1/configs/*.txt ; do
    setup="${file%.txt}"
    setup="${setup#cift1/configs/}"
    args=$(sed '/^##/,$d' $file)
    exp_dir=exp4_"$setup"_full

    echo "$args" | xargs -a - python cift1/train_frame.py \
        --world-size 4 \
        --exp-dir cift1/$exp_dir \
        --num-epochs $epoch \
        --start-epoch 1 \
        --use-fp16 1 \
        --max-duration 850 \
        --lang data/lang_bpe_500 \
        --manifest-dir data/fbank \
        --musan-dir /mnt/host/corpus/musan/fbank \
        --causal 1 \
        --full-libri $libri \
        --telegram-cred misc.ini || { python notify_tg.py "$exp_dir : Something wrong during training." ; exit 1; }

    python notify_tg.py "cift1/$exp_dir Training done."

    mv $file cift1/$exp_dir/

    ./cift1/decode_main_libri"$libri".sh cift1/$exp_dir "7 8 9" $epoch

done

./cift1/sweep_chunksize.sh