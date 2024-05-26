set -eou pipefail

epoch=40
for libri in 3 ; do
    exp_dir=zipformer/exp_causal_libri$i
    python zipformer/train.py \
        --world-size 4 \
        --exp-dir $exp_dir \
        --num-epochs $epoch \
        --start-epoch 1 \
        --use-fp16 1 \
        --max-duration 850 \
        --musan-dir /mnt/host/corpus/musan/fbank \
        --causal 1 \
        --full-libri $libri || { python notify_tg.py "zipformer : Something wrong during training." ; exit 1; }

    python notify_tg.py "$exp_dir Training done."

    ./zipformer/decode_main.sh $exp_dir $epoch
done
