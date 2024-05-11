set -eou pipefail

exp_dir=zipformer/exp_f_causal
# setup=$(find "$exp_dir" -maxdepth 1 -type f -name "*.txt")
# args=$(sed '/^##/,$d' $setup)
python zipformer/train.py \
    --world-size 4 \
    --num-epochs 40 \
    --exp-dir $exp_dir \
    --lang data/lang_char \
    --musan-dir /mnt/host/corpus/musan/fbank \
    --manifest-dir data/fbank \
    --max-duration 650 \
    --causal 1 \
    --transcript-mode fluent \
    --use-fp16 1 || { python notify_tg.py "$exp_dir : Something wrong during training." ; exit 1; }

python notify_tg.py "$exp_dir Training done."

./zipformer/decode_main.sh $exp_dir


