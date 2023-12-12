set -eou pipefail

# python zipformer/train.py \
#     --telegram-cred misc.ini \
#     --world-size 8 \
#     --num-epochs 40 \
#     --exp-dir zipformer/exp_disf \
#     --causal 0 \
#     --lang data/lang_char \
#     --musan-dir /mnt/host/corpus/musan/musan/fbank \
#     --manifest-dir /mnt/host/corpus/csj/fbank \
#     --max-duration 600 \
#     --transcript-mode disfluent \
#     --causal 0 \
#     --use-fp16 1 || { python notify_tg.py "zipformer/exp_disf : Something wrong during training." ; exit 1; }

# python notify_tg.py "zipformer/exp_disf Training done."



# python zipformer/train.py \
#     --telegram-cred misc.ini \
#     --world-size 8 \
#     --num-epochs 40 \
#     --exp-dir zipformer/exp_disf_causal \
#     --causal 0 \
#     --lang data/lang_char \
#     --musan-dir /mnt/host/corpus/musan/musan/fbank \
#     --manifest-dir /mnt/host/corpus/csj/fbank \
#     --max-duration 600 \
#     --transcript-mode disfluent \
#     --causal 1 \
#     --use-fp16 1 || { python notify_tg.py "zipformer/exp_disf : Something wrong during training." ; exit 1; }

# python notify_tg.py "zipformer/exp_disf Training done."

exp_dir=zipformer/exp_f_causal
setup=$(find "$exp_dir" -maxdepth 1 -type f -name "*.txt")
args=$(sed '/^##/,$d' $setup)
echo "$args" | xargs -a - python zipformer/train.py \
    --telegram-cred misc.ini \
    --world-size 8 \
    --num-epochs 40 \
    --exp-dir $exp_dir \
    --lang data/lang_char \
    --musan-dir /mnt/host/corpus/musan/musan/fbank \
    --manifest-dir /mnt/host/corpus/csj/fbank \
    --max-duration 600 \
    --use-fp16 1 || { python notify_tg.py "$exp_dir : Something wrong during training." ; exit 1; }

python notify_tg.py "$exp_dir Training done."

./zipformer/decode_main.sh $exp_dir


exp_dir=zipformer/exp_f
setup=$(find "$exp_dir" -maxdepth 1 -type f -name "*.txt")
args=$(sed '/^##/,$d' $setup)
echo "$args" | xargs -a - python zipformer/train.py \
    --telegram-cred misc.ini \
    --world-size 8 \
    --num-epochs 40 \
    --exp-dir $exp_dir \
    --lang data/lang_char \
    --musan-dir /mnt/host/corpus/musan/musan/fbank \
    --manifest-dir /mnt/host/corpus/csj/fbank \
    --max-duration 600 \
    --use-fp16 1 || { python notify_tg.py "$exp_dir : Something wrong during training." ; exit 1; }

python notify_tg.py "$exp_dir Training done."

./zipformer/decode_main.sh $exp_dir



exp_dir=zipformer/exp_num_causal
setup=$(find "$exp_dir" -maxdepth 1 -type f -name "*.txt")
args=$(sed '/^##/,$d' $setup)
echo "$args" | xargs -a - python zipformer/train.py \
    --telegram-cred misc.ini \
    --world-size 8 \
    --num-epochs 40 \
    --exp-dir $exp_dir \
    --lang data/lang_char \
    --musan-dir /mnt/host/corpus/musan/musan/fbank \
    --manifest-dir /mnt/host/corpus/csj/fbank \
    --max-duration 600 \
    --use-fp16 1 || { python notify_tg.py "$exp_dir : Something wrong during training." ; exit 1; }

python notify_tg.py "$exp_dir Training done."

./zipformer/decode_main.sh $exp_dir


exp_dir=zipformer/exp_num
setup=$(find "$exp_dir" -maxdepth 1 -type f -name "*.txt")
args=$(sed '/^##/,$d' $setup)
echo "$args" | xargs -a - python zipformer/train.py \
    --telegram-cred misc.ini \
    --world-size 8 \
    --num-epochs 40 \
    --exp-dir $exp_dir \
    --lang data/lang_char \
    --musan-dir /mnt/host/corpus/musan/musan/fbank \
    --manifest-dir /mnt/host/corpus/csj/fbank \
    --max-duration 600 \
    --use-fp16 1 || { python notify_tg.py "$exp_dir : Something wrong during training." ; exit 1; }

python notify_tg.py "$exp_dir Training done."

./zipformer/decode_main.sh $exp_dir