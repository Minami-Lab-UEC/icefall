set -eou pipefail

python zipformer/train.py \
    --telegram-cred misc.ini \
    --world-size 1 \
    --num-epochs 2 \
    --exp-dir zipformer/exp \
    --causal 0 \
    --lang data/lang_char \
    