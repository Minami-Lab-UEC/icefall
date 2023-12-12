set -eou pipefail

# python zipformer/export.py \
#     --exp-dir zipformer/exp_f_causal \
#     --epoch 40 \
#     --avg 19 \
#     --lang data/lang_char \
#     --causal 1 \
#     --jit 1 \
#     --chunk-size 64 \
#     --left-context-frames 256

python zipformer/export.py \
    --exp-dir zipformer/exp_num_causal \
    --epoch 40 \
    --avg 15 \
    --lang data/lang_char \
    --causal 1 \
    --jit 1 \
    --chunk-size 64 \
    --left-context-frames 256
