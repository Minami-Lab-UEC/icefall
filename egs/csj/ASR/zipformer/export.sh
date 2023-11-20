set -eou pipefail

python zipformer/export-onnx.py \
    --exp-dir zipformer/exp_disf \
    --epoch 40 \
    --avg 19 \
    --lang data/lang_char \
    --causal 0

# python zipformer/export-onnx.py \
#     --exp-dir zipformer/exp_disf \
#     --epoch 40 \
#     --avg 19 \
#     --lang data/lang_char \
#     --causal 0

# python zipformer/export-onnx.py \
#     --exp-dir zipformer/exp_disf_causal \
#     --epoch 40 \
#     --avg 25 \
#     --lang data/lang_char \
#     --causal 1 \
#     --chunk-size 27 \
#     --left-context-frames "-1"

# python zipformer/export-onnx.py \
#     --exp-dir zipformer/exp_disf_causal \
#     --epoch 40 \
#     --avg 25 \
#     --lang data/lang_char \
#     --causal 1 \
#     --chunk-size 27 \
#     --left-context-frames "-1"

python zipformer/export-onnx.py \
    --exp-dir zipformer/exp_disf_causal \
    --epoch 40 \
    --avg 25 \
    --lang data/lang_char \
    --causal 1 \
    --chunk-size 64 \
    --left-context-frames 256

python zipformer/export-onnx.py \
    --exp-dir zipformer/exp_disf_causal \
    --epoch 40 \
    --avg 25 \
    --lang data/lang_char \
    --causal 1 \
    --chunk-size 64 \
    --left-context-frames 256