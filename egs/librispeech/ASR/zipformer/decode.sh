set -eou pipefail

# non-streaming
# for m in greedy_search modified_beam_search fast_beam_search; do
#   ./zipformer/decode.py \
#     --epoch 40 \
#     --avg 16 \
#     --use-averaged-model 1 \
#     --exp-dir ./zipformer/exp \
#     --max-duration 600 \
#     --decoding-method $m
# done


# streaming

# for m in greedy_search modified_beam_search fast_beam_search; do
#   ./zipformer/decode.py \
#     --epoch 30 \
#     --avg 8 \
#     --use-averaged-model 1 \
#     --exp-dir ./zipformer/exp_causal \
#     --max-duration 600 \
#     --causal 1 \
#     --chunk-size 16 \
#     --left-context-frames 128 \
#     --decoding-method $m
# done

exp_dir=zipformer/exp_causal
args=$(sed '/^##/,$d' $exp_dir/streaming.txt)

echo $args | xargs -a - python zipformer/decode.py \
  --epoch 30 \
  --avg 8 \
  --use-averaged-model 1 \
  --exp-dir $exp_dir \
  --res-dir $exp_dir/todelete \
  --max-duration 600 \
  --chunk-size 32 \
  --left-context-frames 256 \
  --decoding-method modified_beam_search \
  --gpu 0
