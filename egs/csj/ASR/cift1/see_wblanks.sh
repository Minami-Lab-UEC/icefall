set -eou pipefail

exp_dir=$1
avg=$2
maxsym=$3
epoch=30
setup=$(find "$exp_dir" -maxdepth 1 -type f -name "*.txt")
args=$(sed '/^##/,$d' $setup)
echo "$args" | xargs -a - python cift1/see_wblanks.py \
    --exp-dir $exp_dir \
    --epoch $epoch \
    --avg $avg \
    --max-duration 1500 \
    --lang data/lang_char \
    --manifest-dir data/fbank \
    --max-sym-per-frame $maxsym \
    --res-dir $exp_dir/wblanks_avg"$avg"_sym"$maxsym" \
    --chunk 32 \
    --gpu 0 \
    --left-context-frames 64 \
    --causal 1 \
    --transcript-mode fluent

python cift1/sym_per_awe.py --wblank-dir $exp_dir/wblanks_avg"$avg"_sym"$maxsym"

# python cift1/word_seg_analysis.py --wblank-dir $exp_dir/wblanks_avg"$avg"_sym"$maxsym"

python cift1/unique_awes.py --wblank-dir $exp_dir/wblanks_avg"$avg"_sym"$maxsym"

# python -m pdb cift1/see_wblanks.py \
#     --exp-dir cift1/exp2_3gram_1 \
#     --epoch 40 \
#     --avg 12 \
#     --max-duration 150 \
#     --lang data/lang_char \
#     --manifest-dir data/fbank \
#     --max-sym-per-frame 8 \
#     --res-dir cift1/exp2_3gram_1/todelete \
#     --chunk 32 \
#     --gpu 0 \
#     --left-context-frames 64 \
#     --causal 1 \
#     --transcript-mode fluent \
#     --context-size 4 \
#     --phi-arch vanilla \
#     --phi-type "att;8" \
#     --phi-norm layernorm \
#     --alpha-actv abs \
#     --omega-type Mean \
#     --prune-range 8 \
#     --ent2awe-slope 0.55819859528638738 \
#     --ent2awe-intercept 0 \
#     --targetlen-from approx_tgtlens/data/3gram.arpa

# python cift1/sym_per_awe.py --wblank-dir cift1/exp2_3gram_1/todelete
