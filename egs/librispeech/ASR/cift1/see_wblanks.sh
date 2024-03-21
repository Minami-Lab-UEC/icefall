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
    --lang data/lang_bpe_500 \
    --max-sym-per-frame $maxsym \
    --res-dir $exp_dir/wblanks_avg"$avg"_sym"$maxsym" \
    --chunk 32 \
    --gpu 0 \
    --left-context-frames 64 \
    --causal 1

python cift1/sym_per_awe.py --wblank-dir $exp_dir/wblanks_avg"$avg"_sym"$maxsym"

python cift1/word_seg_analysis.py --wblank-dir $exp_dir/wblanks_avg"$avg"_sym"$maxsym"
