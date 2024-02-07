set -eou pipefail

exp_dir=$1
avg=$2
maxsym=$3
setup=$(find "$exp_dir" -maxdepth 1 -type f -name "*.txt")
args=$(sed '/^##/,$d' $setup)
echo "$args" | xargs -a - python cift2/see_blanks.py \
    --exp-dir $exp_dir \
    --epoch 30 \
    --avg $avg \
    --max-duration 1500 \
    --lang data/lang_bpe_500 \
    --max-sym-per-frame $maxsym \
    --res-dir $exp_dir/see_blanks_avg"$avg"_sym"$maxsym" \
    --chunk 32 \
    --gpu 0 \
    --pad-feature 30 \
    --left-context-frames 128 \
    --causal 1


python cift2/blank_stats.py --wblank-dir $exp_dir/see_blanks_avg"$avg"_sym"$maxsym"




# python cift2/blank_stats.py --wblank-dir cift2/exp5_mean@_maxsym5_x/see_blanks_avg12_sym8
# python cift2/blank_stats.py --wblank-dir cift2/exp5_mean@_maxsym8_x/see_blanks_avg12_sym8
# python cift2/blank_stats.py --wblank-dir cift2/exp5_mean@_maxsym11_x/see_blanks_avg12_sym10
# python cift2/blank_stats.py --wblank-dir cift2/exp5_mean@_p_maxsym5_x/see_blanks_avg14_sym5
# python cift2/blank_stats.py --wblank-dir cift2/exp5_mean@_p_maxsym5_x/see_blanks_avg14_sym8
# python cift2/blank_stats.py --wblank-dir cift2/exp5_mean@_p_maxsym8_x/see_blanks_avg12_sym8
# python cift2/blank_stats.py --wblank-dir cift2/exp5_mean@_p_maxsym11_x/see_blanks_avg12_sym10
# python cift2/blank_stats.py --wblank-dir cift2/exp2_mean@_3gram_nwords_x/see_blanks_avg12_sym9
