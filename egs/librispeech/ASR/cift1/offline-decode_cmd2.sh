gpu=$4
res_dir=$5
exp_dir=$6
setup=$7
epoch=$8
args=$(sed '/^##/,$d' $setup)

for chunk in $1; do
    for beam in $2; do
        for avg in $3; do
            echo "$args" | xargs -a - python cift1/decode2.py \
                --exp-dir $exp_dir \
                --epoch $epoch \
                --avg $avg \
                --max-duration 700 \
                --decoding-method beam_search \
                --manifest-dir data/fbank \
                --lang data/lang_bpe_500 \
                --max-sym-per-frame 9 \
                --res-dir $res_dir/bs_chunk"$chunk"_beam"$beam" \
                --decode-chunk-len $chunk \
                --beam-size $beam \
                --gpu $gpu \
                --pad-feature 30 
        done
    done
done || python notify_tg.py "Decoding $gpu gone wrong"


python notify_tg.py "Decoding $gpu done."