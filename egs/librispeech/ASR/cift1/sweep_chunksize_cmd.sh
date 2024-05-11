gpu=$5
res_dir=$6
exp_dir=$7
setup=$8
epoch=$9
maxsyms=${10}
args=$(sed '/^##/,$d' $setup)

for chunk in $1; do
    for beam in $2; do
        for avg in $3; do
            for leftcontext in $4 ; do
                for maxsym in $maxsyms; do
                    echo "$args" | xargs -a - python cift1/decode.py \
                        --exp-dir $exp_dir \
                        --epoch $epoch \
                        --avg $avg \
                        --use-averaged-model 1 \
                        --max-duration 1500 \
                        --decoding-method beam_search \
                        --manifest-dir data/fbank \
                        --lang data/lang_bpe_500 \
                        --max-sym-per-frame $maxsym \
                        --res-dir $res_dir/bs_chunk"$chunk"_beam"$beam"_max"$maxsym" \
                        --chunk-size $chunk \
                        --beam-size $beam \
                        --gpu $gpu \
                        --left-context-frames $leftcontext \
                        --causal 1
                done
            done
        done
    done
done || python notify_tg.py "Decoding $gpu gone wrong"


