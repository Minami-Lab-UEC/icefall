set -eou pipefail

gpu=$5
res_dir=$6
exp_dir=$7
setup=$8
epoch=$9
args=$(sed '/^##/,$d' $setup)

for chunk in $1; do
    for beam in $2; do
        for avg in $3; do
            for leftcontext in $4; do
                echo "$args" | xargs -a - python zipformer/decode.py \
                    --exp-dir $exp_dir \
                    --epoch $epoch \
                    --avg $avg \
                    --use-averaged-model 1 \
                    --max-duration 1500 \
                    --decoding-method modified_beam_search \
                    --manifest-dir data/fbank \
                    --lang data/lang_bpe_500 \
                    --res-dir $res_dir/mbs_chunk"$chunk"_beam"$beam" \
                    --chunk-size $chunk \
                    --beam-size $beam \
                    --gpu $gpu \
                    --left-context-frames $leftcontext \
                    --pad-feature 30
            done
        done
    done
done || python notify_tg.py "Decoding $gpu gone wrong"


# python notify_tg.py "Decoding $gpu done."
