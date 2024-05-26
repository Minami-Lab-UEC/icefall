set -eou pipefail

gpu=$5
res_dir=$6
exp_dir=$7
epoch=$8

for chunk in $1; do
    for beam in $2; do
        for avg in $3; do
            for leftcontext in $4; do
                python zipformer/decode.py \
                    --exp-dir $exp_dir \
                    --epoch $epoch \
                    --avg $avg \
                    --use-averaged-model 1 \
                    --max-duration 1500 \
                    --decoding-method modified_beam_search \
                    --manifest-dir data/fbank \
                    --lang data/lang_char \
                    --res-dir $res_dir/mbs_beam"$beam"_chunk"$chunk"_left"$leftcontext"_max1 \
                    --chunk-size $chunk \
                    --beam-size $beam \
                    --left-context-frames $leftcontext \
                    --gpu $gpu \
                    --causal 1 \
                    --pad-feature 0 \
                    --transcript-mode fluent
            done
        done
    done
done || python local_teo/notify_tg.py "Decoding $gpu gone wrong"


python local_teo/notify_tg.py "Decoding $gpu done."