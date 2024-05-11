gpu=$4
res_dir=$5
exp_dir=$6
setup=$7
epoch=$8
args=$(sed '/^##/,$d' $setup)

for chunk in $1; do
    for beam in $2; do
        for avg in $3; do
            echo "$args" | xargs -a - python zipformer/decode.py \
                --exp-dir $exp_dir \
                --epoch $epoch \
                --avg $avg \
                --use-averaged-model 1 \
                --max-duration 700 \
                --decoding-method modified_beam_search \
                --manifest-dir data/fbank \
                --lang data/lang_char \
                --res-dir $res_dir/mbs_chunk"$chunk"_beam"$beam" \
                --chunk-size $chunk \
                --beam-size $beam \
                --left-context-frames 64 \
                --gpu $gpu \
                --transcript-mode fluent \
                --pad-feature 0 # change
        done
    done
done || python notify_tg.py "Decoding $gpu gone wrong"


python notify_tg.py "Decoding $gpu done."