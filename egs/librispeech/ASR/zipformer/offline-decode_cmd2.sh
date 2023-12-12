set -eou pipefail

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
                --epoch $epoch \
                --avg $avg \
                --gpu $gpu \
                --use-averaged-model 1 \
                --exp-dir $exp_dir \
                --res-dir $res_dir/gs_chunk$chunk \
                --max-duration 600 \
                --chunk-size $chunk \
                --decoding-method greedy_search 
        done
    done 
done
