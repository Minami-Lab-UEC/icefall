#!/bin/bash

set -eou pipefail

delete_checkpoints() {
    local exp_dir=$1
    local avg=$2
    let mid_epoch=40-$avg
    let mid_epoch_minus1=$mid_epoch-1
    let mid_epoch_plus1=$mid_epoch+1
    rm -r $exp_dir/checkpoint-*.pt
    # rm -r zipformer/$exp_dir/epoch-{1..$((mid_epoch-1))}.pt
    # rm -r zipformer/$exp_dir/epoch-{$((mid_epoch+1))..39}.pt
    # rm "$exp_dir/epoch-{1..$mid_epoch_minus1}.pt"
    for i in `seq 1 $mid_epoch_minus1`; do
        rm $exp_dir/epoch-$i.pt
    done
    for i in `seq $mid_epoch_plus1 39`; do
        rm $exp_dir/epoch-$i.pt
    done
    echo "Deleted for $exp_dir"
}

# delete_checkpoints zipformer/exp_disf 19
# delete_checkpoints zipformer/exp_disf_causal 25
# delete_checkpoints zipformer/exp_f 26
# delete_checkpoints zipformer/exp_f_causal 19
# delete_checkpoints zipformer/exp_num 24
# delete_checkpoints zipformer/exp_num_causal 15
# delete_checkpoints zipformer/voicecompass_f_causal 28
# delete_checkpoints zipformer/voicecompass_fc_rever 25
delete_checkpoints zipformer/voicecompass_fc_volume_rever 26
