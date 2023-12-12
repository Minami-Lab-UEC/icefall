set -eou pipefail


exp_dir=$1
epoch=40
setup=$(find "$exp_dir" -maxdepth 1 -type f -name "*.txt")
# setup="${setup#zipformer/exp3_}"
./zipformer/offline-decode_cmd2.sh \
    "64" "4" "9 17 25 33" 0 $exp_dir/paper_ep$epoch $exp_dir $setup $epoch &
./zipformer/offline-decode_cmd2.sh \
    "64" "4" "2 10 18 26 34" 1 $exp_dir/paper_ep$epoch $exp_dir $setup $epoch &
./zipformer/offline-decode_cmd2.sh \
    "64" "4" "3 11 19 27 35" 2 $exp_dir/paper_ep$epoch $exp_dir $setup $epoch &
./zipformer/offline-decode_cmd2.sh \
    "64" "4" "4 12 20 28 36" 3 $exp_dir/paper_ep$epoch $exp_dir $setup $epoch &
./zipformer/offline-decode_cmd2.sh \
    "64" "4" "5 13 21 29 37" 4 $exp_dir/paper_ep$epoch $exp_dir $setup $epoch &
./zipformer/offline-decode_cmd2.sh \
    "64" "4" "6 14 22 30 38" 5 $exp_dir/paper_ep$epoch $exp_dir $setup $epoch &
./zipformer/offline-decode_cmd2.sh \
    "64" "4" "7 15 23 31 39" 6 $exp_dir/paper_ep$epoch $exp_dir $setup $epoch &
./zipformer/offline-decode_cmd2.sh \
    "64" "4" "8 16 24 32" 7 $exp_dir/paper_ep$epoch $exp_dir $setup $epoch &
wait

python local/tabulate_cer.py -i $exp_dir/paper_ep$epoch -o $exp_dir/paper_ep$epoch/results.csv

python notify_tg.py "$exp_dir $epoch epochs decoded"

# python -m pdb zipformer/decode.py \
#     --exp-dir $exp_dir \
#     --epoch 40 \
#     --avg 20 \
#     --use-averaged-model 1 \
#     --max-duration 700 \
#     --decoding-method greedy_search \
#     --manifest-dir /mnt/host/corpus/csj/fbank3 \
#     --lang data/lang_char \
#     --transcript-mode disfluent \
#     --res-dir $exp_dir/todelete/gs_chunk64_beam4 \
#     --chunk-size 64 \
#     --beam-size 4 \
#     --left-context-frames 128 \
#     --gpu 0 \
#     --pad-feature 0 