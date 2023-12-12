set -eou pipefail

exp_dir=$1
epoch=30
setup=$(find "$exp_dir" -maxdepth 1 -type f -name "*.txt")

./zipformer/offline-decode_cmd2.sh \
    "16" "4" "4 12 20" 0 $exp_dir/paper_ep$epoch $exp_dir $setup $epoch &
./zipformer/offline-decode_cmd2.sh \
    "16" "4" "5 13 21" 1 $exp_dir/paper_ep$epoch $exp_dir $setup $epoch &
./zipformer/offline-decode_cmd2.sh \
    "16" "4" "6 14 22" 2 $exp_dir/paper_ep$epoch $exp_dir $setup $epoch &
./zipformer/offline-decode_cmd2.sh \
    "16" "4" "7 15 23" 3 $exp_dir/paper_ep$epoch $exp_dir $setup $epoch &
./zipformer/offline-decode_cmd2.sh \
    "16" "4" "8 16 24" 4 $exp_dir/paper_ep$epoch $exp_dir $setup $epoch &
./zipformer/offline-decode_cmd2.sh \
    "16" "4" "9 17 25" 5 $exp_dir/paper_ep$epoch $exp_dir $setup $epoch &
./zipformer/offline-decode_cmd2.sh \
    "16" "4" "10 18 26" 6 $exp_dir/paper_ep$epoch $exp_dir $setup $epoch &
./zipformer/offline-decode_cmd2.sh \
    "16" "4" "11 19 27" 7 $exp_dir/paper_ep$epoch $exp_dir $setup $epoch &
wait

python local/tabulate_cer.py -i $exp_dir/paper_ep$epoch -o $exp_dir/paper_ep$epoch/results.csv

python notify_tg.py "$exp_dir 30 epochs decoded"