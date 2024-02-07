set -eou pipefail

exp_dir=$1
epoch=30
setup=$(find "$exp_dir" -maxdepth 1 -type f -name "*.txt")


# ./zipformer/offline-decode_cmd2.sh \
#     "32" "4" "4 12 20" 0 $exp_dir/paper5_ep$epoch $exp_dir $setup $epoch &
# ./zipformer/offline-decode_cmd2.sh \
#     "32" "4" "5 13 21" 1 $exp_dir/paper5_ep$epoch $exp_dir $setup $epoch &
# ./zipformer/offline-decode_cmd2.sh \
#     "32" "4" "6 14 22" 2 $exp_dir/paper5_ep$epoch $exp_dir $setup $epoch &
# ./zipformer/offline-decode_cmd2.sh \
#     "32" "4" "7 15 23" 3 $exp_dir/paper5_ep$epoch $exp_dir $setup $epoch &
# ./zipformer/offline-decode_cmd2.sh \
#     "32" "4" "8 16 24" 4 $exp_dir/paper5_ep$epoch $exp_dir $setup $epoch &
# ./zipformer/offline-decode_cmd2.sh \
#     "32" "4" "9 17 25" 5 $exp_dir/paper5_ep$epoch $exp_dir $setup $epoch &
# ./zipformer/offline-decode_cmd2.sh \
#     "32" "4" "10 18 26" 6 $exp_dir/paper5_ep$epoch $exp_dir $setup $epoch &
# ./zipformer/offline-decode_cmd2.sh \
#     "32" "4" "11 19 27" 7 $exp_dir/paper5_ep$epoch $exp_dir $setup $epoch &
# wait

# ./zipformer/offline-decode_cmd2.sh \
#     "32" "4" "4 12 20" 0 $exp_dir/paper5_ep$epoch $exp_dir $setup $epoch &
./zipformer/offline-decode_cmd2.sh \
    "32" "4" "2 9 16 23" 1 $exp_dir/paper5_ep$epoch $exp_dir $setup $epoch &
./zipformer/offline-decode_cmd2.sh \
    "32" "4" "3 10 17 24" 2 $exp_dir/paper5_ep$epoch $exp_dir $setup $epoch &
./zipformer/offline-decode_cmd2.sh \
    "32" "4" "4 11 18 25" 3 $exp_dir/paper5_ep$epoch $exp_dir $setup $epoch &
./zipformer/offline-decode_cmd2.sh \
    "32" "4" "5 12 19 26" 4 $exp_dir/paper5_ep$epoch $exp_dir $setup $epoch &
./zipformer/offline-decode_cmd2.sh \
    "32" "4" "6 13 20 27" 5 $exp_dir/paper5_ep$epoch $exp_dir $setup $epoch &
./zipformer/offline-decode_cmd2.sh \
    "32" "4" "7 14 21 28" 6 $exp_dir/paper5_ep$epoch $exp_dir $setup $epoch &
./zipformer/offline-decode_cmd2.sh \
    "32" "4" "8 15 22 29" 7 $exp_dir/paper5_ep$epoch $exp_dir $setup $epoch &
wait

python local/tabulate_cer.py -i $exp_dir/paper5_ep$epoch -o $exp_dir/paper5_ep$epoch/results.csv

python local/get_best.py -i $exp_dir/paper5_ep$epoch/results.csv -o $exp_dir/paper5_ep$epoch/best.csv -t 10

python notify_tg.py "$exp_dir $epoch epochs decoded"
