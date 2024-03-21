set -eou pipefail

# exp_dir=$1
exp_dir=zipformer/exp_causal_libri2
epoch=30
setup=$(find "$exp_dir" -maxdepth 1 -type f -name "*.txt")

./zipformer/offline-decode_cmd2.sh \
    "32" "4" "4 8 12 16 20 24" 0 $exp_dir/paper5_ep$epoch $exp_dir $setup $epoch &
./zipformer/offline-decode_cmd2.sh \
    "32" "4" "5 9 13 17 21 25" 1 $exp_dir/paper5_ep$epoch $exp_dir $setup $epoch &
./zipformer/offline-decode_cmd2.sh \
    "32" "4" "6 10 14 18 22 26" 2 $exp_dir/paper5_ep$epoch $exp_dir $setup $epoch &
./zipformer/offline-decode_cmd2.sh \
    "32" "4" "7 11 15 19 23 27" 3 $exp_dir/paper5_ep$epoch $exp_dir $setup $epoch &
wait

python local/get_cer_ci.py --res-dir $exp_dir/paper5_ep$epoch

python local/tabulate_cer.py -i $exp_dir/paper5_ep$epoch -o $exp_dir/paper5_ep$epoch/results.csv

python local/get_best_cer_ci.py -i $exp_dir/paper5_ep$epoch/results.csv -o $exp_dir/paper5_ep$epoch/best.csv -t 10

python notify_tg.py "$exp_dir $epoch epochs decoded"

exp_dir=zipformer/exp_causal_libri0
epoch=30
setup=$(find "$exp_dir" -maxdepth 1 -type f -name "*.txt")

./zipformer/offline-decode_cmd2.sh \
    "32" "4" "4 8 12" 0 $exp_dir/paper5_ep$epoch $exp_dir $setup $epoch &
./zipformer/offline-decode_cmd2.sh \
    "32" "4" "16 20 24" 1 $exp_dir/paper5_ep$epoch $exp_dir $setup $epoch &
./zipformer/offline-decode_cmd2.sh \
    "32" "4" "6 10 14" 2 $exp_dir/paper5_ep$epoch $exp_dir $setup $epoch &
./zipformer/offline-decode_cmd2.sh \
    "32" "4" "18 22 26" 3 $exp_dir/paper5_ep$epoch $exp_dir $setup $epoch &
wait

python local/get_cer_ci.py --res-dir $exp_dir/paper5_ep$epoch

python local/tabulate_cer.py -i $exp_dir/paper5_ep$epoch -o $exp_dir/paper5_ep$epoch/results.csv

python local/get_best_cer_ci.py -i $exp_dir/paper5_ep$epoch/results.csv -o $exp_dir/paper5_ep$epoch/best.csv -t 10

python notify_tg.py "$exp_dir $epoch epochs decoded"