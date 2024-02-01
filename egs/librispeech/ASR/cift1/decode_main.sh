set -eou pipefail

exp_dir=$1
epoch=30
setup=$(find "$exp_dir" -maxdepth 1 -type f -name "*.txt")
max_sym_range=$2

./cift2/offline-decode_cmd2.sh \
    "64" "4" "10 18" 0 $exp_dir/paper2_ep$epoch $exp_dir $setup 30 "$max_sym_range" &
./cift2/offline-decode_cmd2.sh \
    "64" "4" "6 14" 1 $exp_dir/paper2_ep$epoch $exp_dir $setup 30 "$max_sym_range" &
./cift2/offline-decode_cmd2.sh \
    "64" "4" "4 12" 2 $exp_dir/paper2_ep$epoch $exp_dir $setup 30 "$max_sym_range" &
./cift2/offline-decode_cmd2.sh \
    "64" "4" "8 16" 3 $exp_dir/paper2_ep$epoch $exp_dir $setup 30 "$max_sym_range" &
wait

python local/tabulate_cer.py -i $exp_dir/paper2_ep$epoch -o $exp_dir/paper2_ep$epoch/results.csv

python local/get_best.py -i $exp_dir/paper2_ep$epoch/results.csv -o $exp_dir/paper2_ep$epoch/best.csv -t 10

python notify_tg.py "$exp_dir $epoch epochs decoded"
