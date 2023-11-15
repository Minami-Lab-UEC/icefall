set -eou pipefail

exp_dir=$1
epoch=$2
setup=$(find "$exp_dir" -maxdepth 1 -type f -name "*.txt")
./cift1/offline-decode_cmd2.sh \
    "64" "4" "4 20" 0 $exp_dir/paper_ep$epoch $exp_dir $setup $epoch &
./cift1/offline-decode_cmd2.sh \
    "64" "4" "6 22" 1 $exp_dir/paper_ep$epoch $exp_dir $setup 40 &
./cift1/offline-decode_cmd2.sh \
    "64" "4" "8 24" 2 $exp_dir/paper_ep$epoch $exp_dir $setup 40 &
./cift1/offline-decode_cmd2.sh \
    "64" "4" "10 26" 3 $exp_dir/paper_ep$epoch $exp_dir $setup 40 &
./cift1/offline-decode_cmd2.sh \
    "64" "4" "12 28" 4 $exp_dir/paper_ep$epoch $exp_dir $setup 40 &
./cift1/offline-decode_cmd2.sh \
    "64" "4" "14 30" 5 $exp_dir/paper_ep$epoch $exp_dir $setup 40 &
./cift1/offline-decode_cmd2.sh \
    "64" "4" "16 32" 6 $exp_dir/paper_ep$epoch $exp_dir $setup 40 &
./cift1/offline-decode_cmd2.sh \
    "64" "4" "18 34" 7 $exp_dir/paper_ep$epoch $exp_dir $setup 40 &
wait

python local/tabulate_cer.py -i $exp_dir/paper_ep$epoch -o $exp_dir/paper_ep$epoch/results.csv

python notify_tg.py "$exp_dir $epoch epochs decoded"
