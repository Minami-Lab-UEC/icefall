set -eou pipefail

exp_dir=$1
epoch=$epoch
setup=$(find "$exp_dir" -maxdepth 1 -type f -name "*.txt")

max_sym_range=$2

# ./cift1/offline-decode_cmd2.sh \
#     "32" "4" "4 8 12 16 20 24" 0 $exp_dir/paper5_ep$epoch $exp_dir $setup $epoch "$max_sym_range" &
# ./cift1/offline-decode_cmd2.sh \
#     "32" "4" "5 9 13 17 21 25" 1 $exp_dir/paper5_ep$epoch $exp_dir $setup $epoch "$max_sym_range" &
# ./cift1/offline-decode_cmd2.sh \
#     "32" "4" "6 10 14 18 22 26" 2 $exp_dir/paper5_ep$epoch $exp_dir $setup $epoch "$max_sym_range" &
# ./cift1/offline-decode_cmd2.sh \
#     "32" "4" "7 11 15 19 23 27" 3 $exp_dir/paper5_ep$epoch $exp_dir $setup $epoch "$max_sym_range" &
# wait

./cift1/offline-decode_cmd2.sh \
    "32" "4" "8 12" 0 $exp_dir/paper5_ep$epoch $exp_dir $setup $epoch "$max_sym_range" &
./cift1/offline-decode_cmd2.sh \
    "32" "4" "16 20 " 1 $exp_dir/paper5_ep$epoch $exp_dir $setup $epoch "$max_sym_range" &
./cift1/offline-decode_cmd2.sh \
    "32" "4" "6 10" 2 $exp_dir/paper5_ep$epoch $exp_dir $setup $epoch "$max_sym_range" &
./cift1/offline-decode_cmd2.sh \
    "32" "4" "18 14" 3 $exp_dir/paper5_ep$epoch $exp_dir $setup $epoch "$max_sym_range" &
wait

python local/get_cer_ci.py --res-dir $exp_dir/paper5_ep$epoch

python local/tabulate_cer.py -i $exp_dir/paper5_ep$epoch -o $exp_dir/paper5_ep$epoch/results.csv

python local/get_best_cer_ci.py -i $exp_dir/paper5_ep$epoch/results.csv -o $exp_dir/paper5_ep$epoch/best.csv -t 10

best=$(head -n 1 $exp_dir/paper5_ep$epoch/best.csv)
avg=$(echo $best | awk -F',' '{print $2}')
maxsym=$(echo $best | awk -F',' '{print $4}')

./cift1/see_wblanks.sh $exp_dir $avg $maxsym

python notify_tg.py "$exp_dir $epoch epochs decoded. Best:
$best"

