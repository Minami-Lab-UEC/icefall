set -eou pipefail


exp_dir=$1
epoch=40
setup=$(find "$exp_dir" -maxdepth 1 -type f -name "*.txt")
# setup="${setup#zipformer/exp3_}"
# ./zipformer/offline-decode_cmd2.sh \
#     "64" "4" "9 17 25 33" 0 $exp_dir/paper_ep$epoch $exp_dir $setup $epoch &
# ./zipformer/offline-decode_cmd2.sh \
#     "64" "4" "2 10 18 26 34" 1 $exp_dir/paper_ep$epoch $exp_dir $setup $epoch &
# ./zipformer/offline-decode_cmd2.sh \
#     "64" "4" "3 11 19 27 35" 2 $exp_dir/paper_ep$epoch $exp_dir $setup $epoch &
# ./zipformer/offline-decode_cmd2.sh \
#     "64" "4" "4 12 20 28 36" 3 $exp_dir/paper_ep$epoch $exp_dir $setup $epoch &
# ./zipformer/offline-decode_cmd2.sh \
#     "64" "4" "5 13 21 29 37" 4 $exp_dir/paper_ep$epoch $exp_dir $setup $epoch &
# ./zipformer/offline-decode_cmd2.sh \
#     "64" "4" "6 14 22 30 38" 5 $exp_dir/paper_ep$epoch $exp_dir $setup $epoch &
# ./zipformer/offline-decode_cmd2.sh \
#     "64" "4" "7 15 23 31 39" 6 $exp_dir/paper_ep$epoch $exp_dir $setup $epoch &
# ./zipformer/offline-decode_cmd2.sh \
#     "64" "4" "8 16 24 32" 7 $exp_dir/paper_ep$epoch $exp_dir $setup $epoch &
# wait

./zipformer/offline-decode_cmd2.sh \
    "32" "4" "4 8 12 16 20" 0 $exp_dir/paper5_ep$epoch $exp_dir $setup $epoch &
./zipformer/offline-decode_cmd2.sh \
    "32" "4" "5 9 13 17 21" 1 $exp_dir/paper5_ep$epoch $exp_dir $setup $epoch &
./zipformer/offline-decode_cmd2.sh \
    "32" "4" "6 10 14 18 22" 2 $exp_dir/paper5_ep$epoch $exp_dir $setup $epoch &
./zipformer/offline-decode_cmd2.sh \
    "32" "4" "7 11 15 19 23" 3 $exp_dir/paper5_ep$epoch $exp_dir $setup $epoch &
wait

python local/get_cer_ci.py --res-dir $exp_dir/paper5_ep$epoch

python local/tabulate_cer.py -i $exp_dir/paper5_ep$epoch -o $exp_dir/paper5_ep$epoch/results.csv

python local/get_best_cer_ci.py -i $exp_dir/paper5_ep$epoch/results.csv -o $exp_dir/paper5_ep$epoch/best.csv -t 10

best=$(head -n 1 $exp_dir/paper5_ep$epoch/best.csv)
avg=$(echo $best | awk -F',' '{print $2}')
maxsym=$(echo $best | awk -F',' '{print $4}')

# ./cift1/see_wblanks.sh $exp_dir $avg $maxsym


python notify_tg.py "$exp_dir $epoch epochs decoded. Best: 
$best"
