set -eou pipefail

# exp_dir=$1
exp_dir=$1
epoch=$2
# setup=$(find "$exp_dir" -maxdepth 1 -type f -name "*.txt")

./zipformer/decode_cmd.sh \
    "32" "4" "4 8 12 16 20 24" 0 $exp_dir/paper5_ep$epoch $exp_dir $epoch &
./zipformer/decode_cmd.sh \
    "32" "4" "5 9 13 17 21 25" 1 $exp_dir/paper5_ep$epoch $exp_dir $epoch &
./zipformer/decode_cmd.sh \
    "32" "4" "6 10 14 18 22 26" 2 $exp_dir/paper5_ep$epoch $exp_dir $epoch &
./zipformer/decode_cmd.sh \
    "32" "4" "7 11 15 19 23 27" 3 $exp_dir/paper5_ep$epoch $exp_dir $epoch &
wait

python teo_local/get_cer_ci2.py --res-dir $exp_dir/paper5_ep$epoch

python teo_local/tabulate_cer.py -i $exp_dir/paper5_ep$epoch -o $exp_dir/paper5_ep$epoch/results.csv

python teo_local/get_best_cer_ci.py -i $exp_dir/paper5_ep$epoch/results.csv -o $exp_dir/paper5_ep$epoch/best.csv -t 10

best=$(head -n 1 $exp_dir/paper5_ep$epoch/best.csv)
avg=$(echo $best | awk -F',' '{print $2}')
maxsym=$(echo $best | awk -F',' '{print $4}')

python teo_local/notify_tg.py "$exp_dir $epoch epochs decoded. Best: 
$best"
