set -eou pipefail

beam=4
epoch=40

exp_dir=zipformer/exp_f_causal
best=$(head -n 1 $exp_dir/paper5_ep$epoch/best.csv)
avg=$(echo $best | awk -F',' '{print $2}')
setup=$(find "$exp_dir" -maxdepth 1 -type f -name "*.txt")
res_dir="$exp_dir"/ctxtsweep
./zipformer/decode_cmd.sh \
    "16" "4" $avg "64 128 256" 0 $res_dir $exp_dir $setup $epoch &
./zipformer/decode_cmd.sh \
    "32" "4" $avg "64 128 256" 1 $res_dir $exp_dir $setup $epoch &
./zipformer/decode_cmd.sh \
    "64" "4" $avg "64 128 256" 2 $res_dir $exp_dir $setup $epoch &
./zipformer/decode_cmd.sh \
    "128" "4" $avg "64 128 256" 3 $res_dir $exp_dir $setup $epoch &
wait

python local_teo/get_cer_ci.py --res-dir $res_dir --subparts eval1 eval2 eval3 excluded valid

python local_teo/tabulate_cer.py -i $res_dir -o $res_dir/results.csv

python local_teo/get_best_cer_ci.py -i $res_dir/results.csv -o $res_dir/best.csv -t 10

python local_teo/notify_tg.py "Done sweeping $exp_dir"


