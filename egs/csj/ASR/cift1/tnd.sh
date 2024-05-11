#!/bin/bash

set -eou pipefail

# python -m pdb cift1/train_frame.py \
#     --world-size 1 \
#     --exp-dir cift1/todelete \
#     --num-epochs 30 \
#     --start-epoch 1 \
#     --use-fp16 1 \
#     --max-duration 850 \
#     --lang data/lang_char \
#     --manifest-dir data/fbank \
#     --musan-dir /mnt/host/corpus/musan/fbank \
#     --causal 1 \
#     --transcript-mode fluent \
#     --context-size 4 \
#     --phi-arch vanilla \
#     --phi-type "att;8" \
#     --phi-norm layernorm \
#     --alpha-actv abs \
#     --omega-type Mean \
#     --prune-range 16 \
#     --ent2awe-slope 0.14512503892360881 \
#     --ent2awe-intercept 0

# file=cift1/configs/nframes.txt
# setup="${file%.txt}"
# setup="${setup#cift1/configs/}"
# args=$(sed '/^##/,$d' $file)
# exp_dir=exp1_"$setup"_1

# echo "$args" | xargs -a - python cift1/train_nframe.py \
#     --world-size 4 \
#     --exp-dir cift1/$exp_dir \
#     --num-epochs 30 \
#     --start-epoch 1 \
#     --use-fp16 1 \
#     --max-duration 850 \
#     --lang data/lang_char \
#     --manifest-dir data/fbank \
#     --musan-dir /mnt/host/corpus/musan/fbank \
#     --causal 1 \
#     --transcript-mode fluent \
#     --telegram-cred misc.ini || { python notify_tg.py "$exp_dir : Something wrong during training." ; exit 1; }

# python notify_tg.py "cift1/$exp_dir Training done."

# mv $file cift1/$exp_dir/

# ./cift1/decode_main.sh cift1/$exp_dir "4 5 6 7 8 9" || python notify_tg.py "Decoding went wrong"

# file=cift1/configs/nwords.txt
# setup="${file%.txt}"
# setup="${setup#cift1/configs/}"
# args=$(sed '/^##/,$d' $file)
# exp_dir=exp1_"$setup"_1

# echo "$args" | xargs -a - python cift1/train_nword.py \
#     --world-size 4 \
#     --exp-dir cift1/$exp_dir \
#     --num-epochs 30 \
#     --start-epoch 1 \
#     --use-fp16 1 \
#     --max-duration 850 \
#     --lang data/lang_char \
#     --manifest-dir data/fbank \
#     --musan-dir /mnt/host/corpus/musan/fbank \
#     --causal 1 \
#     --transcript-mode fluent \
#     --telegram-cred misc.ini || { python notify_tg.py "$exp_dir : Something wrong during training." ; exit 1; }

# python notify_tg.py "cift1/$exp_dir Training done."

# mv $file cift1/$exp_dir/

# ./cift1/decode_main.sh cift1/$exp_dir "4 5 6 7 8 9" || python notify_tg.py "Decoding went wrong"



# ./cift1/decode_main.sh cift1/exp1_2gram_1 "4 5 6 7 8 9"

# for file in cift1/configs/*.txt ; do
#     setup="${file%.txt}"
#     setup="${setup#cift1/configs/}"
#     args=$(sed '/^##/,$d' $file)
#     exp_dir=exp2_"$setup"_1

#     echo "$args" | xargs -a - python cift1/train.py \
#         --world-size 4 \
#         --exp-dir cift1/$exp_dir \
#         --num-epochs 30 \
#         --start-epoch 1 \
#         --use-fp16 1 \
#         --max-duration 850 \
#         --lang data/lang_char \
#         --manifest-dir data/fbank \
#         --musan-dir /mnt/host/corpus/musan/fbank \
#         --causal 1 \
#         --transcript-mode fluent \
#         --telegram-cred misc.ini || { python notify_tg.py "$exp_dir : Something wrong during training." ; exit 1; }

#     python notify_tg.py "cift1/$exp_dir Training done."

#     mv $file cift1/$exp_dir/

#     ./cift1/decode_main.sh cift1/$exp_dir "7 8 9"

# done


# for file in cift1/exp2*/*.txt ; do 
#     setup="${file%.txt}"
#     setup="${setup#cift1/exp2*/}"
#     args=$(sed '/^##/,$d' $file)
#     exp_dir=exp2_"$setup"_1
#     echo "$args" | xargs -a - python cift1/train.py \
#         --world-size 4 \
#         --exp-dir cift1/$exp_dir \
#         --num-epochs 10 \
#         --start-epoch 31 \
#         --use-fp16 1 \
#         --max-duration 850 \
#         --lang data/lang_char \
#         --manifest-dir data/fbank \
#         --musan-dir /mnt/host/corpus/musan/fbank \
#         --causal 1 \
#         --transcript-mode fluent \
#         --telegram-cred misc.ini || { python notify_tg.py "$exp_dir : Something wrong during training." ; exit 1; }

#     python notify_tg.py "cift1/$exp_dir Training done."
# done
# epoch=40
# for exp_dir in cift1/exp2* ; do
#     if [ $exp_dir = cift1/exp2_1gram_1 ] ; then 
#         echo $exp_dir
#     else
#         best=$(head -n 1 $exp_dir/paper5_ep$epoch/best.csv)
#         avg=$(echo $best | awk -F',' '{print $2}')
#         maxsym=$(echo $best | awk -F',' '{print $4}')
#         ./cift1/see_wblanks.sh $exp_dir $avg $maxsym
#     fi
# done

file=cift1/configs/nframes.txt
setup="${file%.txt}"
setup="${setup#cift1/configs/}"
args=$(sed '/^##/,$d' $file)
exp_dir=exp2_"$setup"_1

echo "$args" | xargs -a - python cift1/train.py \
    --world-size 4 \
    --exp-dir cift1/$exp_dir \
    --num-epochs 40 \
    --start-epoch 1 \
    --use-fp16 1 \
    --max-duration 850 \
    --lang data/lang_char \
    --manifest-dir data/fbank \
    --musan-dir /mnt/host/corpus/musan/fbank \
    --causal 1 \
    --transcript-mode fluent \
    --telegram-cred misc.ini || { python notify_tg.py "$exp_dir : Something wrong during training." ; exit 1; }

python notify_tg.py "cift1/$exp_dir Training done."

mv $file cift1/$exp_dir/

./cift1/decode_main.sh cift1/$exp_dir "6 7 8 9" 40
