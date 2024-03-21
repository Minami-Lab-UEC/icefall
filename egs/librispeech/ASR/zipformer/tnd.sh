# python -m pdb zipformer/train.py \
#   --world-size 8 \
#   --num-epochs 40 \
#   --start-epoch 1 \
#   --use-fp16 1 \
#   --exp-dir zipformer/exp \
#   --causal 0 \
#   --full-libri 1 \
#   --max-duration 1000

for i in 2 0 ; do
  exp_dir=zipformer/exp_causal_libri$i
  # python zipformer/train.py \
  #   --world-size 4 \
  #   --exp-dir $exp_dir \
  #   --num-epochs 30 \
  #   --start-epoch 1 \
  #   --use-fp16 1 \
  #   --max-duration 850 \
  #   --musan-dir /mnt/host/corpus/musan/fbank \
  #   --causal 1 \
  #   --full-libri $i || { python notify_tg.py "zipformer : Something wrong during training." ; exit 1; }

  # python notify_tg.py "$exp_dir Training done."

  ./zipformer/decode_main.sh $exp_dir
done


