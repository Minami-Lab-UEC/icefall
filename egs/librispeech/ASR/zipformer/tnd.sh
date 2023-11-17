# python -m pdb zipformer/train.py \
#   --world-size 8 \
#   --num-epochs 40 \
#   --start-epoch 1 \
#   --use-fp16 1 \
#   --exp-dir zipformer/exp \
#   --causal 0 \
#   --full-libri 1 \
#   --max-duration 1000

python -m pdb zipformer/train.py \
  --world-size 1 \
  --num-epochs 1 \
  --start-epoch 1 \
  --use-fp16 1 \
  --musan-dir /mnt/host/corpus/musan/musan/fbank \
  --exp-dir zipformer/exp \
  --full-libri 1 \
  --max-duration 240