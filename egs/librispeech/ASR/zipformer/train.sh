# ./zipformer/train.py \
#   --world-size 8 \
#   --num-epochs 40 \
#   --start-epoch 1 \
#   --use-fp16 1 \
#   --exp-dir zipformer/exp \
#   --causal 0 \
#   --full-libri 1 \
#   --max-duration 1000

# ./zipformer/train.py \
#   --world-size 8 \
#   --num-epochs 40 \
#   --start-epoch 1 \
#   --use-fp16 1 \
#   --exp-dir zipformer/exp_causal \
#   --causal 1 \
#   --full-libri 1 \
#   --max-duration 1000

./zipformer/train.py --world-size 8 --num-epochs 30 --start-epoch 1 --use-fp16 1 --exp-dir zipformer/exp_bpe1.5k --full-libri 1 --max-duration 1000 --bpe-model data/lang_bpe_1500/bpe.model

./zipformer/train.py --world-size 8 --num-epochs 30 --start-epoch 1 --use-fp16 1 --exp-dir zipformer/exp_bpe2k --full-libri 1 --max-duration 1000 --bpe-model data/lang_bpe_2000/bpe.model
