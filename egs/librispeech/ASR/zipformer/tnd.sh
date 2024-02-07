# python -m pdb zipformer/train.py \
#   --world-size 8 \
#   --num-epochs 40 \
#   --start-epoch 1 \
#   --use-fp16 1 \
#   --exp-dir zipformer/exp \
#   --causal 0 \
#   --full-libri 1 \
#   --max-duration 1000

python zipformer/train.py \
  --world-size 4 \
  --exp-dir zipformer/exp_causal_libri2 \
  --num-epochs 30 \
  --start-epoch 1 \
  --use-fp16 1 \
  --max-duration 850 \
  --musan-dir /mnt/host/corpus/musan/musan/fbank \
  --causal 1 \
  --full-libri 2 || { python notify_tg.py "zipformer : Something wrong during training." ; exit 1; }

python notify_tg.py "zipformer/exp_causal_libri2 Training done."

# ./zipformer/decode_main8_full.sh zipformer/exp_causal_libri2

cd /mnt/host/icefall/egs/librispeech/ASR
./cift2/tnd.sh || { python notify_tg.py "Something went wrong during cift2/tnd.sh training." ; exit 1; }
