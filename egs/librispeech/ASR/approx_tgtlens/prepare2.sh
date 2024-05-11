#!/usr/bin/env bash

set -eou pipefail

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

mkdir -p approx_tgtlens/data2

cp -r data/lang_bpe_500/unigram_500.* approx_tgtlens/data2

log "Gathering training data2"

python approx_tgtlens/get_text_and_xlens2.py \
    -i data/fbank/librispeech_cuts_train-all-shuf.jsonl.gz \
    -o approx_tgtlens/data2 \
    --bpe approx_tgtlens/data2/unigram_500.model

log "Training Ngram LMs."

python approx_tgtlens/make_1gram_lm.py -text approx_tgtlens/data2/texts.txt \
    -lm approx_tgtlens/data2/1gram.arpa \
    -verbose 3 \
    -bpe approx_tgtlens/data2/unigram_500.model

for i in {2..5} ; do
    log "Training $i-gram LM."
    python approx_tgtlens/make_kn_lm.py -text approx_tgtlens/data2/texts.txt \
    -lm approx_tgtlens/data2/"$i"gram.arpa \
    -ngram-order $i \
    -verbose 3 \
    -bpe approx_tgtlens/data2/unigram_500.model
done

log "Calculating entropy by Ngram LMs"

python approx_tgtlens/cal_entropy.py -i approx_tgtlens/data2/tokenized_texts.txt \
    -o approx_tgtlens/data2 \
    --lm-dir approx_tgtlens/data2

log "Combining results"

python approx_tgtlens/combine_results.py -o approx_tgtlens/data2

log "Done! Now you can view the results in results.ipynb."
