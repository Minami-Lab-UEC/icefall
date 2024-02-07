cat <(gunzip -c data/fbank/librispeech_cuts_train-clean-100.jsonl.gz) \
    <(gunzip -c data/fbank/librispeech_cuts_train-clean-360.jsonl.gz) | \
    shuf | gzip -c > data/fbank/librispeech_cuts_train-clean-shuf.jsonl.gz
