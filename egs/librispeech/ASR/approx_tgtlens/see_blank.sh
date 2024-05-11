set -eou pipefail

for naive_dir in approx_tgtlens/data/naive/* ; do 
    python approx_tgtlens/sym_per_awe.py --wblank-dir $naive_dir
    python approx_tgtlens/word_seg_analysis.py --wblank-dir $naive_dir
done
