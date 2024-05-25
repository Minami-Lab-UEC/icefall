## RUN OUTSIDE DOCKER

from confidence_intervals import evaluate_with_conf_int
from typing import List
import kaldialign
import numpy as np
import argparse
from pathlib import Path
import re

# recogs-dev-clean-beam_size_4-epoch-40-avg-6-chunk-16-left-context-64-beam_search-beam-size-4-max-sym-per-frame-7-use-averaged-model.txt
matcher = re.compile(
    r"recogs-([a-zA-Z0-9_-]+)-beam_size.*epoch-(\d*)-avg-(\d*)-chunk-(\d*)-left-context-(\d*)-.*-beam-size-(\d*)-max-sym-per-frame-(\d*).*"
)

def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--res-dir",
        type=Path,
    )

    return parser.parse_args()

def compute_wer(
    ref : List[str],
    hyp : List[str],
) -> float:
    subs = 0 #: Dict[Tuple[str, str], int] = defaultdict(int)
    ins = 0 #: Dict[str, int] = defaultdict(int)
    dels = 0 #: Dict[str, int] = defaultdict(int)

    ERR = "*"
    ali = kaldialign.align(ref, hyp, ERR)
    for ref_word, hyp_word in ali:
        if ref_word == ERR:
            ins += 1
        elif hyp_word == ERR:
            dels += 1
        elif hyp_word != ref_word:
            subs += 1
    ref_len = len(ref)
    tot_errs = subs + ins + dels
    tot_err_rate = 100.0 * tot_errs / ref_len

    return tot_err_rate

def metrics(weights, wer):
    return np.average(wer, weights=weights)

def read_recogs(file : Path):
    samples = []
    num_words = []
    with open(file) as fin:
        for ref, hyp in zip(fin, fin):
            ref = ref.strip().split("\t")[1]
            hyp = hyp.strip().split("\t")[1]
            ref = eval(ref[4:])
            hyp = eval(hyp[4:])
            samples.append(compute_wer(ref, hyp))
            num_words.append(len(ref))
    return np.array(samples), np.array(num_words)

def get_ci_filename(recogs_filename : str) -> str:
    matches = matcher.findall(recogs_filename)[0] 
    # r"recogs-([a-zA-Z0-9_-]+)-beam_size.*epoch-(\d*)-avg-(\d*)-chunk-(\d*)-left-context-(\d*)-beam-size-(\d*)-max-sym-per-frame-(\d*).*"
    subpart, epoch, avg, chunk, leftcontext, beam, maxsym = matches
    return f"{subpart}-{beam}_{avg}_{epoch}_{maxsym}_{chunk}_{leftcontext}.ci"

def main():
    args = get_args()
    res_dir : Path = args.res_dir
    subparts = ["test-clean", "test-other", "dev-clean", "dev-other"]

    for subpart in subparts:
        for recogs in res_dir.glob(f"*/recogs-{subpart}-*"):
            parent_dir = recogs.parent
            ci_filename = get_ci_filename(recogs.name)
            samples, num_words = read_recogs(recogs)
            ret = evaluate_with_conf_int(samples, metrics, num_words)
            with (parent_dir / ci_filename).open('w') as fout:
                fout.write(f"{ret[0]:.5f}±{(ret[1][1]-ret[1][0])/2 :.2f}")


if __name__ == "__main__":
    main()



