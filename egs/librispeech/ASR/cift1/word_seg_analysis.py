import argparse
from tabulate import tabulate
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from confidence_intervals import evaluate_with_conf_int

def get_args():
    parser = argparse.ArgumentParser(
        description=(
            "This script generates the raw data for "
            "Figure 1 - Distribution of AWEs based on their relevance to word segmentation."
        )
    )
    
    parser.add_argument(
        "--wblank-dir",
        type=Path,
    )
    return parser.parse_args()

def metrics(wer):
    return np.sum(wer)

def str_ci(ci, tot) -> str:
    return f"{ci[0]/tot:.5f}±{(ci[1][1]-ci[1][0])/(2*tot) :.5f}"

def main():
    args = get_args()
    wblank_dir : Path = args.wblank_dir
    # wblank_dir = Path("cift1/exp1_mean@_2gram_nwords_1/wblanks_avg8_sym8")

    tot_single_word = []
    tot_gt1_word = []
    tot_empty_word = []
    tot_start_word = []
    tot_continue_word = []
    tot_invalid_word = []
    tot_awes = []

    # Gather stats
    for wblank in wblank_dir.glob("wblank-*.txt"):
        with wblank.open() as fin:
            for line in fin:
                line = line.strip().split("hyp=")[1]
                awe_count = line.count("∅")
                line = [awe.strip() for awe in line.split("∅")[:-1]]
                if line[0] == "▁":
                    line[0] = ""

                next_awe_is_word_list = [
                    awe.startswith("▁") for awe in [a for a in line if a][1:]
                ] + [True]
                
                single_word = 0
                gt1_word = 0
                empty_word = 0
                start_word = 0
                continue_word = 0
                invalid_word = 0
                for awe in line:
                    if not awe:
                        empty_word += 1
                        continue
                    next_awe_is_word = next_awe_is_word_list.pop(0)
                    
                    if "▁" not in awe:
                        continue_word += 1
                    elif not awe.startswith("▁"):
                        invalid_word += 1
                    elif awe.count("▁") == 1:
                        single_word += 1
                    elif next_awe_is_word:
                        gt1_word += 1
                    else:
                        start_word += 1
                tot_awes.append(awe_count)
                tot_single_word.append(single_word)
                tot_gt1_word.append(gt1_word)
                tot_empty_word.append(empty_word)
                tot_start_word.append(start_word)
                tot_continue_word.append(continue_word)
                tot_invalid_word.append(invalid_word)
                
                assert not next_awe_is_word_list, next_awe_is_word_list
    
    tot_single_word = np.array(tot_single_word)
    tot_gt1_word = np.array(tot_gt1_word)
    tot_empty_word = np.array(tot_empty_word)
    tot_start_word = np.array(tot_start_word)
    tot_continue_word = np.array(tot_continue_word)
    tot_invalid_word = np.array(tot_invalid_word)
    tot_awes = np.array(tot_awes)
    
    res_single_word = evaluate_with_conf_int(tot_single_word, metrics)
    res_gt1_word = evaluate_with_conf_int(tot_gt1_word, metrics)
    res_empty_word = evaluate_with_conf_int(tot_empty_word, metrics)
    res_start_word = evaluate_with_conf_int(tot_start_word, metrics)
    res_continue_word = evaluate_with_conf_int(tot_continue_word, metrics)
    res_invalid_word = evaluate_with_conf_int(tot_invalid_word, metrics)

    tot_awes = tot_awes.sum()

    with open(wblank_dir / "stats_confiinter.tsv", 'w') as fout:
        fout.write(str_ci(res_single_word, tot_awes) + ",")
        fout.write(str_ci(res_gt1_word, tot_awes) + ",")
        fout.write(str_ci(res_empty_word, tot_awes) + ",")
        fout.write(str_ci(res_start_word, tot_awes) + ",")
        fout.write(str_ci(res_continue_word, tot_awes) + ",")
        fout.write(str_ci(res_invalid_word, tot_awes))
    
if __name__ == "__main__":
    main()
