import argparse
from tabulate import tabulate
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--wblank-dir",
        type=Path,
    )
    return parser.parse_args()

def main():
    args = get_args()
    tot_start_words = 0
    tot_awe_that_nostart_noword_boundary = 0
    tot_num_syms_per_awe = []
    tot_empty_awe = 0
    tot_awes = 0

    # wblank_dir = Path("cift2/exp5_mean@_p_maxsym11_x/see_blanks_avg12_sym10")
    wblank_dir : Path = args.wblank_dir
    for wblank in wblank_dir.glob("wblank-*.txt"):
        with wblank.open() as fin:
            for line in fin:
                line = line.strip().split("hyp=")[1]
                awe_count = line.count("∅")
                line = [awe.strip() for awe in line.split("∅")[:-1]]
                if line[0] == "▁ ":
                    line.pop(0)
                # num_syms_per_awe = []
                for awe in line:
                    aa = awe.strip().count(" ")+1
                    tot_num_syms_per_awe.append(aa)

                tot_start_words += sum(a.startswith("▁") for a in line) 
                tot_empty_awe += sum(not a for a in line)
                tot_awe_that_nostart_noword_boundary += sum((not a.startswith("▁") and "▁" not in a and bool(a)) for a in line)
                tot_awes += awe_count

    histogram = Counter(tot_num_syms_per_awe)
    tot_num_syms_per_awe = np.array(tot_num_syms_per_awe)
    stats = [
        ["Total AWEs", tot_awes],
        ["Empty AWEs", tot_empty_awe],
        ["Empty AWEs (%)", f"{tot_empty_awe/(tot_awes):.3%}"],
        ["AWEs that start words", tot_start_words,],        
        ["AWEs that start words (% of nonempty AWEs)", f"{tot_start_words/(tot_awes-tot_empty_awe):.3%}",],
        ["AWEs without complete words", tot_awe_that_nostart_noword_boundary,],
        ["AWEs without complete words (% of nonstart AWEs)", f"{tot_awe_that_nostart_noword_boundary/(tot_awes-tot_empty_awe-tot_start_words):.3%}",],
        ["---Among AWEs that----", "--produce symbols--"],
        ["Sym per AWE (min)", np.min(tot_num_syms_per_awe)],
        ["Sym per AWE ( 1%)", np.quantile(tot_num_syms_per_awe,0.01)],
        ["Sym per AWE (10%)", np.quantile(tot_num_syms_per_awe,0.1)],
        ["Sym per AWE (25%)", np.quantile(tot_num_syms_per_awe,0.25)],
        ["Sym per AWE (median)", np.median(tot_num_syms_per_awe)],
        ["Sym per AWE (mean)", np.mean(tot_num_syms_per_awe)],
        ["Sym per AWE (75%)", np.quantile(tot_num_syms_per_awe,0.75)],
        ["Sym per AWE (90%)", np.quantile(tot_num_syms_per_awe,0.9)],
        ["Sym per AWE (99%)", np.quantile(tot_num_syms_per_awe,0.99)],
        ["Sym per AWE (max)", np.max(tot_num_syms_per_awe)],
    ]

    tot_num_syms_per_awe = np.concatenate([tot_num_syms_per_awe, np.array([0]*tot_empty_awe)], axis=0)
    values, bins, bars  = plt.hist(tot_num_syms_per_awe, bins=range(0,np.max(tot_num_syms_per_awe)+2), align='left', edgecolor='black', alpha=0.7)
    plt.bar_label(bars, labels=[f"{v/tot_awes:.1%}" for v in values], color='navy')
    plt.xlabel("Tokens per AWE")
    plt.ylabel("Count")
    plt.title("Histogram of Symbol per AWE")
    plt.savefig(wblank_dir / "sym_per_awe_histogram.png")
    
    histogram = sorted(histogram.items(), key=lambda x:x[0])
    stats.append(["----", "----"])
    stats.extend([
        [f"{k} syms", int(v)] for k,v in enumerate(values)
    ])
    
    with (wblank_dir / "stats.txt").open('w') as fout:
        fout.write(f"For {wblank_dir}: \n\n")
        fout.write(tabulate(stats, tablefmt="fancy_grid"))


if __name__ == "__main__":
    main()