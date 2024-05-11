import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def get_args():
    parser = argparse.ArgumentParser(
        description=(
            "This script generates the raw data for "
            "Figure 2 - Historgram showing the distribution of tokens."
        )
    )
    
    parser.add_argument(
        "--wblank-dir",
        type=Path,
    )
    return parser.parse_args()

def main():
    args = get_args()
    wblank_dir : Path = args.wblank_dir
    
    tot_sym_per_awe = []

    # Gather stats
    for wblank in wblank_dir.glob("wblank-*.txt"):
        with wblank.open() as fin:
            for line in fin:
                line = line.strip().split("hyp=")[1]
                line = [awe.strip() for awe in line.split("∅")[:-1]]
                # if line[0] == "▁":
                #     line[0] = ""

                for awe in line:
                    if not awe:
                        tot_sym_per_awe.append(1)
                        continue
                    num_token = awe.strip().count(" ") + 1
                    tot_sym_per_awe.extend(
                        [num_token + 1] * (num_token + 1)
                    )

    tot_sym_per_awe = np.array(tot_sym_per_awe)
    tot_tokens = tot_sym_per_awe.size

    values, bins, bars = plt.hist(
        tot_sym_per_awe, align='left', edgecolor='black', alpha=0.7,
        bins=range(1, np.max(tot_sym_per_awe)+2)
    )

    # Draw bar chart
    plt.bar_label(bars, labels=[f"{v/tot_tokens:.1%}" for v in values], color='navy')
    plt.xlabel("From AWE that generated X tokens")
    plt.ylabel("Count")
    ax = plt.gca()
    ax.yaxis.set_major_formatter(lambda x, pos: f"{x/1000:.1f}k" if x else x)
    plt.title("Histogram of Tokens per AWE")
    plt.savefig(wblank_dir / "sym_per_awe_histogram.png")

    # One-line csv
    with (wblank_dir / "symbol_per_awe.tsv").open('w') as fout:
        fout.write(','.join([f"{s:.0f}" for s in values]))
        fout.write("\n")

if __name__ == "__main__":
    main()
