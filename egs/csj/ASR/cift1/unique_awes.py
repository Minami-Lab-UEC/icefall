import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def get_args():
    parser = argparse.ArgumentParser(
        # description=(
        #     "This script generates the raw data for "
        #     "Figure 2 - Historgram showing the distribution of tokens."
        # )
    )
    
    parser.add_argument(
        "--wblank-dir",
        type=Path,
    )
    return parser.parse_args()

def main():
    args = get_args()
    wblank_dir : Path = args.wblank_dir

    unique_words = defaultdict(int)

    # Gather stats
    for wblank in wblank_dir.glob("wblank-*.txt"):
        with wblank.open() as fin:
            for line in fin:
                line = line.strip().split("hyp=")[1]
                line = [awe.strip() for awe in line.split("∅")[:-1]]
                for awe in line:
                    if not awe:
                        continue
                    unique_words[awe] += 1

    unique_words = sorted(unique_words.items(), key=lambda x:x[1], reverse=True)
    with (wblank_dir / "unique_words.txt").open("w") as fout:
        fout.write(f"word,count\n")
        for w, c in unique_words:
            fout.write(f"{w},{c}\n")


if __name__ == "__main__":
    main()