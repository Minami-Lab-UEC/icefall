from pathlib import Path
import argparse

ngram_orders = [1,2,3,4,5]

parser = argparse.ArgumentParser()

parser.add_argument(
    "-o",
    "--output-dir",
    type=Path,
)
args = parser.parse_args()

fields = [
    "word_count",
    "token_count",
    "frame_count"
]
fields += [f"{n}gram" for n in ngram_orders]

fins = [
    (args.output_dir / f"{f}.txt").open('r') for f in fields
]

with (args.output_dir / "combined_results.tsv").open('w') as fout:
    fout.write("index")
    for f in fields:
        fout.write(f"\t{f}")
    fout.write("\n")
    
    for i, lines in enumerate(zip(*fins)):
        fout.write(f"{i}")
        for line in lines:
            fout.write(f"\t{line.strip()}")
        fout.write("\n")

for fin in fins:
    fin.close()

