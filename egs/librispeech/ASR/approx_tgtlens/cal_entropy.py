import argparse
from pathlib import Path
from kenlmwrapper import Model

parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--tokenized-texts",
    type=Path
)
parser.add_argument(
    "-o",
    "--output-dir",
    type=Path,
)
parser.add_argument(
    "--lm-dir",
    type=Path,    
)
args = parser.parse_args()

for arpa in args.lm_dir.glob("*.arpa"):
    lm = Model(arpa.as_posix())
    with args.tokenized_texts.open('r') as fin, \
        (args.output_dir / f"{arpa.stem}.txt").open('w') as fout:
        
        for line in fin:
            line = line.strip()
            fout.write(f"{-lm.score(line, bos=False, eos=False):.3f}\n")
