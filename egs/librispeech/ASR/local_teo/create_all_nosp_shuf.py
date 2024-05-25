from lhotse import CutSet
import argparse
from pathlib import Path
from lhotse.utils import fix_random_seed

def get_args():
    parser = argparse.ArgumentParser(
        description="Create shuffled nosp cutset for clean-100 and clean-360."
    )
    
    parser.add_argument(
        "--cutset-dir",
        type=Path
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    
    return parser.parse_args()

def main():
    args = get_args()
    fix_random_seed(args.seed)
    
    cuts : CutSet = CutSet.from_file(args.cutset_dir / "librispeech_cuts_train-all-shuf.jsonl.gz")
    cuts = cuts.filter(lambda c: "_sp" not in c.id)
    cuts = cuts.shuffle()
    
    cuts.to_file(args.cutset_dir / "librispeech_cuts_train-all-nosp-shuf.jsonl.gz")

if __name__ == "__main__":
    main()
